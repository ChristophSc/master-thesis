from random import randint
from collections import defaultdict
import torch

def filter_heads_tails(n_ent, train_data, valid_data=None, test_data=None):
    """ Creates filtered set of heads and tails: Returns 2 sparse Tensors for heads and tails with values
        1 if there is a head h for triple (?,r,t) / 1 if there is a tail t for triple (h,r,?)

    Args:
        n_ent ([type]): [description]
        train_data ([type]): [description]
        valid_data ([type], optional): [description]. Defaults to None.
        test_data ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    train_heads, train_rel, train_tails = train_data
    if valid_data:
        valid_heads, valid_rel, valid_tails = valid_data
    else:
        valid_heads = valid_rel = valid_tails = []
    if test_data:
        test_heads, test_rel, test_tails = test_data
    else:
        test_heads = test_rel = test_tails = []
    all_heads = train_heads + valid_heads + test_heads
    all_rel = train_rel + valid_rel + test_rel
    all_tails = train_tails + valid_tails + test_tails
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for h, r, t in zip(all_heads, all_rel, all_tails):
        tails[(h, r)].add(t)
        heads[(t, r)].add(h)
    heads_sp = {}
    tails_sp = {}
    for k in tails.keys():
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                               torch.ones(len(tails[k])), torch.Size([n_ent]))
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                               torch.ones(len(heads[k])), torch.Size([n_ent]))
    return heads_sp, tails_sp


def head_tail_counter(train_data, valid_data, test_data):
    """Counts appearence of (h, t, ?) and (?, r, t) for all triples from training, validation and test data

    Args:
        train_data (torch.tensor): training data with its triples
        valid_data (torch.tensor): validation data with its triples
        test_data (torch.tensor): test data with its triples

    Returns:
        (dict, dict): set with counts for (h, r, ?) and (?, r, t)
    """
    train_heads, train_rel, train_tails = train_data
    valid_heads, valid_rel, valid_tails = valid_data
    test_heads, test_rel, test_tails = test_data

    all_heads = train_heads + valid_heads + test_heads
    all_rel = train_rel + valid_rel + test_rel
    all_tails = train_tails + valid_tails + test_tails
    
    head_rel_count = defaultdict(lambda: 0)
    rel_tail_count = defaultdict(lambda: 0)
    for h, r, t in zip(all_heads, all_rel, all_tails):        
        head_rel_count[(h, r)] += 1
        rel_tail_count[(r, t)] += 1
    return head_rel_count, rel_tail_count





def filter_negatives(heads_neg, relations_neg, tails_neg, true_heads, true_tails):
    heads_neg_filt, rel_neg_filt, tails_neg_filt = [], [], []
    for batch_h, batch_r, batch_t in zip(heads_neg, relations_neg, tails_neg):
        for head, rel, tail in zip(batch_h, batch_r, batch_t):
            h_i = int(head.data.cpu().numpy())
            r_i = int(rel.data.cpu().numpy())
            t_i = int(tail.data.cpu().numpy())
            
            head_exists = (t_i, r_i) in true_heads and true_heads[(t_i, r_i)]._nnz() > 1
            tail_exists = (h_i, r_i) in true_tails and true_tails[(h_i, r_i)]._nnz() > 1
    
            if not head_exists and not tail_exists:
                heads_neg_filt.append(h_i)
                rel_neg_filt.append(r_i)
                tails_neg_filt.append(t_i)
        
    heads_neg_filt =  torch.tensor(heads_neg_filt)
    rel_neg_filt =  torch.tensor(rel_neg_filt)
    tails_neg_filt =  torch.tensor(tails_neg_filt)
    return heads_neg_filt, rel_neg_filt, tails_neg_filt 
    

def get_statistics(gen, dis, heads, relations, tails, heads_neg, relations_neg, tails_neg, heads_filt, tails_filt, print_statistics = False):    
    
    def get_statistics(case, model, head_entities, relations, tail_entities):
        scores = model.mdl.score(head_entities, relations, tail_entities)            
        min_score = torch.min(scores, dim = -1)
        min_score = torch.min(min_score.values, dim = -1).values.item()
        max_score = torch.max(scores, dim = -1)
        max_score =  torch.max(max_score.values, dim = -1).values.item()
        mean_score = torch.mean(scores).item()
        
        if print_statistics:
            print('-----', case, '-----')
            print('min', min_score)
            print('max', max_score)
            print('mean', mean_score)
        return min_score, max_score, mean_score
    
    
    def get_model_statistics(model, pos_head_entities, pos_relations, pos_tail_entities, neg_head_enitities, neg_relations, neg_tail_entities):
        neg_min_score, neg_max_score, neg_mean_score = get_statistics('Negatives', model, neg_head_enitities, neg_relations, neg_tail_entities)
        pos_min_score, pos_max_score, pos_mean_score = get_statistics('Positives', model, pos_head_entities, pos_relations, pos_tail_entities)        
        if print_statistics:
            print('')
        return pos_min_score, neg_max_score
    
    with torch.no_grad():
        # High scores indicate a low probability if a triple to be true   
        heads_neg_filt, rel_neg_filt, tails_neg_filt = filter_negatives(heads_neg, relations_neg, tails_neg, heads_filt, tails_filt)
        if print_statistics:
            print('----------', 'Generator statistics:', '----------')  
        pos_min_score, neg_max_score = get_model_statistics(gen,  heads, relations, tails, heads_neg_filt, rel_neg_filt, tails_neg_filt)  
        if print_statistics:
            print('----------', 'Discriminator statistics:', '----------')  
        get_model_statistics(dis, heads, relations, tails, heads_neg_filt, rel_neg_filt, tails_neg_filt)   
        if print_statistics:
            print('')
        return pos_min_score, neg_max_score
    

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(randint(0, i))
    for ls in lists:
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]


def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    for i in range(n_batch):
        first_idx = int(n_sample * i / n_batch)   # first possible index
        last_idx = int(n_sample * (i + 1) / n_batch) # last possible index (index in dataset) -> last - first = batch size
        # dataset is divided into "n_batch"-batches
        ret = [ls[first_idx:last_idx] for ls in lists] # len(ret)=6: src, rel, dst, src_cand, rel_cand, dst_cand
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]