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