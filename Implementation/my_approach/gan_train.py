import os
import logging
import datetime
import torch
import matplotlib.pyplot as plt
from random import sample, random
from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import filter_heads_tails, inplace_shuffle, batch_by_num
from config_utils import load_sampler
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from logger_init import logger_init
from select_gpu import select_gpu
from corrupter import BernCorrupterMulti
from random_sampler import RandomSampler
from graph_utils import create_figure
from base_model import BaseModel
from random_sampler import RandomSampler
from uncertainty_sampler import UncertaintySampler_Basic, UncertaintySampler_Advanced
from TrainingProcessLogger import TrainingProcessLogger

# load config and logger, overwrite config with args
config()
overwrite_config_with_args()
timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
log_dir = logger_init("gan_train")
device_name = "cuda:" + str(select_gpu()) if torch.cuda.is_available()  else "cpu"
device = torch.device(device_name)
    
dump_config()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join('data', task_dir, 'train.txt'),
                         os.path.join('data', task_dir, 'valid.txt'),
                         os.path.join('data', task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

# load pretrained models for generator and discriminator
models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx}
gen_config = config()[config().g_config]
dis_config = config()[config().d_config]
gen = models[config().g_config](n_ent, n_rel, gen_config)
dis = models[config().d_config](n_ent, n_rel, dis_config)
gen.load(os.path.join('models', task_dir, gen_config.model_file))
dis.load(os.path.join('models', task_dir, dis_config.model_file))

# load train-, valid- and testdata
train_data = read_data(os.path.join('data', task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join('data', task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join('data', task_dir, 'test.txt'), kb_index)
heads_filt, tails_filt = filter_heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
train_data = [torch.LongTensor(vec) for vec in train_data]

# test link prediction of discriminator model with pretrained model only (not adverarial training)
dis.test_link(test_data, n_ent, heads_filt, tails_filt)

# load Corrupter which creates set of negatives (Neg) from positive triples in KG
corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, config().adv.n_sample)
head, rel, tail = train_data
n_train = len(head)
n_epoch = config().adv.n_epoch
n_batch = config().adv.n_batch
sampler = load_sampler(config().adv.sample_type)
mdl_name = 'gan_dis_' + timestamp + '.mdl'

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
    # return heads_neg, relations_neg, tails_neg
    

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
        
# init variables for training
best_mrr = 0
avg_reward = 0
tp_logger = TrainingProcessLogger('gan_train', n_epoch, config().adv.epoch_per_test) 
global_min_score = None
global_max_score = None



max_score = -1e30
for epoch in range(n_epoch):
    epoch_d_loss = 0
    epoch_reward = 0
    # create set Neg of negative triples 
    head_cand, rel_cand, tail_cand = corrupter.corrupt(head, rel, tail, keep_truth=True)   # TODO: use different technique to corrupt triples -> e.g. Bernoulli Sampling
        
    min_score, cur_max_score = get_statistics(gen, dis, head, rel, tail, head_cand, rel_cand, tail_cand, heads_filt, tails_filt, print_statistics = False)  # TODO: edit?
    if cur_max_score > max_score:
        max_score = cur_max_score
        
    for h, r, t, h_neg, r_neg, t_neg in batch_by_num(n_batch, head, rel, tail, head_cand, rel_cand, tail_cand, n_sample=n_train):
        # h,r,t = indices of heads, relations and tails in batch
        # h_neg, t_neg = indices of heads and relations of negative triples from negative set Neg
        # send corrupted triples from Neg of size "n_batch" to generator
        gen_step = gen.gen_step(h, r, t, h_neg, r_neg, t_neg, n_sample = 1, temperature=config().adv.temperature, train = True, sampler = sampler, min_score = min_score, max_score = max_score)
        # randomly sample from probability distribution of current negative triple set 
        head_smpl, tail_smpl = next(gen_step)
        # send sampled negative triple "tail_smpl" and its ground truth triple "head_smpl" to discriminator 
        losses, rewards = dis.dis_step(h, r, t, head_smpl.squeeze(), tail_smpl.squeeze())
        epoch_reward += torch.sum(rewards)        
        rewards = rewards - avg_reward
        # send reward to generator
        gen_step.send(rewards.unsqueeze(1))
        epoch_d_loss += torch.sum(losses)

    avg_loss = epoch_d_loss / n_train
    avg_reward = epoch_reward / n_train
    tp_logger.log_loss_reward(epoch, avg_loss, avg_reward)     
    logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
    if (epoch + 1) % config().adv.epoch_per_test == 0:
        #gen.test_link(valid_data, n_ent, filt_heads, filt_tails)
        mrr, hits = dis.test_link(valid_data, n_ent, heads_filt, tails_filt)
        tp_logger.log_performance(mrr, hits)
        if mrr > best_mrr:
            best_mrr = mrr
            dis.save(os.path.join('models', config().task.dir, mdl_name))
      
if config().log.log_pretrain_graph:
            tp_logger.create_and_save_figures(log_dir)   

dis.load(os.path.join('models', config().task.dir, mdl_name))
logging.info('Best Performance:')
dis.test_link(test_data, n_ent, heads_filt, tails_filt)
