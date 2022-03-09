import os
import logging
import datetime
import torch
import matplotlib.pyplot as plt
from random import sample, random
from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import filter_heads_tails, get_scoring_statistics, inplace_shuffle, batch_by_num, head_tail_counter, filter_negatives
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
from uncertainty_sampler import *
from TrainingProcessLogger import TrainingProcessLogger
from time import time

# set seeds
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
# torch.use_deterministic_algorithms(True)

# load config and logger, overwrite config with args
config()
overwrite_config_with_args()
timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
log_dir = logger_init("gan_train")
device_name = "cuda:0" if torch.cuda.is_available()  else "cpu" # "cuda:" + str(select_gpu()) if torch.cuda.is_available()  else "cpu"
cur_device = torch.device(device_name)
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
if config().adv.use_pretrained_models:        
    gen.load(os.path.join('models', task_dir, gen_config.model_file))
    dis.load(os.path.join('models', task_dir, dis_config.model_file))

# load train-, valid- and testdata
train_data = read_data(os.path.join('data', task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join('data', task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join('data', task_dir, 'test.txt'), kb_index)
heads_filt, tails_filt = filter_heads_tails(n_ent, train_data, valid_data, test_data)


head_rel_count, rel_tail_count = head_tail_counter(train_data, valid_data, test_data)

valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
train_data = [torch.LongTensor(vec) for vec in train_data]

# load Corrupter which creates set of negatives (Neg) from positive triples in KG
corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, config().adv.n_sample)
head, rel, tail = train_data
n_train = len(head)
n_epoch = config().adv.n_epoch
n_batch = config().adv.n_batch
sampler = load_sampler()
mdl_name = 'gan_dis_' + timestamp + '.mdl'

# init variables for training
max_score = -1e30
min_score = +1e30
best_mrr = 0
avg_reward = 0
avg_loss = 0

tp_logger = TrainingProcessLogger('gan_train', n_epoch, config().adv.epoch_per_test) 

logging.info(datetime.datetime.now())

# test link prediction of discriminator model with pretrained model only (not adverarial training)
mrr, hits = dis.test_link(test_data, n_ent, heads_filt, tails_filt)   
tp_logger.log_performance(mrr, hits)

if torch.cuda.is_available():
    head = head.cuda()
    rel = rel.cuda()           
    tail = tail.cuda()

neg_heads_filt, neg_rel_filt, neg_tails_filt = None, None, None
neg_set = dict()
for epoch in range(n_epoch):
    logging.info('Epoch: ' + str(epoch) + ": " + str(datetime.datetime.now()))
    epoch_d_loss = 0
    epoch_reward = 0
    # create set Neg of negative triples 
    head_cand, rel_cand, tail_cand = corrupter.corrupt(head, rel, tail, keep_truth=False)   # TODO: use different technique to corrupt triples -> e.g. Bernoulli Sampling
    
    if torch.cuda.is_available():
        head_cand  = head_cand.cuda()
        rel_cand  = rel_cand.cuda()
        tail_cand = tail_cand.cuda()

    # get statistics
    if type(sampler) == RandomSampler:
        pos_min_score, pos_max_score, neg_min_score, neg_max_score = None, None, None, None
    else:
        # if neg_heads_filt == None:
            # negatives should be always the same, only init and score them once
            # TODO: update with cache like in NSCaching with efficient method to replace negatives in Neg
        
        neg_set, neg_heads_filt, neg_rel_filt, neg_tails_filt = neg_set, head_cand, rel_cand, tail_cand # filter_negatives(neg_set, head_cand, rel_cand, tail_cand, heads_filt, tails_filt)
        # logging.info(len(neg_heads_filt))
        #logging.info(len(neg_set))
        # logging.info(neg_set)
        pos_min_score, pos_max_score, neg_min_score, neg_max_score =  get_scoring_statistics(gen, dis, head, rel, tail, neg_heads_filt, neg_rel_filt, neg_tails_filt, print_statistics = False)
        #t1 = time()    
        #print('get_scoring_statistics all takes %f' %(t1-t0))
        #logging.info('Score ranges for all positives:')
        logging.info('neg_min_score: ' + str(neg_min_score) + ', neg_max_score: ' +  str(neg_max_score) + ', pos_min_score: ' + str(pos_min_score) + ', pos_max_score: ' + str(pos_max_score))
        
    for h, r, t, h_neg, r_neg, t_neg in batch_by_num(n_batch, head, rel, tail, head_cand, rel_cand, tail_cand, n_sample=n_train):
        # h,r,t = indices of heads, relations and tails in batch
        # h_neg, t_neg = indices of heads and relations of negative triples from negative set Neg
        # send corrupted triples from Neg of size "n_batch" to generator
        gen_step = gen.gen_step(head_rel_count, rel_tail_count, h_neg, r_neg, t_neg, n_sample = 1, temperature=config().adv.temperature, train = True, sampler = sampler, min_score = neg_min_score, max_score = pos_max_score)
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
    if epoch % config().log.graph_every_nth_epoch == 0:   
        tp_logger.log_loss_reward(avg_loss.item(), avg_reward.item())     
    logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
    if (epoch + 1) % config().adv.epoch_per_test == 0:
        #gen.test_link(valid_data, n_ent, filt_heads, filt_tails)
        mrr, hits = dis.test_link(valid_data, n_ent, heads_filt, tails_filt)
        tp_logger.log_performance(mrr, hits)
        if mrr > best_mrr:
            best_mrr = mrr
            dis.save(os.path.join('models', config().task.dir, mdl_name))
tp_logger.log_loss_reward(avg_loss.item(), avg_reward.item())        
if config().log.log_pretrain_graph:
    tp_logger.create_and_save_figures(log_dir)   

logging.info(datetime.now())

dis.load(os.path.join('models', config().task.dir, mdl_name))
logging.info('Best Performance:')
dis.test_link(test_data, n_ent, heads_filt, tails_filt)
