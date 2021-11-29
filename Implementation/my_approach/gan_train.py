import os
import logging
import datetime
import torch
import matplotlib.pyplot as plt

from random import sample, random

from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import heads_tails, inplace_shuffle, batch_by_num
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from logger_init import logger_init
from select_gpu import select_gpu
from corrupter import BernCorrupterMulti
from random_sampler import RandomSampler
from graph_utils import create_figure


# load config and logger, overwrite config with args
config()
overwrite_config_with_args()
logger_init("gan_train")
if torch.cuda.is_available():
    torch.cuda.set_device(select_gpu())
dump_config()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join('data', task_dir, 'train.txt'),
                         os.path.join('data', task_dir, 'valid.txt'),
                         os.path.join('data', task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

# load pretrained models for generator and discriminator
models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx}
gen_config = config()[config().g_config]
gen_sampler = RandomSampler()
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
filt_heads, filt_tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
train_data = [torch.LongTensor(vec) for vec in train_data]

# test link prediction of discriminator model with pretrained model only (not adverarial training)
dis.test_link(test_data, n_ent, filt_heads, filt_tails)

# load Corrupter which creates set of negatives (Neg) from positive triples in KG
corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, config().adv.n_sample)
src, rel, dst = train_data
n_train = len(src)
n_epoch = config().adv.n_epoch
n_batch = config().adv.n_batch
create_graphs = config().graph.create_graphs
timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
mdl_name = 'gan_dis_' + timestamp + '.mdl'
best_mrr = 0
avg_reward = 0
logged_rewards = []
logged_D_losses = []
logged_mrrs = []
logged_hit10s = []

for epoch in range(n_epoch):
    epoch_d_loss = 0
    epoch_reward = 0
    # create set Neg of negative triples 
    src_cand, rel_cand, dst_cand = corrupter.corrupt(src, rel, dst, keep_truth=False)   # TODO: use different technique to corrupt triples -> e.g. Bernoulli Sampling
    for s, r, t, ss, rs, ts in batch_by_num(n_batch, src, rel, dst, src_cand, rel_cand, dst_cand, n_sample=n_train):
        # send corrupted triples from Neg of size "n_batch" to generator
        gen_step = gen.gen_step(ss, rs, ts, temperature=config().adv.temperature)
        # randomly sample from probability distribution of current negative triple set 
        src_smpl, dst_smpl = next(gen_step)
        # send sampled negative triple "dst_smpl" and its ground truth triple "src_smpl" to discriminator 
        losses, rewards = dis.dis_step(s, r, t, src_smpl.squeeze(), dst_smpl.squeeze())
        epoch_reward += torch.sum(rewards)        
        rewards = rewards - avg_reward
        # send reward to generator
        gen_step.send(rewards.unsqueeze(1))
        epoch_d_loss += torch.sum(losses)

    avg_loss = epoch_d_loss / n_train
    avg_reward = epoch_reward / n_train
    logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
    if create_graphs:
        logged_rewards.append(avg_loss.data)
        logged_D_losses.append(avg_reward.data)
    
    if (epoch + 1) % config().adv.epoch_per_test == 0:
        #gen.test_link(valid_data, n_ent, filt_heads, filt_tails)
        mrr, hit10 = dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
        if create_graphs:
            logged_mrrs.append(mrr)
            logged_hit10s.append(hit10)
        if mrr > best_mrr:
            best_mrr = mrr
            dis.save(os.path.join('models', config().task.dir, mdl_name))
            
if create_graphs:
    # create folder which contains all figures
    dir = os.path.join(config().graph.dir, timestamp)
    os.mkdir(dir)    

    # plot results of current adversarial learning process
    create_figure(task_dir.upper() + ' Rewards', [x+1 for x in range(n_epoch)], logged_rewards, 'Epochs', 'Rewards','blue').savefig(os.path.join(dir, task_dir + '_rewards'))
    create_figure(task_dir.upper() + ' Discriminator Losses', [x+1 for x in range(n_epoch)], logged_D_losses, 'Epochs', 'Losses', 'red').savefig(os.path.join(dir, task_dir + '_losses'))
    create_figure(task_dir.upper() + ' Validation MRR', [x+1 for x in range(n_epoch) if (x + 1) % config().adv.epoch_per_test == 0], logged_mrrs,'Epochs', 'MRR',  'green').savefig(os.path.join(dir, task_dir + '_mrr'))
    create_figure(task_dir.upper() + ' Validation H@10', [x+1 for x in range(n_epoch) if (x + 1) % config().adv.epoch_per_test == 0], logged_hit10s, 'Epochs', 'H@10', 'orange').savefig(os.path.join(dir, task_dir + '_hit10'))

    # TODO: load plots from original KBGAN approach an compare them 
    #   -> for each model
    #   -> for each dataset
    # TODO: combine plots of my approach for each model and each dataset
    
    
dis.load(os.path.join('models', config().task.dir, mdl_name))
dis.test_link(test_data, n_ent, filt_heads, filt_tails)