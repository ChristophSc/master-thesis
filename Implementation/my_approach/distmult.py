import torch as t
import torch.nn as nn
import torch.nn.functional as f
from TrainingProcessLogger import TrainingProcessLogger
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
import logging
import os
import datetime

class DistMultModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(DistMultModule, self).__init__()
        sigma = 0.2
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.rel_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.ent_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))

    def forward(self, src, rel, dst):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.ent_embed(src)
        # (1.2) Real embeddings of relations
        emb_rel_real = self.rel_embed(rel)
         # (1.3) Real embeddings of tails
        emb_tail_real = self.ent_embed(dst)
        return t.sum(emb_tail_real * emb_head_real * emb_rel_real, dim=-1)    

    def score(self, src, rel, dst):
        return -self.forward(src, rel, dst)
        # return t.sigmoid(self.forward(src, rel, dst))

    def dist(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return self.forward(src, rel, dst)

class DistMult(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(DistMult, self).__init__()
        self.mdl = DistMultModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester, log_dir = None):
        src, rel, dst = train_data
        n_train = len(src)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        best_perf = 0
        tp_logger = TrainingProcessLogger('pretrain', n_epoch, self.config.epoch_per_test)            
        for epoch in range(n_epoch):
            epoch_loss = 0
            if epoch % self.config.sample_freq == 0:
                rand_idx = t.randperm(n_train)
                src = src[rand_idx]
                rel = rel[rand_idx]
                dst = dst[rand_idx]
                src_corrupted, rel_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
                if t.cuda.is_available():
                    src_corrupted = src_corrupted.cuda()
                    rel_corrupted = rel_corrupted.cuda()
                    dst_corrupted = dst_corrupted.cuda()
            for ss, rs, ts in batch_by_num(n_batch, src_corrupted, rel_corrupted, dst_corrupted, n_sample=n_train):
                # zero gradients
                self.mdl.zero_grad()
                if t.cuda.is_available():
                    label = t.zeros(len(ss)).type(t.LongTensor).cuda()
                else:
                    label = t.zeros(len(ss)).type(t.LongTensor)
                # forward pass
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))

                # backward pass
                loss.backward()
                
                # update
                optimizer.step()
                epoch_loss += loss.item()                
           
            avg_epoch_loss = epoch_loss / n_train
            tp_logger.log_loss_reward(epoch, avg_epoch_loss)     
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, avg_epoch_loss)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                mrr, hit10 = tester()
                tp_logger.log_performance(mrr, hit10)
                if mrr > best_perf:
                    self.save(os.path.join('models', config().task.dir, self.config.model_file))
                    best_perf = mrr                    
        if config().log.log_pretrain_graph:
            tp_logger.create_and_save_figures(log_dir)
        return best_perf