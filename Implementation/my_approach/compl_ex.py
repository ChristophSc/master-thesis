import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
import logging
import os
from TrainingProcessLogger import TrainingProcessLogger

class ComplExModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(ComplExModule, self).__init__()
        sigma = 0.2
        self.rel_re_embed = nn.Embedding(n_rel, config.dim)
        self.rel_im_embed = nn.Embedding(n_rel, config.dim)
        self.ent_re_embed = nn.Embedding(n_ent, config.dim)
        self.ent_im_embed = nn.Embedding(n_ent, config.dim)
        for param in self.parameters():
            param.data.div_((config.dim / sigma ** 2) ** (1 / 6))

    def forward(self, head, rel, tail):
        return t.sum(self.rel_re_embed(rel) * self.ent_re_embed(head) * self.ent_re_embed(tail), dim=-1) \
            + t.sum(self.rel_re_embed(rel) * self.ent_im_embed(head) * self.ent_im_embed(tail), dim=-1) \
            + t.sum(self.rel_im_embed(rel) * self.ent_re_embed(head) * self.ent_im_embed(tail), dim=-1) \
            - t.sum(self.rel_im_embed(rel) * self.ent_im_embed(head) * self.ent_re_embed(tail), dim=-1)

    def score(self, head, rel, tail):
        return -self.forward(head, rel, tail)

    def dist(self, head, rel, tail):
        return -self.forward(head, rel, tail)

    def prob_logit(self, head, rel, tail):
        return self.forward(head, rel, tail)

class ComplEx(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(ComplEx, self).__init__()
        self.mdl = ComplExModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester, log_dir):
        head, rel, tail = train_data
        n_train = len(head)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        best_perf = 0
        tp_logger = TrainingProcessLogger('pretrain', n_epoch, self.config.epoch_per_test)
        tp_logger.log_performance(0, [0,0,0])  

        for epoch in range(n_epoch):
            epoch_loss = 0
            if epoch % self.config.sample_freq == 0:
                rand_idx = t.randperm(n_train)
                head = head[rand_idx]
                rel = rel[rand_idx]
                tail = tail[rand_idx]
                head_corrupted, rel_corrupted, tail_corrupted = corrupter.corrupt(head, rel, tail)
                if t.cuda.is_available():
                    head_corrupted = head_corrupted.cuda()
                    rel_corrupted = rel_corrupted.cuda()
                    tail_corrupted = tail_corrupted.cuda()
            for ss, rs, ts in batch_by_num(n_batch, head_corrupted, rel_corrupted, tail_corrupted, n_sample=n_train):
                self.mdl.zero_grad()                
                label = t.zeros(len(ss)).type(t.LongTensor)
                if t.cuda.is_available():
                    label.cuda()
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / n_train
            if epoch % config().log.graph_every_nth_epoch == 0:  
                tp_logger.log_loss_reward(epoch, avg_epoch_loss)     
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, avg_epoch_loss)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                mrr, hits = tester()
                tp_logger.log_performance(mrr, hits)
                if mrr > best_perf:
                    self.save(os.path.join('models', config().task.dir, self.config.model_file))
                    best_perf = mrr

        tp_logger.log_loss_reward(avg_epoch_loss)    
        if config().log.log_pretrain_graph:
            tp_logger.create_and_save_figures(log_dir)
        return best_perf