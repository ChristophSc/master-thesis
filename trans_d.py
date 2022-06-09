import os
import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
from TrainingProcessLogger import TrainingProcessLogger

class TransDModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(TransDModule, self).__init__()
        self.margin = config.margin
        self.p = config.p
        self.temp = config.get('temp', 1)
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.proj_rel_embed = nn.Embedding(n_rel, config.dim)
        self.proj_ent_embed = nn.Embedding(n_ent, config.dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head, rel, tail):
        head_proj = self.ent_embed(head) +\
                   t.sum(self.proj_ent_embed(head) * self.ent_embed(head), dim=-1, keepdim=True) * self.proj_rel_embed(rel)
        tail_proj = self.ent_embed(tail) +\
                   t.sum(self.proj_ent_embed(tail) * self.ent_embed(tail), dim=-1, keepdim=True) * self.proj_rel_embed(rel)
        return t.norm(tail_proj - self.rel_embed(rel) - head_proj + 1e-30, p=self.p, dim=-1)

    def dist(self, head, rel, tail):
        return self.forward(head, rel, tail)

    def score(self, head, rel, tail):
        return self.forward(head, rel, tail)

    def prob_logit(self, head, rel, tail):
        return -self.forward(head, rel, tail) / self.temp

    def constraint(self):
        for param in self.parameters():
            param.data.renorm_(2, 0, 1)

class TransD(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(TransD, self).__init__()
        self.mdl = TransDModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config

    def load_vec(self, path):
        ent_mat = np.loadtxt(os.path.join(path, 'entity2vec.vec'))
        self.mdl.ent_embed.weight.data.copy_(t.from_numpy(ent_mat))
        rel_mat = np.loadtxt(os.path.join(path, 'relation2vec.vec'))
        n_rel = rel_mat.shape[0]
        self.mdl.rel_embed.weight.data.copy_(t.from_numpy(rel_mat))
        a_mat = np.loadtxt(os.path.join(path, 'A.vec'))
        self.mdl.proj_rel_embed.weight.data.copy_(t.from_numpy(a_mat[:n_rel, :]))
        self.mdl.proj_ent_embed.weight.data.copy_(t.from_numpy(a_mat[n_rel:, :]))
        if t.cuda.is_available():
            self.mdl.cuda()

    def pretrain(self, train_data, corrupter, tester, log_dir):
        head, rel, tail = train_data
        n_train = len(head)
        optimizer = Adam(self.mdl.parameters())
        #optimizer = SGD(self.mdl.parameters(), lr=1e-4)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        best_perf = 0
        tp_logger = TrainingProcessLogger('pretrain', n_epoch, self.config.epoch_per_test)     
        tp_logger.log_performance(0, [0,0,0])  

        for epoch in range(n_epoch):
            epoch_loss = 0
            rand_idx = t.randperm(n_train)
            head = head[rand_idx]
            rel = rel[rand_idx]
            tail = tail[rand_idx]
            head_corrupted, tail_corrupted = corrupter.corrupt(head, rel, tail)
            if t.cuda.is_available():
                head = head.cuda()
                rel = rel.cuda()
                tail = tail.cuda()
                head_corrupted = head_corrupted.cuda()
                tail_corrupted = tail_corrupted.cuda()
            for h_pos, r, t_pos, h_neg, t_neg in batch_by_num(n_batch, head, rel, tail, head_corrupted, tail_corrupted,
                                                  n_sample=n_train):
                self.mdl.zero_grad()
                loss = t.sum(self.mdl.pair_loss(Variable(h_pos), Variable(r), Variable(t_pos), Variable(h_neg), Variable(t_neg)))
                loss.backward()
                optimizer.step()
                self.mdl.constraint()
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / n_train
            if epoch % config().log.graph_every_nth_epoch == 0:  
                tp_logger.log_loss_reward(avg_epoch_loss)     
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch,avg_epoch_loss)
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
