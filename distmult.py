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

    def forward(self, head, rel, tail):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.ent_embed(head)
        # (1.2) Real embeddings of relations
        emb_rel_real = self.rel_embed(rel)
         # (1.3) Real embeddings of tails
        emb_tail_real = self.ent_embed(tail)
        return t.sum(emb_tail_real * emb_head_real * emb_rel_real, dim=-1)  

    def score(self, head, rel, tail):
        """ Score function of DistMult.
            Indicates the plausability of a triple to be true 
            -> the higher the score to more likely the triple is true
            we return this value * (-1) such that we can use the same evaluation function as for TransE/TransD

        Args:
            head (torch.tensor): head entities
            rel (torch.tensor): relations
            tail (torch.tensor)): tail entities
            
        Returns:
            torch.tensor: scores of each triple (h, r, t) * (-1)
        """
        # low scores indicate a low probability of a triple to be true and vice versa
        # => *(-1) to use the same evaluation function test_link as it is used for TransE/TransD
        return -self.forward(head, rel, tail)

    def prob_logit(self, head, rel, tail):
        return self.forward(head, rel, tail)

class DistMult(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(DistMult, self).__init__()
        self.mdl = DistMultModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester, log_dir = None):
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
            for h_neg, r_neg, t_neg in batch_by_num(n_batch, head_corrupted, rel_corrupted, tail_corrupted, n_sample=n_train):
                # zero gradients
                self.mdl.zero_grad()
                if t.cuda.is_available():
                    label = t.zeros(len(h_neg)).type(t.LongTensor).cuda()
                else:
                    label = t.zeros(len(h_neg)).type(t.LongTensor)
                # forward pass
                loss = t.sum(self.mdl.softmax_loss(Variable(h_neg), Variable(r_neg), Variable(t_neg), label))

                # backward pass
                loss.backward()
                
                # update
                optimizer.step()
                epoch_loss += loss.item()                
           
            avg_epoch_loss = epoch_loss / n_train
            if epoch % config().log.graph_every_nth_epoch == 0:  
                tp_logger.log_loss_reward(avg_epoch_loss)     
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