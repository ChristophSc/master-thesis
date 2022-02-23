import os
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
from TrainingProcessLogger import TrainingProcessLogger

class TransEModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(TransEModule, self).__init__()
        self.p = config.p
        self.margin = config.margin
        self.temp = config.get('temp', 1)
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head, rel, tail):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head = self.ent_embed(head)
        # (1.2) Real embeddings of relations
        emb_rel = self.rel_embed(rel)
        # (1.3) Real embeddings of tail entities
        emb_tail = self.ent_embed(tail)
        # distance = || h + r - t||
        # => higher distance = smaller score because estimated likelihood of the triple to be true  
        distance = (emb_head + emb_rel) - emb_tail
        score = t.norm((-1)*distance, p=self.p, dim=-1)
        return score

    def score(self, head, rel, tail):
        """ Returns score of TransE. Score function = L1/L2 distance of h + r - t.
        => low score indicates high probaility of triple to be true.

        Args:
            head (torch.tensor): set of head entities
            rel (torch.tensor): set of relations
            tail (torch.tensor): set of tail entities

        Returns:
            Score = Distance of triple
        """
        score = self.forward(head, rel, tail)
        # If distance is very small , then score is very high
        # If distance is very large, then score is very small
        return score

    def prob_logit(self, head, rel, tail):
        """ Function to provide logits for sampling. High logits = higher probability to be sampled.
            Forward returns small distances = small scores for positives.
            Therefore, their logit must be inverted

        Args:
            heads ([type]): [description]
            rels ([type]): [description]
            tails ([type]): [description]

        Returns:
            [type]: [description]
        """
        return -self.forward(head, rel,  tail) / self.temp

    def constraint(self):
        self.ent_embed.weight.data.renorm_(2, 0, 1)
        self.rel_embed.weight.data.renorm_(2, 0, 1)

class TransE(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(TransE, self).__init__()
        self.mdl = TransEModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config

    def pretrain(self, train_data, corrupter, tester, log_dir):
        head, rel, tail = train_data
        n_train = len(head)
        optimizer = Adam(self.mdl.parameters())
        #optimizer = SGD(self.mdl.parameters(), lr=1e-4)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        best_perf = 0
        
        tp_logger = TrainingProcessLogger('pretrain', n_epoch, self.config.epoch_per_test)
        tp_logger.log_loss_reward(0, 0)     
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
                # h0, t0 = original head/tail from positive triple
                # h_corr, t_corr = corrupted head/tail for negative triple
                
                # zero gradients
                self.mdl.zero_grad()
                
                # forward pass
                loss = t.sum(self.mdl.pair_loss(Variable(h_pos), Variable(r), Variable(t_pos), Variable(h_neg), Variable(t_neg)))
                
                #backward pass
                loss.backward()
                
                # update
                optimizer.step()
                self.mdl.constraint()
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / n_train
            tp_logger.log_loss_reward(epoch, avg_epoch_loss)     
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, avg_epoch_loss)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                mrr, hits = tester()
                tp_logger.log_performance(mrr, hits)
                if mrr > best_perf:
                    self.save(os.path.join('models', config().task.dir, self.config.model_file))
                    best_perf = mrr
        if config().log.log_pretrain_graph:
            tp_logger.create_and_save_figures(log_dir)
        return best_perf
