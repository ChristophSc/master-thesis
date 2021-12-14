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

    def forward(self, src, rel, dst):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head = self.ent_embed(src)
        # (1.2) Real embeddings of relations
        emb_rel = self.rel_embed(rel)
        # (1.3) Real embeddings of tail entities
        emb_tail = self.ent_embed(dst)
        distance = t.norm((emb_head + emb_rel) - emb_tail, p=self.p, dim=-1)
        # d = t.norm(self.ent_embed(dst) - self.ent_embed(src) - self.rel_embed(rel) + 1e-30, p=self.p, dim=-1)
        return  t.sigmoid(distance) # all distances >= 0: apply sigmoid to have valued between 0 and 1

    def dist(self, src, rel, dst):
        """Distance between head + rel = tail

        Args:
            src (torch.tensor): head entities
            rel (torch.tensor): relations
            dst (torch.tensor): tail entities

        Returns:
            real value > 0: distance head + rel = tail
            """
        return self.forward(src, rel, dst)

    def score(self, src, rel, dst):
        # If distance is very small , then score is very high, i.e. 1.0
        # If distance is very large, then score is very small, i.e. 0.0
        return self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return -self.forward(src, rel ,dst) / self.temp

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
        src, rel, dst = train_data
        n_train = len(src)
        optimizer = Adam(self.mdl.parameters())
        #optimizer = SGD(self.mdl.parameters(), lr=1e-4)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        best_perf = 0
        tp_logger = TrainingProcessLogger('pretrain', n_epoch, self.config.epoch_per_test)
        for epoch in range(n_epoch):
            epoch_loss = 0
            rand_idx = t.randperm(n_train)
            src = src[rand_idx]
            rel = rel[rand_idx]
            dst = dst[rand_idx]
            src_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
            if t.cuda.is_available():
                src = src.cuda()
                rel = rel.cuda()
                dst = dst.cuda()
                src_corrupted = src_corrupted.cuda()
                dst_corrupted = dst_corrupted.cuda()
            for s0, r, t0, s1, t1 in batch_by_num(n_batch, src, rel, dst, src_corrupted, dst_corrupted,
                                                  n_sample=n_train):
                self.mdl.zero_grad()
                loss = t.sum(self.mdl.pair_loss(Variable(s0), Variable(r), Variable(t0), Variable(s1), Variable(t1)))
                loss.backward()
                optimizer.step()
                self.mdl.constraint()
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
