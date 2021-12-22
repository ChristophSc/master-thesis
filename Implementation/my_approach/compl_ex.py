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

    def forward(self, src, rel, dst):
        return t.sum(self.rel_re_embed(rel) * self.ent_re_embed(src) * self.ent_re_embed(dst), dim=-1) \
            + t.sum(self.rel_re_embed(rel) * self.ent_im_embed(src) * self.ent_im_embed(dst), dim=-1) \
            + t.sum(self.rel_im_embed(rel) * self.ent_re_embed(src) * self.ent_im_embed(dst), dim=-1) \
            - t.sum(self.rel_im_embed(rel) * self.ent_im_embed(src) * self.ent_re_embed(dst), dim=-1)

    def score(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def dist(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return self.forward(src, rel, dst)

class ComplEx(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(ComplEx, self).__init__()
        self.mdl = ComplExModule(n_ent, n_rel, config)
        if t.cuda.is_available():
            self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester, log_dir):
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
                self.mdl.zero_grad()                
                label = t.zeros(len(ss)).type(t.LongTensor)
                if t.cuda.is_available():
                    label.cuda()
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))
                loss.backward()
                optimizer.step()
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