import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging

from random_sampler import RandomSampler
from uncertainty_sampler import UncertaintySampler

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst):
        raise NotImplementedError

    def dist(self, src, rel, dst):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst):
        raise NotImplementedError

    def constraint(self):
        pass
    
    def prob(self, src, rel, dst):
        return nnf.softmax(self.prob_logit(src, rel, dst),  dim=1)



    def pair_loss(self, src, rel, dst, src_bad, dst_bad):
        d_good = self.dist(src, rel, dst)
        d_bad = self.dist(src_bad, rel, dst_bad)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, truth):
        probs = self.prob(src, rel, dst)
        n = probs.size(0)
        if torch.cuda.is_available():
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth] + 1e-30)
        else:
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor), truth] + 1e-30)
        return -truth_probs


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0
        self.smpl = None

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        if torch.cuda.is_available():
            self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))
        else:
            self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage))

    def gen_step(self, src, rel, dst, n_sample=1, temperature=1.0, train=True, sampler=UncertaintySampler()):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = dst.size() # dst.size() same as src.size and rel.size()
        if torch.cuda.is_available():
            rel = rel.cuda()
            src = src.cuda()
            dst = dst.cuda()
        rel_var = Variable(rel)
        src_var = Variable(src)
        dst_var = Variable(dst)

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        
        probs = nnf.softmax(logits, dim=1)
        
        # TODO: get features with information about the KG (e.g. structure) (PEER, POP, FRQ, ...)
        # TODO: add Uncertainty Sampling here
        
        # smpl.sample(n_sample, probs)
        row_idx, sample_idx = sampler.sample(src, rel, dst, n_sample, probs)
    
        sample_srcs = src[row_idx, sample_idx.data.cpu()]
        sample_dsts = dst[row_idx, sample_idx.data.cpu()]        
        
        rewards = yield sample_srcs, sample_dsts
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits, dim=1)
            if torch.cuda.is_available():
                row_idx = row_idx.cuda()
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx, sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        if torch.cuda.is_available():
            src = src.cuda()
            rel = rel.cuda()
            dst = dst.cuda()
            src_fake = src_fake.cuda()
            dst_fake = dst_fake.cuda()
        src_var = Variable(src)
        rel_var = Variable(rel)
        dst_var = Variable(dst)
        src_fake_var = Variable(src_fake)
        dst_fake_var = Variable(dst_fake)
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var)
        fake_scores = self.mdl.score(src_fake_var, rel_var, dst_fake_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails, filt=True):
        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):         
            batch_size = batch_s.size(0)
            if torch.cuda.is_available():
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
                src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
                dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
                with torch.no_grad():
                    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                                .type(torch.LongTensor).cuda())
            else:
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent))
                src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent))
                dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent))
                with torch.no_grad():
                    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                                .type(torch.LongTensor))           
            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var).data      
            for s_tensor, r_tensor, t_tensor, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                # print(s_tensor, r_tensor, t_tensor, dst_scores, src_scores)
                s, r, t = s_tensor.item(), r_tensor.item(), t_tensor.item()
                if filt:                    
                    if tails[(s, r)]._nnz() > 1:
                        tmp = dst_scores[t]
                        if torch.cuda.is_available():
                            dst_scores += tails[(s, r)].cuda() #@IgnoreException# * 1e30   # add +1 (+1e30) at indices in stored in tails vector
                        else:
                            dst_scores += tails[(s, r)].to_dense() * 1e30
                        dst_scores[t] = tmp
                    if heads[(t, r)]._nnz() > 1:
                        tmp = src_scores[s]
                        if torch.cuda.is_available():
                            src_scores += heads[(t, r)].cuda() * 1e30
                        else:
                            src_scores += heads[(t, r)].to_dense() * 1e30
                        src_scores[s] = tmp
                mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t)                
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                mrr, mr, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit10_tot / count)
        return mrr_tot / count
    

    def pretrain(self, train_data, corrupter, tester):
        """ Pretrains model on given training data.

        Args:
            train_data ([type]): Training data]
            corrupter ([type]): Corrupter that creates negative examples
            tester ([type]): test function, usually test_link-function to test link prediction with MRR, MR and H@10.

        Raises:
            NotImplementedError: Only for BaseModel this function is abstract, inherited class should implement it.
        """
        raise NotImplementedError()