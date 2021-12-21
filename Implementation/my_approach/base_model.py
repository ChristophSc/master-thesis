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
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth])   # + 1e-30
        else:
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor), truth])   # + 1e-30
        return -truth_probs


class BaseModel(object):
    """BaseModel for all KGE-Models (including Generator and Discriminator)"""
    
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
            
    #sampled_instances_random = dict()
    #sampled_instances_uncertainty = dict()
    def gen_step(self, src, rel, dst, n_sample=1, temperature=1.0, train=True, sampler=RandomSampler()):
        """One learning step of the Generator component in Adversarial Learning Process.
        

        Args:
            src (torch.tensor): corrupted head entities from Neg (from negative triple)
            rel (torch.tensor): relations
            dst (torch.tensor): corrupted tail entities from Neg (from negative triple)
            n_sample (int, optional): Number of negatives to be sampled from Neg. Defaults to 1.
            temperature (float, optional): dividant of the logits. Defaults to 1.0.
            train (bool, optional): train the Generator?. Defaults to True.
            sampler (BaseSampler, optional): Sampler which samples negative triples from set Neg. Defaults to UncertaintySampler().

        Yields:
            sample_srcs, sample_dsts (torch.tensor, torch.tensor): sampled negative heads and tails
        """
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
        # call sampler to retrieve n_sample from negative triple set Neg
        row_idx, sample_idx = sampler.sample(src, rel, dst, n_sample, probs)
    
        # get head and tail of negative triple by sampled index
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
        """One learning step of the Discriminator component in Adversarial Learning Process.

        Args:
            src (torch.tensor): original head entities from KG (positive triples)
            rel (torch.tensor): original relations from KG
            dst (torch.tensor): original tail entities from KG (positive triples)
            src_fake (torch.tensor): corrupted head entities sampled from Neg (negative triples)
            dst_fake (torch.tensor): corrupted tail entities sampled from Neg (negative triples)
            train (bool, optional): Train the Discriminator. Defaults to True.

        Returns:
            discriminator_losses, rewards (torch.tensor, torch,tensor): 
            Losses of discriminator learning process and rewards for sampled negative triples for Generator
        """
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
        """Evaluation Method to compare model results by link prediction tasks.  
        Returns Mean Retriprocal Rank (MMR) and Hit@10 

        Args:
            test_data (torch.tensor): training/validation data which is used for link prediction task.  
            n_ent (int): Number of entities  
            heads (dict with torch.tensor values): set of head entities  
            tails (dict with torch.tensor values): Set of tail entities  
            filt (bool, optional): Filters head and tail entities which are already in training data. Defaults to True.  

        Returns:
            (float, float): MRR, Hit@10
        """
        mrr_tot = 0
        mr_tot = 0
        hits_tot = 0
        count = 0
        for batch_h, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):         
            batch_size = batch_h.size(0)
            if torch.cuda.is_available():
                head_var = Variable(batch_h.unsqueeze(1).expand(batch_size, n_ent).cuda())
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
                
                tail_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
                with torch.no_grad():
                    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                                .type(torch.LongTensor).cuda())
            else:
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent))
                head_var = Variable(batch_h.unsqueeze(1).expand(batch_size, n_ent))
                tail_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent))
                with torch.no_grad():
                    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                                .type(torch.LongTensor))              
                    
            batch_head_scores = self.mdl.score(all_var, rel_var, tail_var).data
            batch_tail_scores = self.mdl.score(head_var, rel_var, all_var).data   
            
            # compute head andn tail scores for each positive  
            for h_tensor, r_tensor, t_tensor, head_scores, tail_scores in zip(batch_h, batch_r, batch_t, batch_head_scores, batch_tail_scores):
                # src_scores/dst_scores: scores for predicted heads/tails
                h = int(h_tensor.data.cpu().numpy())
                r = int(r_tensor.data.cpu().numpy())
                t = int(t_tensor.data.cpu().numpy())
                
                # filter triples which are already in the training data -> set their score very low
                if filt:      
                    # (h, r): key h=head, r=relation
                    # tails: dict indicates which tails t are connected with current (h,r) in ther KG
                    # -> spare tensor, only indices with value != 0 and their value is stored, here only binary \in {0,1} 0 = no connection, 1 = connection
                    # to_dense(): creates dense tensor with 0s and 1s                    
                    if tails[(h, r)]._nnz() > 1:    # nnz = number of non zeroes => there are tails t which are connected to head h and tail t in triple (h,r,t) in KG
                        #print(tail_scores)
                        tmp = tail_scores[t].data.cpu().numpy()
                        idx = tails[(h, r)]._indices()
                        tail_scores[idx] = 0.0
                        tail_scores[t] = torch.from_numpy(tmp)#.cuda()
                        #print(tail_scores)
                    if heads[(t, r)]._nnz() > 1:
                        tmp = head_scores[h].data.cpu().numpy()
                        idx = heads[(t, r)]._indices()
                        head_scores[idx] = 0.0
                        head_scores[h] = torch.from_numpy(tmp)#.cuda()
                mrr, mr, hits = mrr_mr_hitk(tail_scores, t)               
                mrr_tot += mrr
                mr_tot += mr
                hits_tot += hits                
                mrr, mr, hits = mrr_mr_hitk(head_scores, h)
                mrr_tot += mrr
                mr_tot += mr
                hits_tot += hits
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f, Count=%d', float(mrr_tot)/count, float(mr_tot)/count, hits_tot[2]/count, count)
        return mrr, hits
    

    def pretrain(self, train_data, corrupter, tester, log_dir):
        """ Pretrains model on given training data.

        Args:
            train_data ([type]): Training data]
            corrupter ([type]): Corrupter that creates negative examples
            tester ([type]): test function, usually test_link-function to test link prediction with MRR, MR and H@10.

        Raises:
            NotImplementedError: Only for BaseModel this function is abstract, inherited class should implement it.
        """
        raise NotImplementedError()