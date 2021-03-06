import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging

from original_sampler import OriginalSampler
from uncertainty_sampler import *

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, head, rel, tail):
        raise NotImplementedError

    def dist(self, head, rel, tail):
        raise NotImplementedError

    def prob_logit(self, head, rel, tail):
        raise NotImplementedError

    def constraint(self):
        pass
    
    def prob(self, head, rel, tail):
        """Returns probability distribution over given triples of (head, rel, tail)

        Args:
            heads (torch.tensor): set of head entities
            rels (torch.tensor): relations
            tails (torch.tensor): set of tail entities

        Returns:
            torch.tensor: probability of each triple to be sampled in [0, 1]
                          sum is equals to 1
        """
        return nnf.softmax(self.prob_logit(head, rel, tail),  dim=1)

    def pair_loss(self, head, rel, tail, head_corr, tail_corr):
        """ Calculates the pair loss between score of positive and score of negative triple.
            Activation function: ReLU

        Args:
            head (torch.tensor): set of heads for positive triples
            rel (torch.tensor): set of entities for positive AND corrupted triples
            tail (torch.tensor): set of tails for positive triples 
            head_corr (torch.tensor): set of heads for corrupted triples
            tail_corr (torch.tensor): set of tails for corrupted triples

        Returns:
            float: float of pair loss between positive and corrupted triple (from ReLU activation function)
        """
        # used for TransE and TransD
        # distance of positive triple should be lower than distance of negative triple
        score_pos = self.score(head, rel, tail)
        score_neg = self.score(head_corr, rel, tail_corr) 
        return nnf.relu(self.margin + score_pos - score_neg)

    def softmax_loss(self, head, rel, tail, truth):
        """ Calculates the softmax log-loss between score of corrupted triple and the actual truth (all 0s for negative triples)

        Args:
            head (torch.tensor): set of heads from negative/corrupted triples
            rel (torch.tensor): set of relations from negative/corrupted triples
            tail (torch.tensor): set of tails from negative/corrupted triples
            truth (torch.tensor): torch.tensor with all labels, 0s for negative triples 
            
        Returns:
            torch.tensor: probability distribution for all negative triples to be sampled
        """
        probs = self.prob(head, rel, tail)
        n = probs.size(0)
        if torch.cuda.is_available():
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth])   # + 1e-30
        else:
            truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor), truth])   # + 1e-30
        return -truth_probs    # log(x) returns values < 0 for x < 1 => *(-1) to have probabilities between [0, 1]


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
            
            
    def gen_step(self, head, rel, tail, n_sample=1, temperature=1.0, train=True, sampler=OriginalSampler(), min_score = None, max_score = None):
        """One learning step of the Generator component in Adversarial Learning Process.
        
        Args:
            head (torch.tensor): corrupted head entities from Neg (from negative triple)
            rel (torch.tensor): relations
            tail (torch.tensor): corrupted tail entities from Neg (from negative triple)
            n_sample (int, optional): Number of negatives to be sampled from Neg. Defaults to 1.
            temperature (float, optional): dividant of the logits. Defaults to 1.0.
            train (bool, optional): train the Generator?. Defaults to True.
            sampler (BaseSampler, optional): Sampler which samples negative triples from set Neg. Defaults to RandomSampler().

        Yields:
            sample_heads, sample_tails (torch.tensor, torch.tensor): sampled negative heads and tails
        """
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = tail.size() # tail.size() same as head.size and rel.size()
        if torch.cuda.is_available():
            head = head.cuda()
            rel = rel.cuda()           
            tail = tail.cuda()
        head_var = Variable(head)
        rel_var = Variable(rel)        
        tail_var = Variable(tail)

        logits = self.mdl.prob_logit(head_var, rel_var, tail_var) / temperature    
        
        # call sampler to retrieve n_sample from negative triple set Neg
        row_idx, sample_idx = sampler.sample(n, n_sample, logits, min_score, max_score)
    
        # get head and tail of negative triple by sampled index
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]        
                
        rewards = yield sample_heads, sample_tails
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            if torch.cuda.is_available():
                row_idx = row_idx.cuda()
                
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx, sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, head, rel, tail, head_fake, tail_fake, train=True):
        """One learning step of the Discriminator component in Adversarial Learning Process.

        Args:
            head (torch.tensor): original head entities from KG (positive triples)
            rel (torch.tensor): original relations from KG
            tail (torch.tensor): original tail entities from KG (positive triples)
            head_fake (torch.tensor): corrupted head entities sampled from Neg (negative triples)
            tail_fake (torch.tensor): corrupted tail entities sampled from Neg (negative triples)
            train (bool, optional): Train the Discriminator. Defaults to True.

        Returns:
            discriminator_losses, rewards (torch.tensor, torch,tensor): 
            Losses of discriminator learning process and rewards for sampled negative triples for Generator
        """
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        if torch.cuda.is_available():
            head = head.cuda()
            rel = rel.cuda()
            tail = tail.cuda()
            head_fake = head_fake.cuda()
            tail_fake = tail_fake.cuda()
        head_var = Variable(head)
        rel_var = Variable(rel)
        tail_var = Variable(tail)
        head_fake_var = Variable(head_fake)
        tail_fake_var = Variable(tail_fake)
        # calculate the marginal loss between score of positive and negative triple
        losses = self.mdl.pair_loss(head_var, rel_var, tail_var, head_fake_var, tail_fake_var)
        # calculate score of negative triple => the lower the better 
        fake_scores = self.mdl.score(head_fake_var, rel_var, tail_fake_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        # return reward to generator which has to be maximized => small positive fake_scores = rewards are small negative values (close to zero)
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
                                .type(torch.LongTensor))    #  .cuda())          
                   
            # Scoring functions measure the plausibility of triplets in knowledge graph (KG), 
            # high scores = high probability to be in the KG
            batch_head_scores = self.mdl.score(all_var, rel_var, tail_var).data
            batch_tail_scores = self.mdl.score(head_var, rel_var, all_var).data   
            
            # compute head andn tail scores for each positive  
            for h_tensor, r_tensor, t_tensor, head_scores, tail_scores in zip(batch_h, batch_r, batch_t, batch_head_scores, batch_tail_scores):
                # head_scores/tail_scores: scores for predicted heads/tails
                
                # indices of head, relation and tail in the KG
                h = int(h_tensor.data.cpu().numpy())
                r = int(r_tensor.data.cpu().numpy())
                t = int(t_tensor.data.cpu().numpy())
                
                # filter triples which are already in the training data -> set their score very low
                if filt:      
                    # (h, r): key h=head, r=relation
                    # tails: dict indicates which tails t are connected with current (h,r) in ther KG
                    # -> spare tensor, only indices with value != 0 and their value is stored, here only binary \in {0,1} 0 = no connection, 1 = connection               
                    if tails[(h, r)]._nnz() > 1:    # nnz = number of non zeroes => there are tails t which are connected to head h and tail t in triple (h,r,t) in KG
                        #print(tail_scores)
                        tmp = tail_scores[t].item()   # save score for current predicted
                        idx = tails[(h, r)]._indices()
                         # since we know all other triples (including tails) and that they exist in the KG, we can set score very high
                        tail_scores[idx] = 1e20 
                        # reset score of current triple
                        tail_scores[t] = tmp
                        #print(tail_scores)
                    if heads[(t, r)]._nnz() > 1:
                        tmp = head_scores[h].item()
                        idx = heads[(t, r)]._indices()
                        head_scores[idx] = 1e20  # since we know all other triples (including heads) and that they exist in the KG, we can set score very high
                        head_scores[h] = tmp
                mrr, mr, hits = mrr_mr_hitk(tail_scores, t)               
                mrr_tot += mrr
                mr_tot += mr
                hits_tot += hits                
                mrr, mr, hits = mrr_mr_hitk(head_scores, h)
                mrr_tot += mrr
                mr_tot += mr
                hits_tot += hits
                count += 2
                
        mrr = float(mrr_tot)/count
        mr = float(mr_tot)/count
        hits = hits_tot / count
        logging.info('MRR=%f, MR=%f, H@10=%f', mrr, mr, hits[2])
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