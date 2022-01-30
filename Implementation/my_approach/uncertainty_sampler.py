from base_sampler import BaseSampler
import torch
import math

class UncertaintySampler_Basic(BaseSampler):  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
    
  def sample(self, head_rel_count, rel_tail_count, h_neg, r_neg, t_neg, n_sample, *args):
    n, m = t_neg.size()
    
    if len(args) != 1:
      raise ValueError()
    
    batch_probs = args[0]
    batch_entropies = []
     
    batch_entropies = - (batch_probs * torch.log(batch_probs)) - ((1 - batch_probs) * torch.log(1 - batch_probs)) 
    # get the maximum  
    max = torch.max(batch_entropies, 1) 
    sample_idx_uncertainty = max.indices.unsqueeze(1)  
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty


class UncertaintySampler_Entropy_Max(BaseSampler):  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
   
   
  def rel_tail_frq(self, h_neg, r_neg, t_neg, tail_count):
    n = sum([v for k, v in tail_count.items()])
    head_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        head_score = tail_count[(r.item(), t.item())] / n 
        batch_scores.append(head_score)
      head_scores.append(batch_scores)
    head_scores = torch.tensor(head_scores)
    return head_scores
  
  def head_rel_frq(self, h_neg, r_neg, t_neg, head_count):
    n = sum([v for k, v in head_count.items()])
    tail_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        tail_score = head_count[(h.item(), r.item())] / n
        batch_scores.append(tail_score)
      tail_scores.append(batch_scores)
    tail_scores = torch.tensor(tail_scores)
    return tail_scores
  
   
    
  def sample(self, head_rel_count, rel_tail_count, h_neg, r_neg, t_neg, n_sample, *args):
    n, m = t_neg.size()
    
    if len(args) != 3:
      raise ValueError()
    
    scores = args[0]
    summe = torch.sum(scores[0], dim = -1)
    min_score =  args[1]
    max_score = args[2]
    #sample_idx1 = torch.multinomial(args[0], n_sample, replacement=True)
    #sample_idx2 = torch.multinomial(args[0], n_sample, replacement=True)
    
    batch_entropies = []
    torch.set_printoptions(threshold=10_000) 
     
    #print(batch_probs)
    lambda_1 = lambda_2 = lambda_3 = 1
    # head_rel_score = - self.rel_tail_frq(h_neg, r_neg, t_neg, rel_tail_count)
    # rel_tail_score = - self.head_rel_frq(h_neg, r_neg, t_neg, head_rel_count)  

    generator_score = lambda_1 * scores  # + lambda_2 * head_rel_score + lambda_3 * rel_tail_score       # TODO: change and add more information to generator_score
  
    # TODO: work with softmax here?
    is_positive_probs = (generator_score - min_score) / (max_score - min_score)
    # avoid nan values if there is a score < min_score or a score > max_score
    is_positive_probs[is_positive_probs <= 0] = 0.00001
    is_positive_probs[is_positive_probs >= 1] = 0.99999
       
    is_negative_probs = 1 - is_positive_probs
    entropies = - (is_negative_probs * torch.log2(is_negative_probs)) - ((1 - is_negative_probs) * torch.log2(1 - is_negative_probs)) 


    max = torch.max(entropies, 1) 
    sample_idx_uncertainty = max.indices.unsqueeze(1)  
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty
  
  
  
class UncertaintySampler_Entropy_Max_Distribution(BaseSampler):  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
   
   
  def rel_tail_frq(self, h_neg, r_neg, t_neg, tail_count):
    n = sum([v for k, v in tail_count.items()])
    head_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        head_score = tail_count[(r.item(), t.item())] / n 
        batch_scores.append(head_score)
      head_scores.append(batch_scores)
    head_scores = torch.tensor(head_scores)
    return head_scores
  
  def head_rel_frq(self, h_neg, r_neg, t_neg, head_count):
    n = sum([v for k, v in head_count.items()])
    tail_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        tail_score = head_count[(h.item(), r.item())] / n
        batch_scores.append(tail_score)
      tail_scores.append(batch_scores)
    tail_scores = torch.tensor(tail_scores)
    return tail_scores 
   
    
  def sample(self, head_rel_count, rel_tail_count, h_neg, r_neg, t_neg, n_sample, *args):
    n, m = t_neg.size()
    
    if len(args) != 3:
      raise ValueError()
    
    scores = args[0]
    summe = torch.sum(scores[0], dim = -1)
    min_score =  args[1]
    max_score = args[2]
    #sample_idx1 = torch.multinomial(args[0], n_sample, replacement=True)
    #sample_idx2 = torch.multinomial(args[0], n_sample, replacement=True)
    
    batch_entropies = []
    torch.set_printoptions(threshold=10_000) 
     
    #print(batch_probs)
    lambda_1 = lambda_2 = lambda_3 = 1
    # head_rel_score = - self.rel_tail_frq(h_neg, r_neg, t_neg, rel_tail_count)
    # rel_tail_score = - self.head_rel_frq(h_neg, r_neg, t_neg, head_rel_count)  
    generator_score = lambda_1 * scores # + lambda_2 * head_rel_score + lambda_3 * rel_tail_score       # TODO: change and add more information to generator_score
    
    # print(generator_score)
    
    is_positive_probs = (generator_score - min_score) / (max_score - min_score)
    # avoid nan values if there is a score < min_score or a score > max_score
    is_positive_probs[is_positive_probs <= 0] = 0.00001
    is_positive_probs[is_positive_probs >= 1] = 0.99999
      
    is_negative_probs = 1 - is_positive_probs
    entropies = - (is_negative_probs * torch.log2(is_negative_probs)) - ((1 - is_negative_probs) * torch.log2(1 - is_negative_probs)) 
    # max = torch.max(entropies, 1) 
    # print(entropies[0])
    sample_idx_uncertainty = torch.multinomial(entropies, n_sample, replacement=True)
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty
  
