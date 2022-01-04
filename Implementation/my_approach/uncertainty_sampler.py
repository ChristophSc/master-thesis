from base_sampler import BaseSampler
import torch
import math

class UncertaintySampler_Basic(BaseSampler):  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
    
  def sample(self, h, r, t, h_neg, r_neg, t_neg, n_sample, *args):
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


class UncertaintySampler_Advanced(BaseSampler):  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
    
  def sample(self, h, r, t, h_neg, r_neg, t_neg, n_sample, *args):
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
    generator_score = scores       # TODO: change and add more information to generator_score
    #print(generator_score)
    is_positive_probs = (generator_score - min_score) / (max_score - min_score)
    is_positive_probs[is_positive_probs <= 0] = 0.0001
    is_positive_probs[is_positive_probs >= 1] = 0.9999
    
    #print(is_positive_probs)    
    is_negative_probs = 1 - is_positive_probs
    #print(is_negative_probs)
    #print(sample_probs)
    entropies = - (is_negative_probs * torch.log(is_negative_probs)) - ((1 - is_negative_probs) * torch.log(1 - is_negative_probs)) 
    #print(entropies)
    #print(entropies)
    # get the maximum  
    # max = torch.max(entropies, 1) 
    #print(max.values)
    # sample_idx_uncertainty = max.indices.unsqueeze(1)  
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    logits = entropies
    return row_idx, None, logits
  
