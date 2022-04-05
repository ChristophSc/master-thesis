from base_sampler import BaseSampler
import torch
import torch.nn.functional as nnf
import math


class OriginalSampler(BaseSampler):
  def __init__(self):
    pass  
  
  def sample(self, n, n_sample, generator_logits, min_score, max_score):       
    probs = nnf.softmax(generator_logits, dim=-1)              
            
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)          
    sample_idx = torch.multinomial(probs, n_sample, replacement=True)        
    return row_idx, sample_idx



class ConfidenceSampler(BaseSampler):
  def __init__(self):
    pass  
  
  def sample(self, n, n_sample, generator_logits, min_score, max_score):           
    plausibility_score = generator_logits
    uncertainty_scores = torch.sigmoid(plausibility_score)    
    probs = nnf.softmax(uncertainty_scores, dim=-1)              
            
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)          
    sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        
    return row_idx, sample_idx