from base_sampler import BaseSampler
import torch
 
class RandomSampler(BaseSampler):
  def __init__(self):
    pass
  
  
  def sample(self, h, r, t, h_neg, r_neg, t_neg, n_sample, *args):
    n, m = t_neg.size()
    
    if len(args) != 1:
      raise ValueError()
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)          
    sample_idx = torch.multinomial(args[0], n_sample, replacement=True)
        
    return row_idx, sample_idx