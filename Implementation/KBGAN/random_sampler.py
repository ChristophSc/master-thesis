from base_sampler import BaseSampler
import torch
 
class RandomSampler(BaseSampler):
  def __init__(self):
    pass
  
  
  def sample(self, src, rel, dst, n_sample, *args):
    n, m = dst.size()
    
    if len(args) != 1:
      raise ValueError()
              
    # TODO: move following to RandomSampler
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)          
    sample_idx = torch.multinomial(args[0], n_sample, replacement=True)
        
    return row_idx, sample_idx