from base_sampler import BaseSampler
import torch
import torch.nn.functional as nnf
class RandomSampler(BaseSampler):
  def __init__(self):
    pass
  
  
  def sample(self, head_rel_count, rel_tail_count, head, rel, tail, n_sample, *args):
    n, m = tail.size()
    
    if len(args) != 3:
      raise ValueError()
    
    generator_logits = args[0]
    probs = nnf.softmax(generator_logits, dim=-1)              
            
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)          
    sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        
    return row_idx, sample_idx