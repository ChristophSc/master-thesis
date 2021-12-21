from base_sampler import BaseSampler
import torch

class UncertaintySampler(BaseSampler):
  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
    
  def sample(self, src, rel, dst, n_sample, *args):
    n, m = dst.size()
    
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
