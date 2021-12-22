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
    sample_idx1 = torch.multinomial(args[0], n_sample, replacement=True)
    sample_idx2 = torch.multinomial(args[0], n_sample, replacement=True)
    
    batch_entropies = []
    torch.set_printoptions(threshold=10_000) 
     
    print(batch_probs)
    batch_entropies = - (batch_probs * torch.log(batch_probs)) - ((1 - batch_probs) * torch.log(1 - batch_probs)) 
    print(batch_entropies)
    # get the maximum  

    # max = torch.max(batch_entropies, 1) 
    # sample_idx_uncertainty = max.indices.unsqueeze(1)  
    sample_idx_uncertainty = torch.multinomial(batch_entropies, n_sample, replacement=True)
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty
