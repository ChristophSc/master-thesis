from base_sampler import BaseSampler
import torch

class UncertaintySampler(BaseSampler):
  
  def __init__(self):
    self.measure = None # TODO: replace and add as parameter
  
  
  
  def generator_Score(self, src, rel, dst):
    pass
    
  def sample(self, src, rel, dst, n_sample, *args):
    n, m = dst.size()
    
    if len(args) != 1:
      raise ValueError()
    
    batch_probs = args[0]
      
    batch_entropies = []
    # probs: torch.Size([52, 20])
    # 52 = batch size
    # 20 = size of negative set Neg
    for probs in batch_probs:
      # sum(probs) is always 1 
      # len(probs) = 20 = number of negatives in Neg
      entropies = []
      for prob in probs:        
        ent = - (prob * torch.log(prob)) - ((1 - prob) * torch.log(1 - prob)) 
        entropies.append(ent)
      batch_entropies.append(entropies)
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    batch_entropies = torch.Tensor(batch_entropies  )
    # get the maximum    
    max = torch.max(batch_entropies, 1)   
           
    # sample_idx = torch.multinomial(args[0], n_sample, replacement=True)
    sample_idx_random = torch.multinomial(args[0], n_sample, replacement=True)

    sample_idx = max.indices 
    sample_idx_list = []   
    cnt_equals, cnt_odd = 0, 0
    for i in range(sample_idx.shape[0]): 
      sample_idx_list.append([sample_idx[i],])
      if sample_idx[i] == sample_idx_random[i][0]:
        cnt_equals += 1
      else:
        cnt_odd += 1
        
    sample_idx = torch.tensor(sample_idx_list)
    #print('equals:', cnt_equals)
    #print('odd:', cnt_odd)
    
    return row_idx, sample_idx
