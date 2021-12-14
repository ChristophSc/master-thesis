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
    batch_entropies = torch.Tensor(batch_entropies)
    # get the maximum    
    max = torch.max(batch_entropies, 1) 
      
    # RandomSampling would sample the following:
    sample_idx_random = torch.multinomial(args[0], n_sample, replacement=True)

    sample_idx_uncertainty = max.indices.unsqueeze(1)  
    # cnt_equals, cnt_unequals = 0, 0
    # for i in range(sample_idx_random.shape[0]): 
    #   if sample_idx_random[i][0] == sample_idx_uncertainty[i][0]:
    #     cnt_equals += 1
    #   else:
    #     cnt_unequals += 1
        
    # idx = int(sample_idx_random[0][0])
    # if idx not in base_model.BaseModel.sampled_instances_random.keys():
    #     base_model.BaseModel.sampled_instances_random[idx] = 0                 
    # base_model.BaseModel.sampled_instances_random[idx] += 1
    
    # idx = int(sample_idx[0][0])
    # if idx not in base_model.BaseModel.sampled_instances_uncertainty.keys():
    #     base_model.BaseModel.sampled_instances_uncertainty[idx] = 0    
    # base_model.BaseModel.sampled_instances_uncertainty[idx] += 1
    #print('equals:', cnt_equals)
    #print('unequals:', cnt_odd)
    
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty
