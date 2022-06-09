import torch 

class BaseSampler:  
  def sample(self, n, n_sample, logits, min_score, max_score): 
    return NotImplementedError