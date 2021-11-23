from base_sampler import BaseSampler
 
class UncertaintySampler(BaseSampler):
  
  def __init__(self, measure):
    self.measure = measure
  
