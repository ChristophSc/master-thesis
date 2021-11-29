class BaseSampler(object):
  def __init__(self):
    pass
  
  def sample(src, rel, dst, n_sample, *args):
    raise NotImplementedError()
  
  
class Measure(object):
  def __init__(self):
    self.fct = None
  
  
    
class Entropy(Measure):
  def __init__(self):
    pass
  