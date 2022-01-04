class BaseSampler(object):
  def __init__(self):
    pass
  
  def sample(src, h, r, t, h_neg, r_neg, t_neg, n_sample, *args):
    raise NotImplementedError()
  
  
class Measure(object):
  def __init__(self):
    self.fct = None
  
  
    
class Entropy(Measure):
  def __init__(self):
    pass
  