class BaseSampler(object):
  def __init__(self):
    pass
  
  def sample(src, rel, dst, n_sample, *args):
    raise NotImplementedError()