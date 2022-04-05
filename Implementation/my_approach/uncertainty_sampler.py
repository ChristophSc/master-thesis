from base_sampler import BaseSampler
import torch
import math
import uncertainty_measure as um
import torch.nn.functional as nnf



class Classifier:  
  def classify(self, generator_logits, min_score, max_score):        
    is_positive_probs = abs(generator_logits - min_score) / abs(max_score - min_score) 

    is_positive_probs[is_positive_probs <= 0] = 0.0001
    is_positive_probs[is_positive_probs >= 1] = 0.9999
    return  is_positive_probs 
  
class UncertaintyScorer:
  def __init__(self, measure):
    if measure == "entropy":
      self.measure = um.Entropy()
    elif measure == "least_confidence":
      self.measure = um.LeastConfidence()
    elif measure == "confidence_margin":
      self.measure = um.ConfidenceMargin()
    elif measure == "confidence_ratio":
      self.measure = um.ConfidenceRatio()    
    
  def measure_uncertainty(self, is_positive_probs):
    return self.measure.measure_uncertainty(is_positive_probs)
  
    
class UncertaintySampler(BaseSampler):  
  def __init__(self, measure):
    super().__init__()    
    self.classifier = Classifier()
    self.scorer = UncertaintyScorer(measure)
    
  def rel_tail_frq(self, h_neg, r_neg, t_neg, tail_count):
    n = sum([v for k, v in tail_count.items()])
    head_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        head_score = tail_count[(r.item(), t.item())] / n 
        batch_scores.append(head_score)
      head_scores.append(batch_scores)
    head_scores = torch.tensor(head_scores)
    return head_scores
  
  def head_rel_frq(self, h_neg, r_neg, t_neg, head_count):
    n = sum([v for k, v in head_count.items()])
    tail_scores = []    
    for head_batch, rel_batch, tail_batch in zip(h_neg, r_neg, t_neg):
      batch_scores = []
      for h, r, t in zip(head_batch, rel_batch, tail_batch):
        tail_score = head_count[(h.item(), r.item())] / n
        batch_scores.append(tail_score)
      tail_scores.append(batch_scores)
    tail_scores = torch.tensor(tail_scores)
    return tail_scores
  
  def sample(self, n, n_sample, logits, min_score, max_score):
    raise NotImplementedError

class UncertaintySampler_Max(UncertaintySampler):  
  def __init__(self, measure):
    super().__init__(measure)   
   
  def sample(self, n, n_sample, logits, min_score, max_score):       
    is_positive_probs = self.classifier.classify(logits, min_score, max_score)  
    
    uncertainty_scores = self.scorer.measure_uncertainty(is_positive_probs)

    # Sampler
    max = torch.max(uncertainty_scores, 1) 
    sample_idx_uncertainty = max.indices.unsqueeze(1)  
    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)       
    return row_idx, sample_idx_uncertainty

  
class UncertaintySampler_Softmax(UncertaintySampler):  
  def __init__(self, measure):
     super().__init__(measure)     

  def sample(self, n, n_sample, logits, min_score, max_score):     
    is_positive_probs = self.classifier.classify(generator_logits = logits, min_score = min_score, max_score = max_score)  
    
    uncertainty_scores = self.scorer.measure_uncertainty(is_positive_probs)    
    sampling_probs = torch.softmax(uncertainty_scores, dim = -1)    

    # sampling according to distribution of uncertainty scores
    sample_idx_uncertainty = torch.multinomial(sampling_probs, n_sample, replacement=True)

    row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)      
    return row_idx, sample_idx_uncertainty
  
