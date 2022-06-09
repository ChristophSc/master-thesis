import torch

class UncertaintyMeasure():
  def measure_uncertainty(self, probs):
    raise NotImplementedError
  
  
class Entropy(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
    entropy = - ((1 - is_positive_probs) * torch.log2(1- is_positive_probs)) - (is_positive_probs * torch.log2(is_positive_probs)) 
    return entropy 
    

class LeastConfidence(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
    n = 2
    least_confidence = (1-torch.max(is_positive_probs, 1-is_positive_probs))*(n/(n-1)) # normilization to values in [0,1]
    return least_confidence
    
    
class ConfidenceMargin(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
    # only works for binary classification like this, otherwise the second max has to be find
    confidence_margin = 1-(torch.max(is_positive_probs, 1-is_positive_probs) - torch.min(is_positive_probs, 1-is_positive_probs))
    return confidence_margin


class ConfidenceRatio(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
     # only works for binary classification like this, otherwise the second max has to be find
    confidence_ratio = torch.min(is_positive_probs, 1-is_positive_probs) / torch.min(is_positive_probs, 1-is_positive_probs)
    return confidence_ratio
  
  