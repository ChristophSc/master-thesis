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
    least_confidence = None
    return least_confidence
    
    
class ConfidenceMargin(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
    confidence_margin = None
    return confidence_margin


class ConfidenceRatio(UncertaintyMeasure):
  def measure_uncertainty(self, is_positive_probs):
    confidence_ratio = None
    return confidence_ratio
  
  