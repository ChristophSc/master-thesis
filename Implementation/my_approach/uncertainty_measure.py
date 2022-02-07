import torch

class UncertaintyMeasure():
  def measure_uncertainty(self, probs):
    raise NotImplementedError
  
  
class Entropy(UncertaintyMeasure):
  def measure_uncertainty(self, probs):
    entropy = - (probs * torch.log2(probs)) - ((1 - probs) * torch.log2(1 - probs)) 
    return entropy 
    

class LeastConfidence(UncertaintyMeasure):
  def measure_uncertainty(self, probs):
    least_confidence = None
    return least_confidence
    
    
class ConfidenceMargin(UncertaintyMeasure):
  def measure_uncertainty(self, probs):
    confidence_margin = None
    return confidence_margin


class ConfidenceRatio(UncertaintyMeasure):
  def measure_uncertainty(self, probs):
    confidence_ratio = None
    return confidence_ratio
  
  