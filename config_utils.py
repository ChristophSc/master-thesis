from uncertainty_sampler import UncertaintySampler, UncertaintySampler_Max, UncertaintySampler_Softmax
from original_sampler import ConfidenceSampler, OriginalSampler
from config import config

def load_sampler():
    sample_type = config().adv.sample_type
    measure_type = config().adv.measure_type
     
    sampler = None
    if sample_type == "uncertainty_max":
        sampler = UncertaintySampler_Max(measure_type)
    elif sample_type == "uncertainty_softmax":
        sampler = UncertaintySampler_Softmax(measure_type)
    elif sample_type == "original":
        sampler = OriginalSampler()
    elif sample_type == "confidence":
        sampler = ConfidenceSampler()
    return sampler