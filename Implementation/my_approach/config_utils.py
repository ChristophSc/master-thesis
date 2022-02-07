from uncertainty_sampler import UncertaintySampler_Max, UncertaintySampler_Distribution
from random_sampler import RandomSampler
from config import config

def load_sampler():
    sample_type = config().adv.sample_type
    measure_type = config().adv.measure_type
     
    sampler = None
    if sample_type == "uncertainty_max":
        sampler = UncertaintySampler_Max(measure_type)
    elif sample_type == "uncertainty_distribution":
        sampler = UncertaintySampler_Distribution(measure_type)
    elif sample_type == "random":
        sampler = RandomSampler()
    return sampler