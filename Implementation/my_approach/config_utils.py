from uncertainty_sampler import UncertaintySampler_Basic, UncertaintySampler_Advanced
from random_sampler import RandomSampler

def load_sampler(sample_type):
    sampler = None
    if sample_type == "uncertainty":
        sampler = UncertaintySampler_Advanced()
    elif sample_type == "random":
        sampler = RandomSampler()
    return sampler