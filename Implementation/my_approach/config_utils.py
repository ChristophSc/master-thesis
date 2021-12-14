from uncertainty_sampler import UncertaintySampler
from random_sampler import RandomSampler

def load_sampler(sample_type):
    sampler = None
    if sample_type == "uncertainty":
        sampler = UncertaintySampler()
    elif sample_type == "random":
        sampler = RandomSampler()
    return sampler