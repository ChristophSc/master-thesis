from uncertainty_sampler import UncertaintySampler_Basic, UncertaintySampler_Entropy_Max, UncertaintySampler_Entropy_Max_Distribution
from random_sampler import RandomSampler

def load_sampler(sample_type):
    sampler = None
    if sample_type == "uncertainty":
        sampler = UncertaintySampler_Basic
    if sample_type == "uncertainty_entropy_max":
        sampler = UncertaintySampler_Entropy_Max()
    elif sample_type == "uncertainty_entropy_max_distribution":
        sampler = UncertaintySampler_Entropy_Max_Distribution()
    elif sample_type == "random":
        sampler = RandomSampler()
    return sampler