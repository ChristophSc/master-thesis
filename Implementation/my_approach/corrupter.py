import torch
from collections import defaultdict
import numpy as np
from numpy.random import choice, randint
from random import sample


def get_bern_prob(data, n_ent, n_rel):
    src, rel, dst = data
    edges = defaultdict(lambda: defaultdict(lambda: set()))
    rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
    for s, r, t in zip(src, rel, dst):
        edges[r][s].add(t)
        rev_edges[r][t].add(s)
    bern_prob = torch.zeros(n_rel)
    for r in edges.keys():
        tph = sum(len(tails) for tails in edges[r].values()) / len(edges[r])
        htp = sum(len(heads) for heads in rev_edges[r].values()) / len(rev_edges[r])
        bern_prob[r] = tph / (tph + htp)
    return bern_prob


class BernCorrupter(object):
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, src, rel, dst):
        prob = self.bern_prob[rel]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        ent_random = choice(self.n_ent, len(src))
        src_out = (1 - selection) * src.numpy() + selection * ent_random
        dst_out = selection * dst.numpy() + (1 - selection) * ent_random
        return torch.from_numpy(src_out), torch.from_numpy(dst_out)


class BernCorrupterMulti(object):
    def __init__(self, data, n_ent, n_rel, n_sample):
        self.bern_prob = get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent
        self.n_sample = n_sample

    def corrupt(self, head, rel, tail, keep_truth=True):
        n = len(head)
        prob = self.bern_prob[rel]
        selection = torch.bernoulli(prob).numpy().astype('bool')
        head_out = np.tile(head.numpy(), (self.n_sample, 1)).transpose()
        rel_out = rel.unsqueeze(1).expand(n, self.n_sample)
        tail_out = np.tile(tail.numpy(), (self.n_sample, 1)).transpose()        
        if keep_truth:
            ent_random = choice(self.n_ent, (n, self.n_sample - 1))
            x =  head_out[selection, 0] 
            y = tail_out[~selection, 0]
            head_out[selection, 1:] = ent_random[selection]
            tail_out[~selection, 1:] = ent_random[~selection]
        else:
            ent_random = choice(self.n_ent, (n, self.n_sample))
            head_out[selection, :] = ent_random[selection]
            tail_out[~selection, :] = ent_random[~selection]
        return torch.from_numpy(head_out), rel_out, torch.from_numpy(tail_out)
