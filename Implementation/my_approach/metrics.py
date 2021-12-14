import torch as t

def mrr_mr_hitk(scores, target, k=10):
    _, sorted_idx = t.sort(scores, descending=False)
    find_target = sorted_idx == target
    target_rank = t.nonzero(find_target)[0, 0] + 1
    mrr, mr, hit10 =  1 / target_rank, target_rank, int(target_rank <= k)
    return mrr, mr, hit10
