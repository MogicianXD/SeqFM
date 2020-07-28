import torch
import numpy as np

def mrr(ranks):
    return torch.reciprocal(ranks.float()).sum().item()


def ndcg(ranks):
    return torch.reciprocal((ranks.float() + 1).log2()).sum().item()


def bpr(y, epsilon=1e-9):
    ans = - torch.log(torch.sigmoid(y[:, 0].unsqueeze(1) - y) + epsilon)
    return ans.mean()


def padding(batch, pad=0):
    lens = [len(session) for session in batch]
    len_max = max(lens)
    batch = [session + [pad] * (len_max - l) for session, l in zip(batch, lens)]
    return batch


