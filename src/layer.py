import math

import torch
import torch.nn as nn
from torch.functional import F


def masked_softmax(x, mask, epsilon=1e-9):
    x_max = x.max(-1, keepdim=True)[0]
    x_exp = torch.exp(x - x_max) * mask.float()
    return x_exp / (x_exp.sum(dim=-1, keepdim=True) + epsilon)


def softmax_tril(x, epsilon=1e-9):
    x_max = x.max(-1, keepdim=True)[0]
    x_exp = torch.tril(torch.exp(x - x_max))
    return x_exp / (x_exp.sum(dim=-1, keepdim=True) + epsilon)


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.scale_factor = math.sqrt(dim)
        self.WQ = nn.Linear(dim, dim, bias=False)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)

    def forward(self, E, mask=False):
        W = torch.matmul(self.WQ(E), self.WK(E).transpose(-1, -2)) / self.scale_factor
        if mask:
            M = torch.tril(W.new_ones(W.shape))
            M = torch.where(W == 0, torch.zeros_like(M), M)
            W = masked_softmax(W, M)
            # W = softmax_tril(W)
            # W += torch.triu(torch.full_like(W, -1e6, device=self.device), 1)
        else:
            W = F.softmax(W, -1)
        return torch.matmul(W, self.WV(E))


class MultiHeadSelfAttetion(nn.Module):
    def __init__(self, dim, n_layers=1):
        super(MultiHeadSelfAttetion, self).__init__()
        self.layers = nn.ModuleList([SelfAttention(dim) for i in range(n_layers)])
        self.WO = nn.Linear(n_layers * dim, dim, bias=False)

    def forward(self, E, mask=False):
        heads = [attention(E, mask) for attention in self.layers]
        return self.WO(torch.cat(heads, -1))


