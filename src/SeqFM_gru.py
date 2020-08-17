import torch
import torch.nn as nn
from src.layer import *
from src.BaseModel import BaseModel


class ResFNN(nn.Module):
    def __init__(self, dim, dropout, n_layer=1):
        super(ResFNN, self).__init__()
        self.layerNorm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=True)
        self.n_layer = n_layer
        self.dropout = nn.Dropout(1 - dropout)

    def forward(self, x):
        for i in range(self.n_layer):
            x = x + self.dropout(F.relu(self.linear(self.layerNorm(x))))
        return x


class SeqFM(BaseModel):
    def __init__(self, static_u_m, feature_m, dynamic_m, emb_dim,
                 dropout, n_layer=1, n_head=1, use_cuda=True, unshared=False, pos_emb_dim=0):
        super(SeqFM, self).__init__(use_cuda=use_cuda)
        self.feature = feature_m > 0
        self.emb_dim = emb_dim
        self.static_U = nn.Embedding(static_u_m + 1, emb_dim, padding_idx=0)
        if self.feature:
            self.feature_E = nn.Embedding(feature_m + 1, emb_dim, padding_idx=0)
        self.static_E = nn.Embedding(dynamic_m + 1, emb_dim, padding_idx=0)
        self.dynamic_E = nn.Embedding(dynamic_m + 1, emb_dim, padding_idx=0)
        if n_head > 1:
            self.static_a = MultiHeadSelfAttetion(emb_dim, n_head)
            self.dynamic_a = MultiHeadSelfAttetion(emb_dim, n_head)
            self.cross_a = MultiHeadSelfAttetion(emb_dim, n_head)
        else:
            self.static_a = SelfAttention(emb_dim)
            self.dynamic_a = SelfAttention(emb_dim)
            self.cross_a = SelfAttention(emb_dim)
        self.fnn = ResFNN(emb_dim, dropout, n_layer)
        self.unshared = unshared
        if unshared:
            self.fnn2 = ResFNN(emb_dim, dropout, n_layer)
            self.fnn3 = ResFNN(emb_dim, dropout, n_layer)
        self.p = nn.Linear(emb_dim * 3, 1, bias=True)
        self.static_u_w = nn.Embedding(static_u_m + 1, 1, padding_idx=0)
        self.feature_w = nn.Embedding(feature_m + 1, 1, padding_idx=0)
        self.static_e_w = nn.Embedding(dynamic_m + 1, 1, padding_idx=0)
        self.dynamic_w = nn.Embedding(dynamic_m + 1, 1, padding_idx=0)
        self.dropout = nn.Dropout(1 - dropout)
        self.static_layer_norm = nn.LayerNorm(emb_dim)
        self.dynamic_layer_norm = nn.LayerNorm(emb_dim)
        self.cross_layer_norm = nn.LayerNorm(emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim)

    def forward(self, X):
        for i in range(len(X)):
            X[i] = X[i].to(self.device)
        if self.feature:
            static_U_X, static_E_X, feature_X, dynamic_X, lengths = X
        else:
            static_U_X, static_E_X, dynamic_X, lengths = X
        static_U = self.static_U(static_U_X)
        static_E = self.static_E(static_E_X)
        if self.feature:
            feature_E = self.feature_E(self.feature)
            static_E = torch.cat((static_U, static_E, feature_E), -2)
        else:
            static_E = torch.cat((static_U, static_E), -2)
        static_E = self.dropout(static_E)
        static_H = self.static_a(static_E, mask=False)
        static_H = static_H.mean(-2)
        # static_H = self.static_layer_norm(static_H)
        static_H = self.fnn(static_H)

        # lengths = (dynamic_X > 0).sum(-1)
        dynamic_freq = 1 / lengths.float().unsqueeze(-1)
        if dynamic_X.dim() == 3:
            dynamic_freq = dynamic_freq.unsqueeze(-1)
        dynamic_E = self.dynamic_E(dynamic_X) * dynamic_freq.unsqueeze(-1)
        dynamic_E = self.dropout(dynamic_E)
        dynamic_H = self.dynamic_a(dynamic_E, mask=True)

        dynamic_H = dynamic_H.view(dynamic_H.shape[-2], -1, dynamic_H.shape[-1])
        _, dynamic_H = self.gru(dynamic_H)
        dynamic_H = dynamic_H.squeeze(0)
        # dynamic_H = self.dynamic_layer_norm(dynamic_H)
        if dynamic_X.dim() == 3:
            dynamic_H = dynamic_H.view(dynamic_X.shape[0], dynamic_X.shape[1], dynamic_H.shape[-1])
        if self.unshared:
            dynamic_H = self.fnn2(dynamic_H)
        else:
            dynamic_H = self.fnn(dynamic_H)

        cross_E = torch.cat((static_E, dynamic_E), dim=-2)
        cross_H = self.cross_a(cross_E, mask=True, static_dim=static_E.shape[-2])
        cross_H = cross_H.mean(-2)
        # cross_H = self.cross_layer_norm(cross_H)
        if self.unshared:
            cross_H = self.fnn3(cross_H)
        else:
            cross_H = self.fnn(cross_H)

        h_agg = torch.cat((static_H, dynamic_H, cross_H), dim=-1)

        f = self.p(h_agg)

        y = f + self.static_u_w(static_U_X).sum(-2) + \
            self.static_e_w(static_E_X).sum(-2) + \
            self.dynamic_w(dynamic_X).sum(-2) * dynamic_freq     # (bs,)
        if self.feature:
            y += self.feature_w(feature_X).sum(-2)
        return y.squeeze(-1)





