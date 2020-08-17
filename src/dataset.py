import numpy as np
from torch.utils.data.dataset import Dataset
from .util import padding


class StaticDynamicDataset(Dataset):
    def __init__(self, datapath, e_feature_path=None, u_feature_path=None,
                 augment=True, maxlen=0, neg_sample=True, for_test=False):
        if for_test:
            augment = False

        self.static = []
        self.target = []
        self.feature = []
        self.dynamic = []
        self.lengths = []
        self.cur = 0

        with open(datapath, 'r') as f:
            for line in f:
                if neg_sample:
                    uid, seq, neg = line.strip().split('\t')
                    neg = [int(oid) for oid in neg.split(',')]
                else:
                    uid, seq = line.strip().split('\t')
                uid = int(uid)
                seq = [int(oid) for oid in seq.split(',')]
                if neg_sample and uid == 1:
                    self.neg_num = len(neg) if for_test else len(neg) // (len(seq) - 1)
                for i in range(1, len(seq)) if augment else [len(seq) - 1]:
                    if neg_sample:
                        self.static.append([uid] * (self.neg_num + 1))
                        self.dynamic.append([seq[:i][-maxlen:] + (maxlen - i) * [0]] * (self.neg_num + 1))
                        if for_test:
                            self.target.append([seq[i]] + neg)
                        else:
                            self.target.append([seq[i]] + neg[(i-1) * self.neg_num: i * self.neg_num])
                    else:
                        self.static.append(uid)
                        self.dynamic.append(seq[:i][-maxlen:] + (maxlen - i) * [0])
                        self.target.append(seq[i])
                    self.lengths.append(min(maxlen, i))
        self.size = len(self.static)
        self.static = np.array(self.static)
        self.dynamic = np.array(self.dynamic)
        self.lengths = np.array(self.lengths)
        self.target = np.array(self.target)

        if neg_sample:
            self.static = np.expand_dims(self.static, -1)
            self.target = np.expand_dims(self.target, -1)
        self.n_fea = 0
        if self.feature:
            self.feature = np.array(padding(self.feature))
            self.n_fea = self.feature.max()

        self.n_item = max(self.dynamic.max(), self.target.max())
        self.n_user = self.static.max()

    def __getitem__(self, indice):
        if self.feature:
            return self.static[indice], self.target[indice], self.feature[indice], self.dynamic[indice], self.lengths[indice]
        else:
            return self.static[indice], self.target[indice], self.dynamic[indice], self.lengths[indice]

    def __len__(self):
        return self.size

    # def __iter__(self):
    #     self.cur = 0
    #     if self.shuffle:
    #         self.indices = np.random.permutation(range(self.size))
    #     return self
    #
    # def __next__(self):
    #     if self.cur > self.size:
    #         raise StopIteration
    #     sample_iter = self.indices[self.cur: self.cur + self.batch_size]
    #     self.cur += self.batch_size
    #     if self.feature:
    #         return self.static[sample_iter], self.target[sample_iter], \
    #                self.feature[sample_iter], self.dynamic[sample_iter]
    #     else:
    #         return self.static[sample_iter], self.target[sample_iter], self.dynamic[sample_iter]


class StaticDynamicRatingDataset(Dataset):
    def __init__(self, datapath, e_feature_path=None, u_feature_path=None,
                 augment=True, maxlen=0, for_test=False):
        if for_test:
            augment = False

        self.static = []
        self.target = []
        self.feature = []
        self.dynamic = []
        self.lengths = []
        self.gt = []
        self.cur = 0

        with open(datapath, 'r') as f:
            for line in f:
                uid, seq, rating = line.strip().split('\t')
                uid = int(uid)
                seq = [int(oid) for oid in seq.split(',')]
                rating = [float(r) for r in rating.split(',')]
                for i in range(1, len(seq)) if augment else [len(seq) - 1]:
                    self.static.append(uid)
                    self.dynamic.append(seq[:i][-maxlen:] + (maxlen - i) * [0])
                    self.lengths.append(min(maxlen, i))
                    self.target.append(seq[i])
                    self.gt.append(rating[i])
        self.size = len(self.static)
        self.static = np.array(self.static)
        self.dynamic = np.array(self.dynamic)
        self.lengths = np.array(self.lengths)
        self.target = np.array(self.target)

        self.static = np.expand_dims(self.static, -1)
        self.target = np.expand_dims(self.target, -1)
        # self.gt = np.expand_dims(self.gt, -1)
        self.n_fea = 0
        if self.feature:
            self.feature = np.array(padding(self.feature))
            self.n_fea = self.feature.max()

        self.n_item = max(self.dynamic.max(), self.target.max())
        self.n_user = self.static.max()

    def __getitem__(self, indice):
        if self.feature:
            return (self.static[indice], self.target[indice], self.feature[indice],
                    self.dynamic[indice], self.lengths[indice]), self.gt[indice]
        else:
            return (self.static[indice], self.target[indice], self.dynamic[indice],
                    self.lengths[indice]), self.gt[indice]

    def __len__(self):
        return self.size