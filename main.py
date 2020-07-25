import argparse
import numpy as np
import pandas as pd
from torch import optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from src.dataset import StaticDynamicDataset, StaticDynamicRatingDataset
from src.SeqFM import SeqFM

np.random.seed(666)
torch.manual_seed(666)


def parse_args():
    parser = argparse.ArgumentParser(description="SeqFM.")
    parser.add_argument('--data', default='Trivago',
                        help='[Gowalla, Foursquare, Taobao, Trivago, Toys, Beauty]')
    parser.add_argument('--epochs', type=int, default=2400,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--dropout', type=int, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--maxlen', type=int, default=20,
                        help='Max length of seqs')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='Number of Residual FNN layers')
    parser.add_argument('--num_neg', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--candidate', type=int, default=1000,
                        help='Number of candidates when eval')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--reload', action='store_true',
                        help='restore saved params if true')
    parser.add_argument('--eval', action='store_true',
                        help='only eval once, non-train')
    parser.add_argument('--save', action='store_true',
                        help='if save model or not')
    parser.add_argument('--savepath',
                        help='for customization')
    parser.add_argument('--cuda', default='0',
                        help='gpu No.')
    return parser.parse_args()


dataset_path = '../benchmarks/datasets/'
args = parse_args()
if not args.savepath:
    args.savepath = 'checkpoints/' + args.data + '.model'
print(args)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

if args.data in ['Gowalla', 'Foursquare']:
    task = 'rank'
elif args.data in ['Trivago', 'Taobao']:
    task = 'classification'
elif args.data in ['Beauty', 'Toys']:
    task = 'regression'
else:
    raise Exception('Please settle for new dataset')

if task == 'regression':
    Dataset = StaticDynamicRatingDataset
else:
    Dataset = StaticDynamicDataset
train = Dataset(dataset_path + args.data + '/train.txt', maxlen=args.maxlen)
valid = Dataset(dataset_path + args.data + '/valid.txt', maxlen=args.maxlen, for_test=True)
test = Dataset(dataset_path + args.data + '/test.txt', maxlen=args.maxlen, for_test=True)

static_u_m = max(train.n_user, valid.n_user, test.n_user)
dynamic_m = max(train.n_item, valid.n_item, test.n_item)
feature_m = max(train.n_fea, valid.n_fea, test.n_fea)
print(static_u_m, dynamic_m, feature_m, len(train) + len(valid) + len(test) * 2)

model = SeqFM(use_cuda=True, static_u_m=static_u_m, feature_m=feature_m, dynamic_m=dynamic_m,
              emb_dim=args.emb_dim, dropout=args.dropout, n_layer=args.n_layer)

test_bs = args.batch_size if task == 'regression' else 8
train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
valid = DataLoader(valid, batch_size=test_bs)
test = DataLoader(test, batch_size=test_bs)

fn = {'rank': (model.fit_bpr, model.test_rank_with_neg),
      'classification': (model.fit_nll_neg, model.test_classify),
      'regression': (model.fit_mse, model.test_regression)}

model.train_test(fn[task][0], fn[task][1], train, valid, test, savepath=args.savepath,
                 n_epochs=args.epochs, lr=args.lr, topk=task=='rank', small_better=task=='regression')
