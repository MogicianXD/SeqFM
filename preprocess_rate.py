import argparse

import pandas as pd
import time
import numpy as np
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description="preprocess.")
parser.add_argument('--datapath', default='../benchmarks/datasets/',
                    help='dataset folder path')
parser.add_argument('--data', default='Toys',
                    help='[Toys, Beauty]')
args = parser.parse_args()

np.random.seed(666)

dirpath = args.datapath + args.data + '/'
df = pd.read_csv(dirpath + 'data.csv', header=0, delimiter='\t',
                 names=['uid', 'oid', 'rating', 'time'])

def del_infreq(key, threshold):
    cnts = df[key].value_counts()
    left = cnts[cnts >= threshold]
    return df[df[key].isin(left.index)]

# df = del_infreq('oid', 10)
# df = del_infreq('uid', 10)

def reindex(key):
    unique = df[key].unique()
    dic = pd.Series(index=unique, data=np.arange(1, len(unique) + 1))
    df[key] = df[key].apply(lambda x: dic[x])
    print(len(unique))

reindex('uid')
reindex('oid')

print(len(df))

maxlen = 20

with open(dirpath + 'train.txt', 'w') as train_f, \
        open(dirpath + 'valid.txt', 'w') as valid_f, \
        open(dirpath + 'test.txt', 'w') as test_f:
    for uid, group in tqdm(df.groupby('uid')):
        group = group.sort_values('time')
        history = group['oid'].values
        rating = group['rating'].values
        history = [str(oid) for oid in history]
        rating = [str(r) for r in rating]

        train_f.write(str(uid) + '\t' + ','.join(history[:-2]) + '\t' + ','.join(rating[:-2]) + '\n')
        valid_f.write(str(uid) + '\t' + ','.join(history[:-1]) + '\t' + ','.join(rating[:-1]) + '\n')
        test_f.write(str(uid) + '\t' + ','.join(history) + '\t' + ','.join(rating) + '\n')







