import argparse

import pandas as pd
import time
import numpy as np
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description="preprocess.")
parser.add_argument('--datapath', default='../benchmarks/datasets/',
                    help='dataset folder path')
parser.add_argument('--data', default='Trivago',
                    help='dataset folder name under ../benchmarks/datasets')
args = parser.parse_args()

np.random.seed(666)

dirpath = args.datapath + args.data + '/'
if args.data == 'Gowalla':
    df = pd.read_csv(dirpath + 'Gowalla_totalCheckins.txt', header=None, delimiter='\t',
                     names=['uid', 'time', 'latitude', 'longitude', 'oid'],
                     usecols=['uid', 'oid', 'time'])
    df['time'] = df['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%dT%H:%M:%SZ')))
elif args.data == 'Trivago':
    df = pd.read_csv(dirpath + 'train.csv', header=0, delimiter=',',
                     usecols=['user_id', 'reference', 'timestamp'])
    df.columns = ['uid', 'oid', 'time']
    df = df.loc[np.random.choice(df.index, 2500000, replace=False)]
elif args.data == 'Taobao':
    df = pd.read_csv(dirpath + 'UserBehavior.csv', header=None, delimiter=',',
                     names=['uid', 'oid', 'cid', 'type', 'time'],
                     usecols=['uid', 'oid', 'time'])
    df = df.loc[np.random.choice(df.index, 2000000, replace=False)]

def del_infreq(key, threshold):
    cnts = df[key].value_counts()
    left = cnts[cnts >= threshold]
    return df[df[key].isin(left.index)]

df = del_infreq('oid', 10)
df = del_infreq('uid', 10)

def reindex(key):
    unique = df[key].unique()
    dic = pd.Series(index=unique, data=np.arange(1, len(unique) + 1))
    df[key] = df[key].apply(lambda x: dic[x])
    print(len(unique))

reindex('uid')
reindex('oid')
print(len(df))

maxlen = 20
n_neg = 5
n_test = 1000

all_oids = set(df['oid'].unique())
seqs = {}
# oid2idx, cur = {}, 1
with open(dirpath + 'train.txt', 'w') as train_f, \
        open(dirpath + 'valid.txt', 'w') as valid_f, \
        open(dirpath + 'test.txt', 'w') as test_f:
    for uid, group in tqdm(df.groupby('uid')):
        group = group.sort_values('time')
        history = group['oid'].values
        candidates = list(all_oids - set(history))
        # history = history[-(maxlen+2):]
        negs = np.random.choice(candidates, size=n_neg * (len(history) - 2), replace=False)
        test = np.random.choice(candidates, size=n_test, replace=False)

        history = [str(oid) for oid in history]
        negs = [str(oid) for oid in negs]
        test = [str(oid) for oid in test]

        train_f.write(str(uid) + '\t' + ','.join(history[:-2]) + '\t' + ','.join(negs) + '\n')
        valid_f.write(str(uid) + '\t' + ','.join(history[:-1]) + '\t' + ','.join(test) + '\n')
        test_f.write(str(uid) + '\t' + ','.join(history) + '\t' + ','.join(test) + '\n')







