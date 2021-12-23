import os
import sys
sys.path.append("/amax/home/gangwei/project/LightFM")
import torch.nn as nn
from data.avazu_dataset import AvazuDataset
from data.criteo_dataset import CriteoDataset
# from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
import tqdm
import numpy as np
# dataset_path = "/data2/home/gangwei/project/dataset/Avazu_raw/train"
# cache_path = "/data2/home/gangwei/project/pytorch-fm/project/.avazu"
# dataset_path = "/amax/home/gangwei/project/dataset/Criteo/train.txt"
# cache_path = "/amax/home/gangwei/project/dataset/Criteo/.criteo"
dataset_path = "/amax/home/gangwei/project/dataset/Avazu/train.txt"
cache_path = "/amax/home/gangwei/project/dataset/Avazu/.avazu"
dataset = AvazuDataset(dataset_path, cache_path)
# dataset = CriteoDataset(dataset_path, cache_path)

train_length = int(len(dataset) * 0.8)
valid_length = int(len(dataset) * 0.1)
test_length = len(dataset) - train_length - valid_length
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(19))
# self.train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=32)
train_data_loader = DataLoader(train_dataset, batch_size=10240, num_workers=32)

field_dims = dataset.field_dims
# embs = nn.Embedding(sum(field_dims),1)

from collections import Counter

cnt = np.ones(sum(field_dims))
offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
for i, (fields, target) in enumerate(tk0):
    fields = fields.numpy() + offsets
    c = Counter(fields.flatten())
    for j in c.keys():
        cnt[j] += c[j]

import pickle

offsets = np.array((0, *np.cumsum(field_dims)[:]), dtype=np.long)
cnt = np.log2(cnt)
for i in range(len(field_dims)):
    cnt[offsets[i]:offsets[i+1]] = cnt[offsets[i]:offsets[i+1]]/(cnt[offsets[i]:offsets[i+1]]).sum()

# f = open("/data2/home/gangwei/project/xlightfm/dataset/avazu_popular.pkl", "wb")
# f = open("/data2/home/gangwei/project/xlightfm/dataset/criteo_popular.pkl", "wb")
# pickle.dump(cnt, f)

print(cnt)
