import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

import os
import sys
sys.path.append("/data2/home/gangwei/project/xlightfm")

from utils import md_solver

from dataset.avazu import AvazuDataset
from dataset.criteo import CriteoDataset
from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset

from model.fm import FactorizationMachineModel
from model.qfm import QuantizationFactorizationMachine
from model.qfm_bc import BinRecQuantizationFM
from model.qfm_gs import GumbelSoftmaxQuantizationFM
from model.dfm import DFactorizationMachineModel
from model.qrfm import QRFactorizationMachineModel
from model.mdfm import MDFactorizationMachineModel
from model.dhefm import DHEFactorizationMachineModel
from model.nfm import NeuralFactorizationMachineModel
from model.deepfm import DeepFactorizationMachineModel
from model.qdeepfm import QuatDeepFactorizationMachineModel
from model.qnfm import QuatNeuralFactorizationMachineModel
# from model.dfm_test import DFactorizationMachineModel

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def set_dataloader(name=None):
    dataset = None
    dataset_path = "/data2/home/gangwei/project/dataset/Avazu_raw/train"
    cache_path = "/data2/home/gangwei/project/pytorch-fm/project/.avazu"
    # popular_path = 

    dataset = AvazuDataset(dataset_path, cache_path)

    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(19))
    data_loader = DataLoader(test_dataset, batch_size=2048, num_workers=32)

    return dataset.field_dims, data_loader

if __name__ == "__main__":
    params = {}
    field_dims, data_loader = set_dataloader()
    params["field_dims"] = field_dims
    params["dim"] = 32
    params["device"] = "cpu"
    params["M"] = 4
    params["share"] = 0
    params["K"] = 1024
    params["popular_path"] = "/data2/home/gangwei/project/xlightfm/dataset/avazu_popular.pkl"
    device = torch.device(params["device"])
    
    # model = QRFactorizationMachineModel(params).to(device)
    # model = MDFactorizationMachineModel(params).to(device)
    # model = FactorizationMachineModel(params).to(device)
    # model = DFactorizationMachineModel(params).to(device)
    # model = DHEFactorizationMachineModel(params).to(device)
    # model = QuantizationFactorizationMachine(params).to(device)
    model = GumbelSoftmaxQuantizationFM(params).to(device)
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        if i > 100:
            break
        # print(fields.shape)
        fields = fields.to(device=device, dtype=torch.long)
        y = model(fields)
        
    print(model.time_sum/ model.time_cnt)