import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, QuantizationEmbedding
from utils.utils import md_solver
import time
import pickle
import os
from math import log
from collections import Counter

class QFM(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.search = False

        self.field_dims, self.dim = opt.field_dims, opt.dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:]), dtype=np.long)
        self.field_len = len(self.field_dims)
        self.base_men = self.offsets[-1]*self.dim*4/1024/1024

        self.M = opt.M
        self.K = opt.K
        self.q_size = int(self.dim/self.M)
        self.time_sum = 0
        self.time_cnt = 0
        self.memory_cost =0
        for i in self.field_dims:
            if i<500:
                self.memory_cost += i*32*8
                continue
            self.memory_cost += self.K*32*8
            self.memory_cost += i*log(self.K,2)*4/8
        self.memory_cost = self.memory_cost/1024/1024

        pth = "%s_%d_%d_%d.pt" % (self.opt.model.split("_")[0], self.opt.dim, self.opt.K, self.opt.M)
        self.path = os.path.join(self.opt.pre_dir, self.opt.data_name, "quats", pth)

        with open(opt.popular_path, "rb") as f:
            popular = pickle.load(f)
        self.popular_numpy = popular
        self.popular = torch.from_numpy(popular).to(self.device)
        dim_importance = md_solver(torch.Tensor(self.field_dims.astype(np.float32)), 0.15, d0=self.dim, round_dim=False)
        self.dim_importance = (dim_importance - dim_importance.min())/(dim_importance.max() - dim_importance.min())
        print(self.dim_importance)

        # model initial
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.quantization = QuantizationEmbedding(self.field_dims, self.dim, self.K, self.M, self.device)

        # load pre model

        if opt.pre_train or opt.pre_train_quat:
            print("load pre model from %s" % opt.pre_model_path)
            pre_state_dict = torch.load(opt.pre_model_path, map_location=self.device)
            self.copy(pre_state_dict)

        if not opt.pre_train_quat:
            if opt.pre_quat:
                self.quat_copy()
            else:
                self.Embedding_pQ()
                if opt.save_quat:
                    self.quantization.save(self.path)

        # optimizer inital 
        if opt.loss == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(params=self.parameters(),
                                lr=opt.learning, weight_decay=opt.weight_decay)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        self.time_cnt += 1
        self.time_sum -= time.time()
        x_emb = self.quantization(x)
        x = self.linear(x) + self.fm(x_emb)
        self.time_sum += time.time()
        return x.squeeze(1)

    def test(self, fields, target):
        y = self(fields)
        loss = self.criterion(y, target.float()).item()
        return y, loss

    def optimize_parameters(self, fields, target):
        y = self(fields)
        self.bce_loss = self.criterion(y, target.float())
        self.loss = self.bce_loss
        self.dis_loss = 0
        # add distance loss, when loss < n
        if self.loss < 0.47 and self.opt.dis_loss != "none":
            dis, w_dis = self.calcu_distance_field()
            if self.opt.dis_loss == "avg":
                self.dis_loss = dis.mean()/3.0
                self.loss += self.dis_loss
            elif self.opt.dis_loss == "avg_import":
                self.dis_loss = (dis*self.dim_importance).mean()*0.5
                self.loss += self.dis_loss
            elif self.opt.dis_loss == "weight":
                self.dis_loss = w_dis.mean()/5.0
                self.loss += self.dis_loss
            elif self.opt.dis_loss == "weight_import":
                self.dis_loss = (w_dis*self.dim_importance).mean()*1.2
                self.loss += self.dis_loss
        self.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return {"bce_loss": self.bce_loss, 
                "dis_loss": self.dis_loss, 
                "loss": self.loss,
                "memory_cost": self.memory_cost}

    def copy(self, pre_state_dict):
        # print(pre_state_dict.keys())
        self.linear.bias.data = pre_state_dict['linear.bias'].to(self.device)
        self.linear.fc.weight.data = pre_state_dict['linear.fc.weight'].to(self.device)
        self.embedding.embedding.weight.data = pre_state_dict['embedding.embedding.weight'].to(self.device)

        if self.opt.pre_train_quat:
            self.quantization.codebooks.weight.data = pre_state_dict['quantization.codebooks.weight'].to(self.device)
            self.quantization.cb_index.weight.data = pre_state_dict['quantization.cb_index.weight'].to(self.device)

        self.linear.bias.requires_grad = False
        self.linear.fc.weight.requires_grad = False
        self.embedding.embedding.weight.requires_grad = False

        self.weigt_on_cpu = np.float32(pre_state_dict['embedding.embedding.weight'].cpu())

    def quat_copy(self):
        print("load pre quantization model from %s" % self.path)
        self.quantization.load(self.path)

    def Embedding_pQ(self, method="pq"):
        self.quantization.initial_params(raw_weight=self.weigt_on_cpu)

    def update_b(self):
        self.quantization.update_cb_index(raw_weight=self.weigt_on_cpu)

    def calcu_distance_field(self):
        plen = int(self.dim/self.M)
        distance = torch.ones(self.field_len)
        w_distance = torch.ones(self.field_len)
        for i in range(self.field_len):
            if self.field_dims[i] < self.K:
                distance[i] = 0 
                continue
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            ind = self.quantization.cb_index.weight.data[self.offsets[i]: self.offsets[i+1], ] + i*self.K
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                cluster_result[:, j*plen:j*plen+plen] = self.quantization.codebooks(ind[:, j])[:, j*plen:j*plen+plen]
            dis = F.pairwise_distance(material, cluster_result, p=2)
            distance[i] = dis.mean()
            w_distance[i] = (dis*self.popular[self.offsets[i]:self.offsets[i+1]]).sum()
        # print(distance)
        return distance, w_distance

    def initial_freq_vector(self):
        self.frequence_vector = np.zeros((sum(self.field_dims), 10))    
        self.frequence_vector[:, 0] = self.popular_numpy
        for i in range(len(self.field_dims)):
            pi = self.popular_numpy[self.offsets[i]:self.offsets[i+1]]
            order = sorted(range(len(pi)), key=lambda k:pi[k])
            self.frequence_vector[self.offsets[i]:self.offsets[i+1],1] = order
            for m in range(2):
                a = self.frequence_vector[self.offsets[i]:self.offsets[i+1],m]
                self.frequence_vector[self.offsets[i]:self.offsets[i+1],m] = (a-np.min(a))/(np.max(a)-np.min(a))

        for i in range(len(self.field_dims)):
            if self.field_dims[i]<1024:
                self.frequence_vector[self.offsets[i]:self.offsets[i+1],2:] = 0
                continue
            for j in range(4):
                cb_inds = self.quantization.cb_index.weight.data[self.offsets[i]:self.offsets[i+1],j]
                cb_inds = cb_inds.cpu().numpy()
                ind_counts_dict = Counter(cb_inds)

                ind_counts = sorted(ind_counts_dict.items(), key= lambda x:x[1], reverse=True)
                ind_rank_dict = {}
                for k in range(len(ind_counts)):
                    ind_rank_dict[ind_counts[k][0]]= k 
                # ind_proportion = [[k[0], k[1], round(1000*k[1]/self.field_dims[i],4)] for k in ind_counts]
                for k in range(self.offsets[i], self.offsets[i+1]):
                    self.frequence_vector[k,j*2+2] = ind_counts_dict[cb_inds[k-self.offsets[i]]]
                    self.frequence_vector[k,j*2+3] = ind_rank_dict[cb_inds[k-self.offsets[i]]]
                # pi = self.frequence_vector[self.offsets[i]:self.offsets[i+1],j*2+2]
                # order = np.argsort(pi)
                # self.frequence_vector[self.offsets[i]:self.offsets[i+1],j*2+3] = order
                # print(self.field_dims[i], len(ind_counts))
                # print("most: ",ind_proportion[:20])
                # print("least: ",ind_proportion[-20:])
            for m in range(2, self.frequence_vector.shape[1]):
                a = self.frequence_vector[self.offsets[i]:self.offsets[i+1],m]
                if np.max(a) == 0 and np.min(a) == 0:
                    continue
                # self.frequence_vector[self.offsets[i]:self.offsets[i+1],m] = (a-np.mean(a))/np.std(a)
                self.frequence_vector[self.offsets[i]:self.offsets[i+1],m] = (a-np.min(a))/(np.max(a)-np.min(a))