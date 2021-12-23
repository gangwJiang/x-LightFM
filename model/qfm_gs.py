import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MonopolySumQuatEmbedding, WeightedSumQuatEmbedding
from utils.utils import md_solver
import time
from torch.autograd import Variable
import os
import pickle


K_Space = [1, 64, 128, 256, 512, 1024, 2048]
# K_Space = [1, 64, 128, 256, 512]


def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])

class QFMgs(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.search = True

        self.field_dims, self.dim = opt.field_dims, opt.dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        self.M = opt.M
        self.K = opt.K
        self.share = opt.share
        self.cnt = 0
        self.threshold = opt.threshold
        self.q_size = int(self.dim/self.M)
        self.field_len = len(self.field_dims)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:]), dtype=np.long)
        self.base_men = self.offsets[-1]*self.dim*4/1024/1024
        self.memory_cost = self.base_men
        print(self.memory_cost)
        self.time_sum = 0
        self.time_cnt = 0
        self.temperature = 0
        self.optim_iter = 0

        pth = "%s_x_%d_%d_%d.pt" % (self.opt.model.split("_")[0], self.opt.dim, self.opt.K, self.opt.M)
        self.path = os.path.join(self.opt.pre_dir, self.opt.data_name, "quats", pth)

        with open(opt.popular_path, "rb") as f:
            popular = pickle.load(f)
        self.popular_numpy = popular
        self.popular = torch.from_numpy(popular).to(self.device)
        dim_importance = md_solver(torch.Tensor(self.field_dims.astype(np.float32)), 0.15, d0=self.dim, round_dim=False)
        self.dim_importance = (dim_importance - dim_importance.min())/(dim_importance.max() - dim_importance.min())
        # print(self.dim_importance)

        self._arch_parameters = Variable(
            torch.ones((self.field_len, len(K_Space)), dtype=torch.float, device=self.device) / 2, requires_grad=True)
        self._arch_parameters.data.add_(torch.randn_like(self._arch_parameters)*1e-3)
        # the K_Space space which is avliable
        self.prior_flag = torch.ones(self.field_len, len(K_Space), device=self.device)* -1e5
        for i in range(self.field_len):
            if self.field_dims[i] < self.threshold:
                self.prior_flag[i, 0] = 1
            for k in range(1, len(K_Space)):
                if K_Space[k]*2.5 > self.field_dims[i]:
                    break
                self.prior_flag[i, k] = 1
                self._arch_parameters.data[i, k] -= 0.002*k
            # if self.field_dims[i] > 190000:
            #     self._arch_parameters.data[i, len(K_Space)-1] = 0.85
        self.arch_prob = torch.zeros(self.field_len, len(K_Space), device=self.device)
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        if self.share == 1:
            self.quantization = WeightedSumQuatEmbedding(self.field_dims, self.dim, self.K, self.M, K_Space, self.device, opt)
        else:
            self.quantization = MonopolySumQuatEmbedding(self.field_dims, self.dim, self.K, self.M, K_Space, self.device, opt)
        
        if opt.pre_train:
            print("load pre model from %s" % opt.pre_model_path)
            pre_state_dict = torch.load(opt.pre_model_path, map_location=self.device)
            self.copy(pre_state_dict)
            print("load finish ... ...")
        if opt.pre_quat:
            self.quat_copy()
            print("load finish ... ...")
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
        self.arch_optimizer = torch.optim.Adam(params=self.arch_parameters(),
                                    lr=opt.arch_learning, weight_decay=opt.weight_decay)

        self.fix_arch = False
        self.arch_loss = 0
        self.bce_loss = 0
        self.dis_loss = 0
        self.loss = 0

    def forward(self, x, flag=0):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        self.time_cnt += 1
        self.time_sum -= time.time()
        # print(self.arch_prob, self._arch_parameters, self.prior_flag)
        x_emb = self.quantization(x, self.arch_prob, self._arch_parameters, self.prior_flag, self.temperature, flag)
        x = self.linear(x) + self.fm(x_emb)
        self.time_sum += time.time()
        return x.squeeze(1)

    def test(self, fields, target):
        self.arch_argmax()
        y = self(fields, flag=1)
        loss = self.criterion(y, target.float()).item()
        return y, loss

    def optimize_parameters(self, fields, target):
        # update arch parameters
        self.zero_grad()

        if not self.fix_arch and self.optim_iter % self.opt.frequence == 0:
            self.arch_optimizer.zero_grad()
            self.temperature = max(0.01, 1-5e-5*self.optim_iter)

            valid_fields, valid_target = next(self.valid_data_iter)
            valid_fields = valid_fields.to(device=self.device, dtype=torch.long)
            valid_target = valid_target.to(self.device)
            self.valid_cnt += 1
            if self.valid_cnt > len(self.valid_data_iter)-3:
                self.valid_cnt = 0
                self.valid_data_iter = iter(self.valid_data_loader)
            if self.opt.unrolled:
                self.arch_loss = self._backward_step_unrolled(valid_fields, valid_target, valid_fields, valid_target, 1e-4)
            else:
                self.arch_loss = self._backward_step(valid_fields, valid_target)

            self.calcu_memory_cost()
            if self.opt.memory_limit == -1:        
                self.arch_optimizer.step()
            else:
                last = self._arch_parameters.clone()
                grad =self._arch_parameters.grad.clone()
                self.arch_optimizer.step()
                self.calcu_memory_cost()
                if self.memory_cost/self.base_men > 1.0/self.opt.memory_limit:
                    self.arch_back_update(last, grad)
                    self.calcu_memory_cost()

        # update model parameters
        if self.fix_arch:
            y = self(fields, flag=1)
        else:
            y = self(fields)
        self.bce_loss = self.criterion(y, target.float())
        self.loss = self.bce_loss
        self.dis_loss = 0
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
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.optimizer.step()
        self.clip()

        self.optim_iter += 1

    def get_current_losses(self):
        return {"arch_loss": self.arch_loss,
                "bce_loss": self.bce_loss, 
                "dis_loss": self.dis_loss, 
                "loss": self.loss,
                "memory_cost": self.memory_cost}

    def arch_parameters(self):
        return [self._arch_parameters]

    def random_arch(self):
        arch = []
        for i in range(self.field_len):
            ind = np.random.randint(0, len(K_Space))
            while self.prior_flag[i, ind] != 1:
                ind = np.random.randint(0, len(K_Space))
            arch.append([self.field_dims[i], K_Space[ind]])
        return arch

    def set_arch(self, arch):
        for i in range(self.arch_prob.shape[0]):
            for j in range(len(K_Space)):
                if K_Space[j] == arch[i][1]:
                    m = j
                    break
            for j in range(self.arch_prob.shape[1]):
                if j == m:
                    self.arch_prob[i,j] = 1
                else:
                    self.arch_prob[i,j] = 0
        self.calcu_memory_cost()

    def initial_validset(self, valid_data_loader):
        self.valid_data_loader = valid_data_loader
        self.valid_data_iter = iter(self.valid_data_loader)
        self.valid_cnt = 0

    def arch_back_update(self, last, grad):
        if self.opt.limit_method == 0:
            # set the arch parameter back to older state when the growth leading in a heavier comsuption
            for i in range(self.field_len):
                s = (last[i, :]*self.prior_flag[i, :]).argmax().item()
                for j in range(s+1, len(K_Space)):
                    if self._arch_parameters[i, j] > last[i, j]:
                        self._arch_parameters[i, j].data.fill_(last[i, j]) 
        elif self.opt.limit_method == 1:
            # set current arch to a lower or higher probability based on the rank of the gradient
            grads_info = grad.cpu().numpy().flatten()
            grads_order = sorted(range(len(grads_info)), key=lambda k:grads_info[k])
            for i in range(self.field_len):
                s = (last[i, :]*self.prior_flag[i, :]).argmax().item()
                if self._arch_parameters[i, s] < last[i, s]:
                    self._arch_parameters[i, s].data.fill_(last[i, s]) 
                for j in range(s+1, len(K_Space)):
                    if self._arch_parameters[i, j] > last[i, j]:
                        if grads_order[i*len(K_Space)+j] > 3:
                            self._arch_parameters[i, j].data.fill_(last[i, j]) 
        elif self.opt.limit_method == 2:
            # set the arch parameter back to older state when the growth leading in a heavier comsuption
            for i in range(self.field_len):
                s = (last[i, :]*self.prior_flag[i, :]).argmax().item()
                if self._arch_parameters[i, s] < last[i, s]:
                    self._arch_parameters[i, s].data.fill_(last[i, s]) 
                for j in range(s+1, len(K_Space)):
                    if self._arch_parameters[i, j] > last[i, j]:
                        self._arch_parameters[i, j].data.fill_(last[i, j]) 
        elif self.opt.limit_method == 3:
            # set the arch parameter back to older state when the growth leading in a heavier comsuption
            grads_info = (grad.cpu().numpy()*self.prior_flag).sum(axis=1)/self.prior_flag.sum(axis=1)
            grads_order = sorted(range(len(grads_info)), key=lambda k:grads_info[k])
            for i in range(self.field_len):
                s = (last[i, :]*self.prior_flag[i, :]).argmax().item()
                if self._arch_parameters[i, s] < last[i, s]:
                    self._arch_parameters[i, s].data.fill_(last[i, s]) 
                for j in range(s+1, len(K_Space)):
                    if self._arch_parameters[i, j] > last[i, j]:
                        self._arch_parameters[i, j].data.fill_(last[i, j]) 
                if grads_order[i] > len(grads_order)/2:
                    for j in range(s):
                        self._arch_parameters[i, j].data.add_(0.0005)


    def arch_argmax(self):
        ind = []
        for i in range(self.arch_prob.shape[0]):
            if self.fix_arch:
                m = self.arch_prob[i, :].argmax().item()
            else:
                m = (self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()
            for j in range(self.arch_prob.shape[1]):
                if j == m:
                    self.arch_prob[i,j] = 1
                else:
                    self.arch_prob[i,j] = 0
            ind.append(m)
        return ind 

    def calcu_memory_cost(self):
        cost = 0.0
        for i in range(self.field_len):
            if self.fix_arch:
                select=self.arch_prob[i, :].argmax().item()
            else:
                select=(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()
            select =  K_Space[select]
            if select==1:
                cost += self.field_dims[i]*4*self.dim
                continue
            cost += select*4*self.dim
            cost += self.field_dims[i]*np.log2(select)*self.M/8
        self.memory_cost = cost/1024/1024
        # print(cost/float(self.base_men))
        return cost/float(self.base_men)

    def genotype(self):
        genotype = []
        gen = []
        for i in range(self.field_len):
            # if self.prior_flag[i, 0] == 1:
            #     continue
            if self.fix_arch:
                pos=self.arch_prob[i, :].argmax().item()
            else:
                pos=(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()
            genotype.append([self.field_dims[i], K_Space[pos]])
            gen.append(genotype[i][1])
            # print(self.field_dims[i], self._arch_parameters[i, :].data)
        # print(genotype)
        # genotype = [(self.field_dims[i],K_Space[(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()]) for i in range(self.field_len)]
        return genotype

        
    def clip(self):
        m = nn.Hardtanh(0.01, 1)
        self._arch_parameters.data = m(self._arch_parameters)

    def _backward_step(self, x, labels):
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        loss.backward()
        return loss

    def copy(self, pre_state_dict):
        self.linear.bias.data = pre_state_dict['linear.bias'].to(self.device)
        self.linear.fc.weight.data = pre_state_dict['linear.fc.weight'].to(self.device)
        self.embedding.embedding.weight.data = pre_state_dict['embedding.embedding.weight'].to(self.device)

        self.linear.bias.requires_grad = False
        self.linear.fc.weight.requires_grad = False
        self.embedding.embedding.weight.requires_grad = False

        self.weigt_on_cpu = np.float32(pre_state_dict['embedding.embedding.weight'].cpu())

    def quat_copy(self):
        print("load pre quantization model from %s" % self.path)
        self.quantization.load(self.path)

    def Embedding_pQ(self):
        self.quantization.initial_params(raw_weight=self.weigt_on_cpu)

    def update_b(self):
        self.quantization.update_cb_index(raw_weight=self.weigt_on_cpu, flag=self.prior_flag)

    def calcu_distance_field(self):
        plen = int(self.dim/self.M)
        used_ind = self.arch_argmax()
        distance = torch.ones(self.field_len)
        w_distance = torch.ones(self.field_len)
        for i in range(self.field_len):
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            if used_ind[i] == 0:
                distance[i] = 0
                continue
            else:
                ind = self.quantization.cb_index[used_ind[i]].weight.data[self.offsets[i]: self.offsets[i+1], ] + i*K_Space[used_ind[i]]
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                cluster_result[:, j*plen:j*plen+plen] = self.quantization.codebooks[used_ind[i]](ind[:, j])[:, j*plen:j*plen+plen]
            dis = F.pairwise_distance(material, cluster_result, p=2)
            distance[i] = dis.mean()
            w_distance[i] = (dis*self.popular[self.offsets[i]:self.offsets[i+1]]).sum()
            # distance_max[i] = F.pairwise_distance(material, cluster_result, p=2).max()
        return distance, w_distance
    
    def new(self):
        model_new = QFMgs(self.params).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def used_parameters(self):
        for name, param in self.named_parameters(recurse=True):
            if param.requires_grad:
                # print(name)
                yield param

    def _backward_step_unrolled(self, x_train, labels_train,
		                            x_valid, labels_valid, lr):
        unrolled_model = self._compute_unrolled_model(x_train, labels_train, lr)
        unrolled_model.g_softmax(self.temperature)
        unrolled_inference = unrolled_model(x_valid)
        unrolled_loss = self.criterion(unrolled_inference, labels_valid.float())
        
        unrolled_loss.backward()
        
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad for v in unrolled_model.used_parameters()]
        # print(vector)
        implicit_grads = self._hessian_vector_product(vector, x_train, labels_train)
        
        for g,ig in zip(dalpha,implicit_grads):
            g.sub_(lr, ig)
        
        for v,g in zip(self.arch_parameters(), dalpha):
            v.grad = g.clone()
        return unrolled_loss
    
    def _compute_unrolled_model(self, x, labels, lr):
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        # print(type(self.used_parameters()))
        theta = _concat(self.used_parameters())
        dtheta = _concat(torch.autograd.grad(loss, self.used_parameters()))
        unrolled_model = self._construct_model_from_theta(theta.sub(dtheta, alpha=lr))
        return unrolled_model
    
    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()
        params, offset = {}, 0
        for k,v in self.named_parameters():
            if v.requires_grad:
                v_length = np.prod(v.size())
                params[k] = theta[offset: offset+v_length].view(v.size())
                offset += v_length
            else:
                params[k] = v.clone()
        # print(params)
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        model_new.linear.bias.requires_grad = False
        model_new.linear.fc.weight.requires_grad = False
        model_new.embedding.embedding.weight.requires_grad = False
        return model_new.to(self.device)
    
    def _hessian_vector_product(self, vector, x, labels, r=1e-2):
        R = r / _concat(vector).norm()
        for p,v in zip(self.used_parameters(), vector):
            p.data.add_(R, v)
            
        self.g_softmax(self.temperature)
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.used_parameters(), vector):
            p.data.sub_(2*R, v)
        
        self.g_softmax(self.temperature)
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.used_parameters(), vector):
            p.data.add_(R, v)

        return [(i-y).div_(2*R) for i,y in zip(grads_p,grads_n)]

