import torch
from torch._C import device

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

import time

class FM(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, opt):
        super().__init__()
        field_dims, embed_dim = opt.field_dims, opt.dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.time_sum = 0
        self.time_cnt = 0

        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        if opt.pre_train:
            print("load pre model from %s" % opt.pre_model_path)
            pre_state_dict = torch.load(opt.pre_model_path, map_location=self.device)
            self.copy(pre_state_dict)

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
        x = self.linear(x) + self.fm(self.embedding(x))
        self.time_sum += time.time()
        return x.squeeze(1)

    def test(self, fields, target):
        y = self(fields)
        loss = self.criterion(y, target.float()).item()
        return y, loss

    def optimize_parameters(self, fields, target):
        y = self(fields)
        self.bce_loss = self.criterion(y, target.float())
        self.zero_grad()
        self.bce_loss.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return {"bce_loss": self.bce_loss}

    def copy(self, pre_state_dict):
        self.linear.bias.data = pre_state_dict['linear.bias'].to(self.device)
        self.linear.fc.weight.data = pre_state_dict['linear.fc.weight'].to(self.device)
        self.embedding.embedding.weight.data = pre_state_dict['embedding.embedding.weight'].to(self.device)
        