  
import torch
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear
import torch.nn.functional as F 
from scipy.cluster.vq import vq

class NFM(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, opt):
        super().__init__()
        self.field_dims, self.dim = opt.field_dims, opt.dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        mlp_dims = (64,)
        dropouts = (0.2,0.2)

        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(self.dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(self.dim, mlp_dims, dropouts[1])

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
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
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
        for name, param in self.named_parameters():
            param.data = pre_state_dict[name]
