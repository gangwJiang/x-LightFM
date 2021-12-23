import torch
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, opt):
        super().__init__()     
        self.field_dims, self.dim = opt.field_dims, opt.dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        
        mlp_dims = (16,16)
        dropout = 0.2

        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.embed_output_dim = len(self.field_dims) * self.dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

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
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
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

