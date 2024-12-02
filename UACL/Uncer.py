import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''

    def __init__(self, in_feat=512, out_feat=1):
        super(UncertaintyHead, self).__init__()
        self.convf_dim = in_feat
        self.out_dim = out_feat

        self.log_kappa = nn.Sequential(
            nn.Linear(self.convf_dim, self.convf_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.convf_dim // 2, self.convf_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.convf_dim // 4, self.out_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.log_kappa(x)
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


class UncertaintyHead_MLS(nn.Module):
    ''' Evaluate the log(sigma^2) '''

    def __init__(self, in_feat=512):
        super(UncertaintyHead_MLS, self).__init__()
        self.fc1 = Parameter(torch.Tensor(in_feat, in_feat))
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU(in_feat)
        self.fc2 = Parameter(torch.Tensor(in_feat, in_feat))
        self.bn2 = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))  # default = -7.0

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


if __name__ == "__main__":

    unh = UncertaintyHead_MLS(in_feat=512)

    # mu_data = np.array([[-1.7847768, -1.0991699, 1.4248079],
    #                     [1.0405252, 0.35788524, 0.7338794],
    #                     [1.0620259, 2.1341069, -1.0100055],
    #                     [-0.00963581, 0.39570177, -1.5577421],
    #                     [-1.064951, -1.1261107, -1.4181522],
    #                     [1.008275, -0.84791195, 0.3006532],
    #                     [0.31099692, -0.32650718, -0.60247767]])
    #
    # muX = torch.from_numpy(mu_data).float()
    muX = torch.randn(8, 512, 1, 1)
    log_sigma_sq = unh(muX)
    print(log_sigma_sq.shape)

    # unh = UncertaintyHead(in_feat=512, out_feat=1)
    # muX = torch.randn(8, 512, 1, 1)
    # log_sigma_sq = unh(muX)
    # print(log_sigma_sq.shape)