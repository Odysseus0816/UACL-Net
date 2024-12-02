import torch
import torch.nn as nn
import torch.nn.functional as F
from ive import *
import math


class MLSLoss(nn.Module):

    def __init__(self, mean=False):

        super(MLSLoss, self).__init__()
        self.mean = mean

    def negMLS(self, mu_X, sigma_sq_X):

        if self.mean:
            XX = torch.mul(mu_X, mu_X).sum(dim=1, keepdim=True)
            YY = torch.mul(mu_X.T, mu_X.T).sum(dim=0, keepdim=True)
            XY = torch.mm(mu_X, mu_X.T)
            mu_diff = XX + YY - 2 * XY
            sig_sum = sigma_sq_X.mean(dim=1, keepdim=True) + sigma_sq_X.T.sum(dim=0, keepdim=True)
            diff = mu_diff / (1e-8 + sig_sum) + mu_X.size(1) * torch.log(sig_sum)
            return diff
        else:
            mu_diff = mu_X.unsqueeze(1) - mu_X.unsqueeze(0)
            sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
            diff = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(sig_sum)  # BUG
            diff = diff.sum(dim=2, keepdim=False)
            return diff

    def forward(self, mu_X, log_sigma_sq, gty):

        mu_X = F.normalize(mu_X)  # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if gty.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)
        sig_X = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
        # pos_loss = torch.abs(pos_loss)
        return pos_loss


class KLDiracVMF(nn.Module):
    def __init__(self, z_dim, radius):
        super().__init__()
        self.z_dim = z_dim
        self.radius = radius

    def forward(self, mu, kappa, wc):
        # mu and wc: (B, dim)
        # kappa: (B, 1)

        B = mu.size(0)
        d = self.z_dim
        r = self.radius

        log_ive_kappa = torch.log(1e-6 + ive(d / 2 - 1, kappa))
        log_iv_kappa = log_ive_kappa + kappa

        cos_theta = torch.sum(mu * wc, dim=1, keepdim=True) / r

        l1 = -kappa * cos_theta
        l2 = - (d / 2 - 1) * torch.log(1e-6 + kappa)
        l3 = log_iv_kappa * 1.0

        losses = l1 + l2 + l3 \
                 + (d / 2) * math.log(2 * math.pi) \
                 + d * math.log(r)

        return losses, l1, l2, l3

class SphereFace(nn.Module):
    def __init__(self, feature_num, class_num, s=10, m=0.5):
        super().__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.s = s
        self.m = torch.tensor(m)
        self.w = nn.Parameter(torch.rand(feature_num, class_num), requires_grad=True)  # 2*10

    def forward(self, feature):
        feature = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.w, dim=0)
        cos_theat = torch.matmul(feature, w) / 10
        sin_theat = torch.sqrt(1.0 - torch.pow(cos_theat, 2))
        cos_theat_m = cos_theat * torch.cos(self.m) - sin_theat * torch.sin(self.m)
        cos_theat_ = torch.exp(cos_theat * self.s)
        sum_cos_theat = torch.sum(torch.exp(cos_theat * self.s), dim=1, keepdim=True) - cos_theat_
        top = torch.exp(cos_theat_m * self.s)
        div = top / (top + sum_cos_theat)
        return div

#
# # 实现方式2
# class SphereFace(nn.Module):
#     def __init__(self, feature_dim=2, cls_dim=10):
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)
#
#     def forward(self, feature, m=1, s=10):
#         x = nn.functional.normalize(feature, dim=1)
#         w = nn.functional.normalize(self.W, dim=0)
#         cos = torch.matmul(x, w) / 10  # 求两个向量夹角的余弦值
#         a = torch.acos(cos)  # 反三角函数求得 α
#         top = torch.exp(s * torch.cos(a + m))  # e^(s * cos(a + m))
#         down2 = torch.sum(torch.exp(s * torch.cos(a)), dim=1, keepdim=True) - torch.exp(s * torch.cos(a))
#         out = torch.log(top / (top + down2))
#         return out


if __name__ == "__main__":
    # mls = MLSLoss(mean=False)
    #
    # gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    # mu_data = np.array([[-1.7847768, -1.0991699, 1.4248079],
    #                     [1.0405252, 0.35788524, 0.7338794],
    #                     [1.0620259, 2.1341069, -1.0100055],
    #                     [-0.00963581, 0.39570177, -1.5577421],
    #                     [-1.064951, -1.1261107, -1.4181522],
    #                     [1.008275, -0.84791195, 0.3006532],
    #                     [0.31099692, -0.32650718, -0.60247767]])
    #
    # si_data = np.array([[-0.28463233, -2.5517333, 1.4781238],
    #                     [-0.10505871, -0.31454122, -0.29844758],
    #                     [-1.3067418, 0.48718405, 0.6779812],
    #                     [2.024449, -1.3925922, -1.6178994],
    #                     [-0.08328865, -0.396574, 1.0888542],
    #                     [0.13096762, -0.14382902, 0.2695235],
    #                     [0.5405067, -0.67946523, -0.8433032]])
    #
    # muX = torch.from_numpy(mu_data)
    # siX = torch.from_numpy(si_data)
    # print(gty.shape)
    # print(muX.shape)
    # gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2, 2])
    # muX, siX = torch.randn(8, 512), torch.randn(8, 512)
    # print(gty.shape)
    # print(muX.shape)
    # diff = mls(muX, siX, gty)
    # print(diff)
    #
    # KL = KLDiracVMF(512, 64)
    # # gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2, 2])
    # muX, siX, gty = torch.randn(8, 512), torch.randn(8, 1), torch.randn(8, 512)
    # print(gty.shape)
    # print(muX.shape)
    # losses, l1, l2, l3 = KL(muX, siX, gty)
    # print(losses.mean())

    feature = torch.randn(8, 512)
