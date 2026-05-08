import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def cal_kl_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                              F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss / len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                         F.softmax(Variable(outputs2), dim=1))
    return loss


def cal_triplet_loss(outputs, outputs2, labels, loss_func, split_num=8):
    if isinstance(outputs, list):
        loss = 0
        for i in range(len(outputs)):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss += loss_func(out_concat, labels_concat)
        loss = loss / len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels, labels), dim=0)
        loss = loss_func(out_concat, labels_concat)
    return loss

class CPMLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CPMLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=0.2)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2)
    def forward(self, inputs, targets):
        n = inputs.size(0)
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(targets, 4, 0)

        lb_num = len(lb1.unique())
        lbs = lb1.unique()

        n = lbs.size(0)

        ft1 = ft1.chunk(lb_num, 0)
        ft2 = ft2.chunk(lb_num, 0)
        ft3 = ft3.chunk(lb_num, 0)
        ft4 = ft4.chunk(lb_num, 0)

        center1 = []
        center2 = []
        center3 = []
        center4 = []
        for i in range(lb_num):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)

        ft5 = torch.cat(center1)
        ft6 = torch.cat(center2)
        ft7 = torch.cat(center3)
        ft8 = torch.cat(center4)

        dist_13 = pdist_torch(ft1, ft3)
        dist_23 = pdist_torch(ft2, ft3)
        dist_33 = pdist_torch(ft3, ft3)
        dist_11 = pdist_torch(ft1, ft1)

        dist_14 = pdist_torch(ft1, ft4)
        dist_24 = pdist_torch(ft2, ft4)
        dist_44 = pdist_torch(ft4, ft4)
        dist_22 = pdist_torch(ft2, ft2)
        dist_13c = 1 - F.cosine_similarity(ft1, ft3)
        dist_23c = 1 - F.cosine_similarity(ft2, ft3)
        dist_33c = 1 - F.cosine_similarity(ft3, ft3)
        dist_11c = 1 - F.cosine_similarity(ft1, ft1)

        dist_44c = 1 - F.cosine_similarity(ft1, ft4)
        dist_22c = 1 - F.cosine_similarity(ft2, ft4)
        dist_14c = 1 - F.cosine_similarity(ft4, ft4)
        dist_24c = 1 - F.cosine_similarity(ft2, ft2)
        mask1 = lbs.unsqueeze(1) == lbs.unsqueeze(0)
        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())

        dist_ap_123, dist_an_123, dist_ap_124, dist_an_124, dist_an_33, dist_an_44, dist_an_11, dist_an_22 = [], [], [], [], [], [], [], []
        for i in range(n):
            dist_ap_123.append(dist_23[i][mask[i]].max().unsqueeze(0))
            dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            dist_an_33.append(dist_33[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_11.append(dist_11[i][mask[i] == 0].min().unsqueeze(0))

            dist_ap_124.append(dist_14[i][mask[i]].max().unsqueeze(0))
            dist_an_124.append(dist_24[i][mask[i]].min().unsqueeze(0))
            dist_an_44.append(dist_44[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_22.append(dist_22[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap_123c, dist_an_123c, dist_ap_124c, dist_an_124c, dist_an_33c, dist_an_44c, dist_an_11c, dist_an_22c = [], [], [], [], [], [], [], []
        for i in range(n):
            # 获取 mask[i] 为 True 的索引
            true_indices = torch.where(mask1[i])[0]
            false_indices = torch.where(~mask1[i])[0]

            # 应用 mask 索引并进行相应的最大值和最小值操作
            if true_indices.nelement() > 0:
                dist_ap_123c.append(dist_23c[true_indices].max().unsqueeze(0))
                dist_ap_124c.append(dist_24c[true_indices].max().unsqueeze(0))

            if false_indices.nelement() > 0:
                dist_an_123c.append(dist_13c[false_indices].min().unsqueeze(0))
                dist_an_124c.append(dist_14c[false_indices].min().unsqueeze(0))
                dist_an_33c.append(dist_33c[false_indices].min().unsqueeze(0))
                dist_an_44c.append(dist_44c[false_indices].min().unsqueeze(0))
                dist_an_11c.append(dist_11c[false_indices].min().unsqueeze(0))
                dist_an_22c.append(dist_22c[false_indices].min().unsqueeze(0))
        dist_ap_123c = torch.cat(dist_ap_123c)
        dist_an_123c = torch.cat(dist_an_123c).detach()
        dist_an_33c = torch.cat(dist_an_33c)
        dist_an_11c = torch.cat(dist_an_11c)

        dist_an_44c = torch.cat(dist_an_44c)
        dist_an_22c = torch.cat(dist_an_22c)
        dist_ap_124c = torch.cat(dist_ap_124c)
        dist_an_124c = torch.cat(dist_an_124c).detach()

        dist_ap_123 = torch.cat(dist_ap_123)
        dist_an_123 = torch.cat(dist_an_123).detach()
        dist_an_33 = torch.cat(dist_an_33)
        dist_an_11 = torch.cat(dist_an_11)


        dist_ap_124 = torch.cat(dist_ap_124)
        dist_an_124 = torch.cat(dist_an_124).detach()
        dist_an_44 = torch.cat(dist_an_44)
        dist_an_22 = torch.cat(dist_an_22)
        loss_123c = self.ranking_loss(dist_an_123c, dist_ap_123c, torch.ones_like(dist_an_123c)) + (
                    self.ranking_loss(dist_an_33c, dist_ap_123c, torch.ones_like(dist_an_33c)) + self.ranking_loss(
                dist_an_11c, dist_ap_123c, torch.ones_like(dist_an_33c))) * 0.5
        loss_124c = self.ranking_loss(dist_an_124c, dist_ap_124c, torch.ones_like(dist_an_124c)) + (
                    self.ranking_loss(dist_an_44c, dist_ap_124c, torch.ones_like(dist_an_44c)) + self.ranking_loss(
                dist_an_22c, dist_ap_124c, torch.ones_like(dist_an_44c))) * 0.5
        loss_123 = self.ranking_loss(dist_an_123, dist_ap_123, torch.ones_like(dist_an_123)) + (self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_an_33)) + self.ranking_loss(dist_an_11, dist_ap_123, torch.ones_like(dist_an_33))) * 0.5
        loss_124 = self.ranking_loss(dist_an_124, dist_ap_124, torch.ones_like(dist_an_124)) + (self.ranking_loss(dist_an_44, dist_ap_124, torch.ones_like(dist_an_44)) + self.ranking_loss(dist_an_22, dist_ap_124, torch.ones_like(dist_an_44))) * 0.5
        return (loss_123 + loss_124+loss_123c + loss_124c)/2

# def pdist_torch(emb1, emb2):
#     '''
#     compute the eucilidean distance matrix between embeddings1 and embeddings2
#     using gpu
#     '''
#     if emb1.dtype != emb2.dtype:
#         emb2 = emb2.to(emb1.dtype)
#     # emb1 = emb1.half()
#     # emb2 = emb2.half()
#     m, n = emb1.shape[0], emb2.shape[0]
#     emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
#     emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
#     dist_mtx = emb1_pow + emb2_pow
#     dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
#     # dist_mtx = dist_mtx.clamp(min = 1e-12)
#     dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
#     return dist_mtx
def pdist_torch(emb1, emb2):
    '''
    compute the euclidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    if emb1.dtype != emb2.dtype:
        emb2 = emb2.to(emb1.dtype)
    device = emb1.device

    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    dist_mtx = emb1_pow + emb2_pow
    # 强制dist_mtx类型与emb1保持一致
    dist_mtx = dist_mtx.to(emb1.dtype)

    # 使用非 inplace 版本addmm，防止类型冲突
    dist_mtx = torch.addmm(dist_mtx, emb1, emb2.t(), beta=1, alpha=-2)

    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()

    return dist_mtx