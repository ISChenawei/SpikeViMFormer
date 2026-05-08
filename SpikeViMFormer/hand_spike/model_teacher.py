import torch.nn as nn
from torch import autocast
import torch.nn.functional as F
# from .ConvNext import make_spike_model
from .Spikeformer import make_convnext_model
import torch
import numpy as np

class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, resnet=False):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

        # 1. temperature factor for contrastive learning
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.02))
        # self.logit_scale = torch.scalar_tensor(3.569)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 2. weight for blocks_infoNCE
        self.w_blocks1 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks2 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks3 = torch.nn.Parameter(torch.ones([]))


    def get_config(self):
        input_size = (3, 224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        config = {
            'input_size': input_size,
            'mean': mean,
            'std': std
        }
        return config

    def forward(self, x1, x2=None):
        # if x1 is None:
        #     y1 = None
        # else:
        #     y1 = self.model_1(x1)
        #     # print("pause")
        #
        # if x2 is None:
        #     y2 = None
        # else:
        #     y2 = self.model_1(x2)
        # return y1, y2

        if x2 is not None:
            y1 = self.model_1(x1)
            y2 = self.model_1(x2)
            return y1, y2
        else:
            y1 = self.model_1(x1)
            return y1


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight=False, block=4, return_f=False, resnet=False):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        self.model_1 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

        if self.share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 =make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

    def forward(self, x1, x2, x3, x4=None):  # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)

        if x3 is None:
            y3 = None
        else:
            y3 = self.model_1(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = self.model_2(x4)
        return y1, y2, y3, y4

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def rerank(q_feat, g_feat, k1=20, k2=6, alpha=0.7, eval_type=True):
    with autocast(device_type='cuda',enabled=False):
        device = q_feat.device
        feats = torch.cat([q_feat, g_feat], dim=0)  # [N, D]
        N, D = feats.size()
        query_num = q_feat.size(0)

        with torch.no_grad():
            # Step 1: 计算 pairwise 距离（GPU）
            dist = pairwise_distance(feats, feats)  # [N, N]
            original_dist = dist.clone().detach()
            original_dist = original_dist / torch.max(original_dist, dim=0, keepdim=True)[0]
            original_dist = original_dist.t().cpu().numpy()  # [N, N] -> transpose

            if eval_type:
                dist[:, query_num:] = dist.max()

            # Step 2: 用 torch.topk 替代 argsort，仅保留前 k1+1 个
            initial_rank = torch.topk(dist, k=k1, dim=1, largest=False).indices.cpu().numpy().astype(
                np.int32)  # [N, k1+1]

            V = np.zeros((N, N), dtype=np.float32)

            for i in range(N):
                k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
                k_reciprocal_expansion_index = k_reciprocal_index

                for candidate in k_reciprocal_index:
                    candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, k1 // 2)
                    if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (2 / 3) * len(
                            candidate_k_reciprocal_index):
                        k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

                k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
                weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
                V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-6)

            if k2 != 1:
                V_qe = np.zeros_like(V, dtype=np.float32)
                for i in range(N):
                    V_qe[i, :] = np.mean(V[initial_rank[i]], axis=0)
                V = V_qe

        # Step 3: 重新加权后的特征计算
        feats_np = feats.detach().cpu().numpy()
        refined_feats = np.matmul(V, feats_np)
        refined_feats = alpha * feats_np + (1 - alpha) * refined_feats
        refined_feats = torch.from_numpy(refined_feats).to(device)
        refined_feats = F.normalize(refined_feats, dim=1)

        refined_query = refined_feats[:query_num]
        refined_gallery = refined_feats[query_num:]
        return refined_query,refined_gallery
def make_model_teacher(opt):
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, return_f=opt.triplet_loss, resnet=opt.resnet)
    # elif opt.views == 3:
    #     model = three_view_net(opt.nclasses, share_weight=opt.share, block=opt.block, return_f=opt.triplet_loss,
    #                            resnet=opt.resnet)
    return model
