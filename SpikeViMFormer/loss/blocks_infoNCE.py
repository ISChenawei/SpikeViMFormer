import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable


def get_heartmap_pool(part_features, blocks=3, add_global=False, otherbranch=False, fine_grained_attention=False):
    # 计算热力图，作为特征聚合的基础
    heatmap = torch.mean(part_features, dim=-1)
    size = part_features.size(1)
    arg = torch.argsort(heatmap, dim=1, descending=True)
    x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    x_sort = torch.stack(x_sort, dim=0)

    # 根据地物自动聚类的类别数来将区域进行分类
    split_each = size // blocks
    split_list = [int(split_each) for i in range(blocks - 1)]
    split_list.append(size - sum(split_list))  # 确保区域大小平衡
    split_x = x_sort.split(split_list, dim=1)

    # 计算每块区域的均值
    split_list = [torch.mean(split, dim=1) for split in split_x]
    part_features_ = torch.stack(split_list, dim=2)

    if fine_grained_attention:
        # 使用自注意力机制来增强局部细节
        part_features_ = apply_attention(part_features_)

    if add_global:
        # 全局特征添加
        global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, blocks)
        part_features_ = part_features_ + global_feat

    if otherbranch:
        # 处理其他分支
        otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
        return part_features_, otherbranch_

    return part_features_


def apply_attention(features):
    """
    Apply attention mechanism to enhance fine-grained features.
    """
    # 使用自注意力机制来加强局部细节
    attn = nn.MultiheadAttention(features.size(-1), num_heads=8)  # 假设8个头
    features = features.transpose(0, 1)  # 转换为 (seq_len, batch_size, feature_dim)
    attn_output, _ = attn(features, features, features)
    return attn_output.transpose(0, 1)  # 转回 (batch_size, seq_len, feature_dim)


class blocks_InfoNCE(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', temperature=0.07):
        super().__init__()
        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device
        self.temperature = temperature  # 引入温度缩放来控制相似度强度

    def forward(self, image_features1, image_features2, logit_scale, blocks=3, fine_grained_attention=False):
        image_features1_flatten = image_features1.view(image_features1.size(0), image_features1.size(1), -1).transpose(-2, -1)
        image_features2_flatten = image_features2.view(image_features2.size(0), image_features2.size(1), -1).transpose(-2, -1)

        heat_result_1 = get_heartmap_pool(image_features1_flatten, blocks, fine_grained_attention=fine_grained_attention)
        heat_result_2 = get_heartmap_pool(image_features2_flatten, blocks, fine_grained_attention=fine_grained_attention)

        # Concatenate and normalize features for comparison
        image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]), dim=-1)
        image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]), dim=-1)

        image_features1 = F.normalize(image_features_blocks_1, dim=-1)
        image_features2 = F.normalize(image_features_blocks_2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T / self.temperature
        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        # Compute loss using cross-entropy
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss
