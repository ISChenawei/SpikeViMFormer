import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
#
# class InfoNCE(nn.Module):
#     def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', temperature=0.07, block_size=3, fine_grained_attention=False):
#         super().__init__()
#
#         self.loss_function = loss_function  # -- default CrossEntropy
#         self.device = device
#         self.temperature = temperature  # 自适应温度缩放
#         self.block_size = block_size  # 对应的分块数，可以增强细粒度特征
#         self.fine_grained_attention = fine_grained_attention  # 是否启用细粒度注意力机制
#
#     def forward(self, image_features1, image_features2, logit_scale):
#         # 归一化图像特征
#         image_features1 = F.normalize(image_features1, dim=-1)
#         image_features2 = F.normalize(image_features2, dim=-1)
#
#         # 计算对比损失时的 logits
#         logits_per_image1 = logit_scale * image_features1 @ image_features2.T
#         logits_per_image2 = logits_per_image1.T
#
#         # 标签：匹配样本的索引
#         labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
#
#         # 基本的 InfoNCE 损失计算
#         loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2
#
#         # 如果启用细粒度注意力机制，可以加入细粒度信息增强
#         if self.fine_grained_attention:
#             # 计算图像特征的局部对比损失
#             loss += self.calculate_local_contrastive_loss(image_features1, image_features2, labels)
#
#         # 返回损失
#         return loss
#
#     def calculate_local_contrastive_loss(self, image_features1, image_features2, labels):
#         """
#         基于局部区域的对比损失，可以增强细粒度的关注。
#         直接对每个分块进行对比损失计算
#         """
#         # 将图像分成多个块，进行局部对比损失计算
#         part_features1 = self.split_into_blocks(image_features1)
#         part_features2 = self.split_into_blocks(image_features2)
#
#         # 计算局部区域之间的对比损失
#         local_logits = part_features1 @ part_features2.T / self.temperature  # 计算局部区域的相似度
#         local_loss = self.loss_function(local_logits, labels)
#         return local_loss
#
#     def split_into_blocks(self, features):
#         """
#         将图像特征分成多个小块，返回每个块的特征。
#         通过这种方式可以增强细粒度的学习。
#         """
#         batch_size, channels, height, width = features.shape
#         # 将图像特征按块分割
#         block_height = height // self.block_size
#         block_width = width // self.block_size
#
#         blocks = []
#         for i in range(self.block_size):
#             for j in range(self.block_size):
#                 # 提取每个小块的特征
#                 block = features[:, :, i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
#                 blocks.append(block)
#
#         # 合并所有的小块特征，形成一个新的张量
#         blocks = torch.cat([block.flatten(2) for block in blocks], dim=2)  # [batch_size, channels, num_blocks]
#         return blocks

