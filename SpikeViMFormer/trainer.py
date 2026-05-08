import time

import numpy as np
import torch
from thop import profile
from tqdm import tqdm

from .loss.triplet_loss import Tripletloss
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
# from sample4geo.loss.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
from SpikeViMFormer.loss.cal_loss import CPMLoss, cal_loss,cal_triplet_loss
import torch.nn as nn
from spikingjelly.activation_based import  functional
from torch.cuda.amp import autocast,grad_scaler

def train(train_config, model,model_teacher,dataloader, loss_functions, optimizer, epoch, train_steps_per,tensorboard=None,
          scheduler=None, scaler=None):

    # set model train mode
    model.train()
    model_teacher.eval()
    losses = AverageMeter()


    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    triplet_loss = Tripletloss(margin=0.3)
    criterion = nn.CrossEntropyLoss()
    criterion_cpm = CPMLoss(margin=0.3)
    # for loop over one epoch
    # 初始化一次全局队列
    global_queue = FeatureQueue(dim=360, size=4096)
    global_query_queue = FeatureQueue(dim=360, size=4096)

    for query, reference, ids, labels in bar:
        query = query.to(train_config.device).float()
        reference = reference.to(train_config.device).float()
        labels = labels.to(train_config.device)
        # scaler = grad_scaler
        # with torch.no_grad():
            # output1_tea, output2_tea = model_teacher(query, reference)
        if scaler:
            with autocast():  # -- 使用混合精度
                # data (batches) to device   

                # functional.reset_net(model.module)
                # Forward pass
                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    output1, output2 = model(query, reference)

                    features1, features2 = output1[-2], output2[-2]  # -- for contrastive
                    # features1_tea, features2_tea = output1_tea[-2], output2_tea[-2]
                    # features_tri_1, features_tri_2 = output1[2], output2[2]  # -- for triplet

                    features_cls_1, features_cls_2 = output1[1], output2[1]  # -- for classifier
                    features_fine_1, features_fine_2 = output1[-1], output2[-1]  # -- for fine-grained
                    features_dsa_1, features_dsa_2 = output1[0], output2[0]  # -- for DSA loss

                with torch.no_grad():
                    refined1, refined2 = rerank(features1, features2)
                    # 在每个 batch 中：
                    # global_queue.enqueue(features1)
                    global_queue.enqueue(features2)
                    global_query_queue.enqueue(features1)
                    # 从队列中取 gallery 特征
                    global_gallery_feats = global_queue.get(num_samples=2048)
                    global_query_feats = global_query_queue.get(num_samples=2048)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                    # 将 teacher 特征做 L2 normalize


                    # loss1 = rerank_alignment_loss(features1, features2, features1_tea[:,:360], features2_tea[:,:360], loss_type='cosine',
                    #                                     alpha=0.7, lambda_weight=3)
                    # loss1 = rerank_alignment_loss(features1, features2, features1_tea, features2_tea, loss_type='cosine',
                    #                                     alpha=0.7, lambda_weight=3)
                    # loss2 = rerank_alignment_loss(features1, features2, refined1, refined2, loss_type='cosine',
                    #                                     alpha=0.7, lambda_weight=1)
                    loss2 = rerank_alignment_loss_with_queue(global_query_feats, global_gallery_feats, rerank, loss_type='cosine',
                                                     alpha=0.7, lambda_weight=2)
                    # loss = loss+loss1+loss2
                    # loss_cls = criterion_cpm(features_tri_1, labels) + criterion_cpm(features_tri_2, labels)
                    # loss_cls = cal_triplet_loss(features_cls_1,features_cls_2, labels, criterion,triplet_loss)
                    loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)
                    loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                                                          model.module.logit_scale_blocks.exp())
                    loss_rerank = rerank_alignment_loss(features1, features2, refined1, refined2, loss_type='cosine',
                                                        alpha=0.7, lambda_weight=2) + loss2

                else:
                    # 1. infoNCE
                    loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())

                    # 2. Classification
                    # loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)
                    #
                    #
                    # # 3. Domian Space Alignment Loss
                    # loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                    #                                       model.logit_scale_blocks.exp())


                lossall = train_config.weight_infonce * loss  + train_config.weight_cls * loss_cls + train_config.weight_dsa * loss_DSA  + loss_rerank

                # lossall = 1 * loss + train_config.weight_dsa * loss_DSA + loss_rerank

                losses.update(lossall.item())

            # scaler.scale(loss).backward()  # -- 混合精度好像是这样用的
            scaler.scale(lossall).backward()  # -- 这里才是反向传播，上面就是记录一下

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            # tst = model.logit_scale
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_reran": "{:.4f}".format(loss_rerank.item()),
                "loss_cls": "{:.4f}".format(train_config.weight_cls * loss_cls.item()),
                "loss_dsa": "{:.4f}".format(train_config.weight_dsa * loss_DSA.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

            if tensorboard is not None:
                steps = step + (epoch - 1) * train_steps_per
                tensorboard.add_scalar("Loss", lossall.item(), steps)
                tensorboard.add_scalar("Loss_Avg", losses.avg, steps)
                tensorboard.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], steps)
                tensorboard.add_scalar("Learning_Rate_Temp", optimizer.param_groups[-1]['lr'], steps)
                tensorboard.add_scalar("Temperature", model.module.logit_scale.detach().cpu().numpy(), steps)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

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
    with autocast(enabled=False):
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

def rerank_alignment_loss(features1, features2, refined1, refined2, loss_type='cosine', alpha=0.7, lambda_weight=1.0):
    """
    features1: Tensor [B, D], query
    features2: Tensor [B, D], gallery
    rerank_fn: function to rerank and return refined (query, gallery)
    loss_type: 'cosine' or 'mse'
    alpha: blending parameter used in rerank
    lambda_weight: scaling weight
    """
    # with torch.no_grad():
    # refined1, refined2 = rerank_fn(features1, features2, alpha=alpha)

    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    refined1 = F.normalize(refined1, dim=1)
    refined2 = F.normalize(refined2, dim=1)

    if loss_type == 'cosine':
        loss1 = 1 - F.cosine_similarity(features1, refined1, dim=1).mean()
        loss2 = 1 - F.cosine_similarity(features2, refined2, dim=1).mean()
        log_p = F.log_softmax(features1 @ refined1.t(), dim=1)
        q = F.softmax(refined1 @ refined1.t(),dim=1)
        kl_loss1 = F.kl_div(log_p, q, reduction='batchmean')
        log_p = F.log_softmax(features2 @ refined2.t(), dim=1)
        q = F.softmax(refined2 @ refined2.t(),dim=1)
        kl_loss2 = F.kl_div(log_p, q, reduction='batchmean')

    elif loss_type == 'mse':
        loss1 = F.mse_loss(features1, refined1)
        loss2 = F.mse_loss(features2, refined2)
        loss3 = 1 - F.cosine_similarity(features1, refined2, dim=1).mean()
        loss4 = 1 - F.cosine_similarity(features2, refined1, dim=1).mean()
    else:
        raise ValueError("Unsupported loss type")

    return lambda_weight * (loss1 + loss2 + kl_loss1 + kl_loss2) / 2

def rerank_alignment_loss_with_queue(global_query_feats, global_gallery_feats, rerank_fn, loss_type='cosine', alpha=0.7, lambda_weight=1.0):
    global_query_feats = F.normalize(global_query_feats, dim=1)
    global_gallery_feats = F.normalize(global_gallery_feats, dim=1)

    refined_query, refined_gallery = rerank_fn(global_query_feats, global_gallery_feats, alpha=alpha)

    refined_query = F.normalize(refined_query, dim=1)
    refined_gallery = F.normalize(refined_gallery, dim=1)

    if loss_type == 'cosine':
        loss1 = 1 - F.cosine_similarity(global_query_feats, refined_query, dim=1).mean()
        log_p = F.log_softmax(global_query_feats @ refined_query.t(), dim=1)
        q = F.softmax(refined_query @ refined_query.t(), dim=1)
        kl_loss1 = F.kl_div(log_p, q, reduction='batchmean')
        loss2= 1 - F.cosine_similarity(global_gallery_feats, refined_gallery, dim=1).mean()
        log_p = F.log_softmax(global_gallery_feats @ refined_gallery.t(), dim=1)
        q = F.softmax(refined_gallery @ refined_gallery.t(), dim=1)
        kl_loss2 = F.kl_div(log_p, q, reduction='batchmean')
    elif loss_type == 'mse':
        # loss1 = F.mse_loss(query_feats, refined_query)
        kl_loss = torch.tensor(0.0).cuda()

    else:
        raise ValueError("Unsupported loss type")

    return lambda_weight * (loss1 + kl_loss1 + loss2 +kl_loss2)/2

class FeatureQueue:
    def __init__(self, dim=512, size=4096):
        self.size = size
        self.dim = dim
        self.ptr = 0
        self.queue = torch.zeros(size, dim).cuda()
        self.initialized = False

    @torch.no_grad()
    def enqueue(self, feats):
        B = feats.size(0)
        if B > self.size:
            feats = feats[-self.size:]
            B = feats.size(0)
        if self.ptr + B > self.size:
            first = self.size - self.ptr
            second = B - first
            self.queue[self.ptr:] = feats[:first]
            self.queue[:second] = feats[first:]
            self.ptr = second
        else:
            self.queue[self.ptr:self.ptr+B] = feats
            self.ptr = (self.ptr + B) % self.size
        self.initialized = True

    def get(self, num_samples=None):
        if not self.initialized:
            raise RuntimeError("Queue not initialized with any features.")
        if num_samples is None or num_samples > self.size:
            return self.queue.clone().detach()
        else:
            indices = torch.randperm(self.size)[:num_samples]
            return self.queue[indices].clone().detach()

def predict(train_config, model, dataloader):
    model.eval()

    draw_vis = False
    if draw_vis:
        import cv2
        import os
        import numpy as np
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 2)
        for i in range(iterations):
            uav_ori = cv2.imread(rf"{pic_path}/{i}_uav.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.jpg")

            uav_shape = uav_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            uav = cv2.resize(uav_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            uav = torch.tensor(uav).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            uav = normalize(uav)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    uav = uav.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_uav = model(uav)[1]
                    img_feature_uav = F.normalize(img_feature_uav, dim=1)

                    img_feature_sat = model(sat)[1]
                    img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_uav = img_feature_uav[0].permute(1, 2, 0)
                    heat_map_uav = torch.mean(heat_map_uav, dim=2).detach().cpu().numpy()
                    heat_map_uav = (heat_map_uav - heat_map_uav.min()) / (heat_map_uav.max() - heat_map_uav.min())
                    heat_map_uav = cv2.resize(heat_map_uav, [uav_shape[1], uav_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_uav = cv2.applyColorMap((heat_map_uav * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_uav = cv2.addWeighted(uav_ori, alpha, colored_image_uav, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)


                    out_path = r"/media/xiapanwang/主数据盘/xiapanwang/Codes/python/Cross-View_Geo-Localization/DAC/DSA_off"
                    cv2.imwrite(rf"{out_path}/{i}_uav_vis.jpg", blended_image_uav)
                    cv2.imwrite(rf"{out_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    img_feature = model(img)[-2]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list
