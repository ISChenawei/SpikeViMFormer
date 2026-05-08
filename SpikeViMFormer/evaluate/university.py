import scipy
import torch
import numpy as np
from tqdm import tqdm
import gc
from ..trainer import predict
import torch.nn.functional as F
import time
# def evaluate(config,
#              model,
#              query_loader,
#              gallery_loader,
#              ranks=[1, 5, 10],
#              step_size=1000,
#              cleanup=True):
#     print("Extract Features:")
#     img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
#     img_features_query, ids_query = predict(config, model, query_loader)
#
#
#     gl = ids_gallery.cpu().numpy()
#     ql = ids_query.cpu().numpy()
#
#     print("Compute Scores:")
#     CMC = torch.IntTensor(len(ids_gallery)).zero_()
#     ap = 0.0
#     for i in tqdm(range(len(ids_query))):
#
#         ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
#         if CMC_tmp[0] == -1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#
#     AP = ap / len(ids_query) * 100
#
#     CMC = CMC.float()
#     CMC = CMC / len(ids_query)  # average CMC
#
#     # top 1%
#     top1 = round(len(ids_gallery) * 0.01)
#
#     string = []
#
#     for i in ranks:
#         string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))
#
#     string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
#     string.append('AP: {:.4f}'.format(AP))
#
#     print(' - '.join(string))
#
#     # cleanup and free memory on GPU
#     if cleanup:
#         del img_features_query, ids_query, img_features_gallery, ids_gallery
#         gc.collect()
#         # torch.cuda.empty_cache()
#
#     return CMC[0]
#
#
# def eval_query(qf, ql, gf, gl):
#
#     score = gf @ qf.unsqueeze(-1)
#     # score = rerank(gf,qf)
#     score = score.squeeze().cpu().numpy()
#
#     # predict index
#     index = np.argsort(score)  # from small to large
#     index = index[::-1]
#
#     # good index
#     query_index = np.argwhere(gl == ql)
#     good_index = query_index
#
#     # junk index
#     junk_index = np.argwhere(gl == -1)
#
#     CMC_tmp = compute_mAP(index, good_index, junk_index)
#     return CMC_tmp
#
#
# def compute_mAP(index, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size == 0:  # if empty
#         cmc[0] = -1
#         return ap, cmc
#
#     # remove junk_index
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]
#
#     # find good_index index
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask == True)
#     rows_good = rows_good.flatten()
#
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0 / ngood
#         precision = (i + 1) * 1.0 / (rows_good[i] + 1)
#         if rows_good[i] != 0:
#             old_precision = i * 1.0 / rows_good[i]
#         else:
#             old_precision = 1.0
#         ap = ap + d_recall * (old_precision + precision) / 2
#
#     return ap, cmc

# def evaluate(config,
#              model,
#              query_loader,
#              gallery_loader,
#              ranks=[1, 5, 10],
#              step_size=1000,
#              cleanup=True,
#              use_rerank=True):
#
#     print("Extract Features:")
#     img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
#     img_features_query, ids_query = predict(config, model, query_loader)
#
#     gl = ids_gallery.cpu().numpy()
#     ql = ids_query.cpu().numpy()
#
#     print("Compute Scores:")
#     CMC = torch.IntTensor(len(ids_gallery)).zero_()
#     ap = 0.0
#
#     if use_rerank:
#         print("Computing re-ranking...")
#         qf = F.normalize(img_features_query, p=2, dim=1).cpu().numpy()
#         gf = F.normalize(img_features_gallery, p=2, dim=1).cpu().numpy()
#         distmat = rerank(qf, gf)  # shape: [num_query, num_gallery]
#
#         for i in tqdm(range(len(ql))):
#             dist = distmat[i]
#             ap_tmp, CMC_tmp = eval_query_rerank(dist, ql[i], gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp
#             ap += ap_tmp
#     else:
#         for i in tqdm(range(len(ql))):
#             ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp
#             ap += ap_tmp
#
#     AP = ap / len(ids_query) * 100
#     CMC = CMC.float() / len(ids_query)
#
#     top1 = round(len(ids_gallery) * 0.01)
#     string = []
#
#     for i in ranks:
#         string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))
#
#     string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
#     string.append('AP: {:.4f}'.format(AP))
#
#     print(' - '.join(string))
#
#     if cleanup:
#         del img_features_query, ids_query, img_features_gallery, ids_gallery
#         gc.collect()
#
#     return CMC[0]
#
#
# def eval_query_rerank(dist, ql, gl):
#     index = np.argsort(dist)  # distance from small to large
#     good_index = np.argwhere(gl == ql)
#     junk_index = np.argwhere(gl == -1)
#
#     ap, cmc = compute_mAP(index, good_index, junk_index)
#     return ap, cmc
#
# def eval_query(qf, ql, gf, gl):
#     score = gf @ qf.unsqueeze(-1)
#     score = score.squeeze().cpu().numpy()
#
#     index = np.argsort(score)[::-1]  # from high to low
#     good_index = np.argwhere(gl == ql)
#     junk_index = np.argwhere(gl == -1)
#
#     ap, cmc = compute_mAP(index, good_index, junk_index)
#     return ap, cmc
#
#
# def compute_mAP(index, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size == 0:
#         cmc[0] = -1
#         return ap, cmc
#
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]
#
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask == True).flatten()
#
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0 / ngood
#         precision = (i + 1) * 1.0 / (rows_good[i] + 1)
#         old_precision = i * 1.0 / rows_good[i] if rows_good[i] != 0 else 1.0
#         ap += d_recall * (old_precision + precision) / 2
#
#     return ap, cmc
#
# def k_reciprocal_neigh(initial_rank, i, k1):
#     forward_k_neigh_index = initial_rank[i, :k1 + 1]
#     backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#     fi = np.where(backward_k_neigh_index == i)[0]
#     return forward_k_neigh_index[fi]
#
# def pairwise_distance(x, y):
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
# def rerank(q_feat, g_feat, k1=20, k2=6, alpha=0.7, eval_type=True):
#     """
#     Fast Feature-level re-ranking optimization: directly output distance matrix.
#
#     Args:
#         q_feat: [num_query, D]
#         g_feat: [num_gallery, D]
#
#     Returns:
#         final_dist: [num_query, num_gallery]
#     """
#     device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
#     feats = torch.cat([q_feat, g_feat], dim=0)  # [N, D]
#     N, D = feats.size()
#     query_num = q_feat.size(0)
#
#     # Step 1: Compute pairwise distance
#     with torch.no_grad():
#         dist = pairwise_distance(feats, feats)  # [N, N]
#         original_dist = dist.detach().cpu().numpy()
#         original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
#
#         if eval_type:
#             dist[:, query_num:] = dist.max()
#         dist = dist.cpu().numpy()
#
#         initial_rank = np.argsort(dist).astype(np.int32)
#
#         # Step 2: Build neighbor graph (V matrix)
#         V = np.zeros((N, N), dtype=np.float32)
#
#         for i in range(N):
#             k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
#             k_reciprocal_expansion_index = k_reciprocal_index
#
#             for candidate in k_reciprocal_index:
#                 candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, k1//2)
#                 if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (2/3)*len(candidate_k_reciprocal_index):
#                     k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#             k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#             weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#             V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-6)
#
#         if k2 != 1:
#             V_qe = np.zeros_like(V, dtype=np.float32)
#             for i in range(N):
#                 V_qe[i, :] = np.mean(V[initial_rank[i, :k2]], axis=0)
#             V = V_qe
#
#     # Step 3: Feature updating (矩阵乘法批处理，快速版)
#     feats_np = feats.detach().cpu().numpy()  # [N, D]
#     refined_feats = np.matmul(V, feats_np)  # [N, D]
#     refined_feats = alpha * feats_np + (1 - alpha) * refined_feats
#     refined_feats = torch.from_numpy(refined_feats).to(device)
#     refined_feats = torch.nn.functional.normalize(refined_feats, dim=1)
#
#     # Step 4: Compute final distance
#     refined_query = refined_feats[:query_num]
#     refined_gallery = refined_feats[query_num:]
#
#     final_dist = pairwise_distance(refined_query, refined_gallery)
#
#     return final_dist
    # return refined_query,refined_gallery



import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc

# def evaluate(config,
#              model,
#              query_loader,
#              gallery_loader,
#              ranks=[1, 5, 10],
#              step_size=1000,
#              cleanup=True,
#              use_rerank=False):
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print("Extract Features:")
#     img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
#     img_features_query, ids_query = predict(config, model, query_loader)
#
#     img_features_gallery = img_features_gallery.to(device)
#     img_features_query = img_features_query.to(device)
#     ids_gallery = ids_gallery.to(device)
#     ids_query = ids_query.to(device)
#
#     gl = ids_gallery.cpu().numpy()
#     ql = ids_query.cpu().numpy()
#
#     print("Compute Scores:")
#     CMC = torch.IntTensor(len(ids_gallery)).zero_().to(device)
#     ap = 0.0
#
#     if use_rerank:
#         # print("Computing TRM_Module...")
#         qf = F.normalize(img_features_query, p=2, dim=1)
#         gf = F.normalize(img_features_gallery, p=2, dim=1)
#         # distmat = pairwise_distance(qf, gf)
#         distmat = rerank(qf, gf)  # [num_query, num_gallery]
#
#         for i in tqdm(range(len(ql))):
#             dist = distmat[i].detach().cpu().numpy()
#             ap_tmp, CMC_tmp = eval_query_rerank(dist, ql[i], gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp.to(device)
#             ap += ap_tmp
#     else:
#         for i in tqdm(range(len(ql))):
#             ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp.to(device)
#             ap += ap_tmp
#
#     AP = ap / len(ids_query) * 100
#     CMC = CMC.float() / len(ids_query)
#
#     top1 = round(len(ids_gallery) * 0.01)
#     string = []
#
#     for i in ranks:
#         string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1].item() * 100))
#
#     string.append('Recall@top1: {:.4f}'.format(CMC[top1].item() * 100))
#     string.append('AP: {:.4f}'.format(AP))
#
#     print(' - '.join(string))
#
#     if cleanup:
#         del img_features_query, ids_query, img_features_gallery, ids_gallery
#         gc.collect()
#         torch.cuda.empty_cache()
#
#     return CMC[0].item()
# def evaluate(config,
#              model,
#              query_loader,
#              gallery_loader,
#              ranks=[1, 5, 10],
#              step_size=1000,
#              cleanup=True,
#              use_rerank=False):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print("Extract Features:")
#
#     # 记录画廊特征提取的开始时间
#     start_time = time.time()
#     img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
#     end_time = time.time()
#     gallery_inference_time = end_time - start_time
#     gallery_fps = len(ids_gallery) / gallery_inference_time
#     print(f"Gallery feature extraction time: {gallery_inference_time:.4f} seconds")
#     print(f"Gallery FPS: {gallery_fps:.2f}")
#
#     # 记录查询特征提取的开始时间
#     start_time = time.time()
#     img_features_query, ids_query = predict(config, model, query_loader)
#     end_time = time.time()
#     query_inference_time = end_time - start_time
#     query_fps = len(ids_query) / query_inference_time
#     print(f"Query feature extraction time: {query_inference_time:.4f} seconds")
#     print(f"Query FPS: {query_fps:.2f}")
#
#     # 总推理时间和总样本数
#     total_inference_time = gallery_inference_time + query_inference_time
#     total_samples = len(ids_gallery) + len(ids_query)
#
#     # 计算总FPS
#     total_fps = total_samples / total_inference_time
#     print(f"Total FPS (Gallery + Query): {total_fps:.2f}")
#
#     img_features_gallery = img_features_gallery.to(device)
#     img_features_query = img_features_query.to(device)
#     ids_gallery = ids_gallery.to(device)
#     ids_query = ids_query.to(device)
#
#     gl = ids_gallery.cpu().numpy()
#     ql = ids_query.cpu().numpy()
#     save_features_to_mat(img_features_query,img_features_gallery,ql,gl,paths_query, paths_gallery)
#     print("Compute Scores:")
#     CMC = torch.IntTensor(len(ids_gallery)).zero_().to(device)
#     ap = 0.0
#
#     if use_rerank:
#         qf = F.normalize(img_features_query, p=2, dim=1)
#         gf = F.normalize(img_features_gallery, p=2, dim=1)
#         distmat = rerank(qf, gf)  # [num_query, num_gallery]
#
#         for i in tqdm(range(len(ql))):
#             dist = distmat[i].detach().cpu().numpy()
#             ap_tmp, CMC_tmp = eval_query_rerank(dist, ql[i], gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp.to(device)
#             ap += ap_tmp
#     else:
#         for i in tqdm(range(len(ql))):
#             ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
#             if CMC_tmp[0] == -1:
#                 continue
#             CMC += CMC_tmp.to(device)
#             ap += ap_tmp
#
#     AP = ap / len(ids_query) * 100
#     CMC = CMC.float() / len(ids_query)
#
#     top1 = round(len(ids_gallery) * 0.01)
#     string = []
#
#     for i in ranks:
#         string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1].item() * 100))
#
#     string.append('Recall@top1: {:.4f}'.format(CMC[top1].item() * 100))
#     string.append('AP: {:.4f}'.format(AP))
#
#     print(' - '.join(string))
#
#     if cleanup:
#         del img_features_query, ids_query, img_features_gallery, ids_gallery
#         gc.collect()
#         torch.cuda.empty_cache()
#
#     return CMC[0].item()
#
# def eval_query(qf, ql, gf, gl):
#     score = gf @ qf.unsqueeze(-1)
#     score = score.squeeze().detach().cpu().numpy()
#
#     index = np.argsort(score)[::-1]  # from high to low
#     good_index = np.argwhere(gl == ql)
#     junk_index = np.argwhere(gl == -1)
#
#     ap, cmc = compute_mAP(index, good_index, junk_index)
#     return ap, cmc
#
#
# def eval_query_rerank(dist, ql, gl):
#     index = np.argsort(dist)  # distance from small to large
#     good_index = np.argwhere(gl == ql)
#     junk_index = np.argwhere(gl == -1)
#
#     ap, cmc = compute_mAP(index, good_index, junk_index)
#     return ap, cmc
#
#
# def compute_mAP(index, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size == 0:
#         cmc[0] = -1
#         return ap, cmc
#
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]
#
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask == True).flatten()
#
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0 / ngood
#         precision = (i + 1) * 1.0 / (rows_good[i] + 1)
#         old_precision = i * 1.0 / rows_good[i] if rows_good[i] != 0 else 1.0
#         ap += d_recall * (old_precision + precision) / 2
#
#     return ap, cmc
#
#
# def pairwise_distance(x, y):
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
#
# def k_reciprocal_neigh(initial_rank, i, k1):
#     forward_k_neigh_index = initial_rank[i, :k1 + 1]
#     backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#     fi = np.where(backward_k_neigh_index == i)[0]
#     return forward_k_neigh_index[fi]
#
#
# # def rerank(q_feat, g_feat, k1=20, k2=6, alpha=0.7, eval_type=True):
# #     device = q_feat.device
# #     feats = torch.cat([q_feat, g_feat], dim=0)
# #     N, D = feats.size()
# #     query_num = q_feat.size(0)
# #
# #     with torch.no_grad():
# #         dist = pairwise_distance(feats, feats)
# #         original_dist = dist.detach().cpu().numpy()
# #         original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
# #
# #         if eval_type:
# #             dist[:, query_num:] = dist.max()
# #         dist = dist.cpu().numpy()
# #
# #         initial_rank = np.argsort(dist).astype(np.int32)
# #         V = np.zeros((N, N), dtype=np.float32)
# #
# #         for i in range(N):
# #             k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
# #             k_reciprocal_expansion_index = k_reciprocal_index
# #             for candidate in k_reciprocal_index:
# #                 candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, k1 // 2)
# #                 if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (2/3) * len(candidate_k_reciprocal_index):
# #                     k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
# #             k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
# #             weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
# #             V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-6)
# #
# #         if k2 != 1:
# #             V_qe = np.zeros_like(V, dtype=np.float32)
# #             for i in range(N):
# #                 V_qe[i, :] = np.mean(V[initial_rank[i, :k2]], axis=0)
# #             V = V_qe
# #
# #     feats_np = feats.detach().cpu().numpy()
# #     refined_feats = np.matmul(V, feats_np)
# #     refined_feats = alpha * feats_np + (1 - alpha) * refined_feats
# #     refined_feats = torch.from_numpy(refined_feats).to(device)
# #     refined_feats = F.normalize(refined_feats, dim=1)
# #
# #     refined_query = refined_feats[:query_num]
# #     refined_gallery = refined_feats[query_num:]
# #     final_dist = pairwise_distance(refined_query, refined_gallery)
# #
# #     return final_dist
# def rerank(q_feat, g_feat, k1=20, k2=6, alpha=0.7, eval_type=True):
#     device = q_feat.device
#     feats = torch.cat([q_feat, g_feat], dim=0)  # [N, D]
#     N, D = feats.size()
#     query_num = q_feat.size(0)
#
#     with torch.no_grad():
#         # Step 1: 计算 pairwise 距离（GPU）
#         dist = pairwise_distance(feats, feats)  # [N, N]
#         original_dist = dist.clone().detach()
#         original_dist = original_dist / torch.max(original_dist, dim=0, keepdim=True)[0]
#         original_dist = original_dist.t().cpu().numpy()  # [N, N] -> transpose
#
#         if eval_type:
#             dist[:, query_num:] = dist.max()
#
#         # Step 2: 用 torch.topk 替代 argsort，仅保留前 k1+1 个
#         initial_rank = torch.topk(dist, k=30, dim=1, largest=False).indices.cpu().numpy().astype(
#             np.int32)  # [N, k1+1]
#         # initial_rank = torch.argsort(dist).cpu().numpy().astype(np.int32)
#
#         V = np.zeros((N, N), dtype=np.float32)
#
#         for i in range(N):
#             k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
#             k_reciprocal_expansion_index = k_reciprocal_index
#
#             for candidate in k_reciprocal_index:
#                 candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, k1 // 2)
#                 if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (2 / 3) * len(
#                         candidate_k_reciprocal_index):
#                     k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#             k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#             weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#             V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-6)
#
#         if k2 != 1:
#             V_qe = np.zeros_like(V, dtype=np.float32)
#             for i in range(N):
#                 V_qe[i, :] = np.mean(V[initial_rank[i]], axis=0)
#             V = V_qe
#
#     # Step 3: 重新加权后的特征计算
#     feats_np = feats.detach().cpu().numpy()
#     refined_feats = np.matmul(V, feats_np)
#     refined_feats = alpha * feats_np + (1 - alpha) * refined_feats
#     refined_feats = torch.from_numpy(refined_feats).to(device)
#     refined_feats = F.normalize(refined_feats, dim=1)
#
#     refined_query = refined_feats[:query_num]
#     refined_gallery = refined_feats[query_num:]
#     final_dist = pairwise_distance(refined_query, refined_gallery)  # [num_query, num_gallery]
#
#     return final_dist

def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    img_features_query, ids_query = predict(config, model, query_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    # save_features_to_mat(img_features_query,img_features_gallery,ql,gl)
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i],img_features_gallery,gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    top_k = 10
    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)
    top_k_results = index[:top_k]
    # top_k_path =[paths_gallery[i] for i in top_k_results]
    # print(f"Query Image:{query_path}")
    # save_folder='/home/hk/PAPER/UCRVS/DenseUAV-rank'
    # #
    # print(f"Query ID:{ql},Top {top_k} Gallery Results:{gl[top_k_results]}")
    # for rank,gallery_path in enumerate(top_k_path,1):
    #     plot_query_and_gallery(query_path, top_k_path, save_folder)
    #     print(f"Top{rank} Gallery Image:{gallery_path}")
    ## T-tsne
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def save_features_to_mat(query_features, gallery_features, query_labels, gallery_labels, save_path='86.10.mat'):
    """
    保存提取的特征和标签到 .mat 文件
    """
    result = {
        'query_features': query_features.cpu().numpy(),
        'gallery_features': gallery_features.cpu().numpy(),
        'query_labels': query_labels,
        'gallery_labels': gallery_labels,
        # 'paths_query': paths_query,
        # 'paths_gallery': paths_gallery
    }
    scipy.io.savemat(save_path, result)
    print(f'Features and labels saved to {save_path}')

# def plot_query_and_gallery(query_image_path,gallery_image_path,save_folder):
#     fig,axes =plt.subplots(1,len(gallery_image_path) + 1,figsize = (20,5))
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     query_folder = os.path.basename(os.path.dirname(query_image_path))
#     split_path = query_image_path.split('query_satellite',1)[-1]
#     query_filename = split_path.strip(os.sep)
#
#     output_filename = query_filename.replace(os.sep,"_")
#     output_filename = f"{output_filename}_result.png"
#     query_img = cv.imread(query_image_path)
#     query_img = cv.cvtColor(query_img,cv.COLOR_BGR2RGB)
#     axes[0].imshow(query_img)
#     # axes[0].set.title('Query Image')
#     axes[0].axis('off')
#
#     # rect = patches.Rectangle((0,0),query_img.shape[1],query_img.shape[0],
#     #                          linewidth=5 , edgecolor='orange',facecolor='none')
#     # axes[0].add_patch(rect)
#
#     for i,gallery_image_path in enumerate(gallery_image_path):
#         gallery_img = cv.imread(gallery_image_path)
#         gallery_img = cv.cvtColor(gallery_img,cv.COLOR_BGR2RGB)
#         axes[i + 1].imshow(gallery_img)
#         # axes[i + 1].set_totle(f'Gallery {i +1 }')
#         axes[i + 1].axis('off')
#         gallery_folder = os.path.basename(os.path.dirname(gallery_image_path))
#         is_match = query_folder == gallery_folder
#         rect_color ='green' if is_match else 'red'
#         rect = patches.Rectangle((0, 0), query_img.shape[1], query_img.shape[0],
#                                  linewidth=8, edgecolor=rect_color, facecolor='none')
#         axes[i + 1].add_patch(rect)
#     plt.tight_layout()
#     out_path = os.path.join(save_folder,output_filename)
#     plt.savefig(out_path)
#     plt.close()
#     print(f"Image saved to : {out_path}")
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
