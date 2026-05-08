import numpy as np
import torch
import torch.nn.functional as F
# def k_reciprocal_neigh( initial_rank, i, k1):
#     forward_k_neigh_index = initial_rank[i,:k1+1]
#     backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
#     fi = np.where(backward_k_neigh_index==i)[0]
#     return forward_k_neigh_index[fi]
#
# def pairwise_distance(query_features, gallery_features):
#     x = query_features
#     y = gallery_features
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
# def re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, eval_type=True):
#     # The following naming, e.g. gallery_num, is different from outer scope.
#     # Don't care about it.
#     feats = torch.cat([q_feat, g_feat], 0)
#     dist = pairwise_distance(feats, feats)
#     original_dist = dist.detach().cpu().numpy()
#     all_num = original_dist.shape[0]
#     original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
#     V = np.zeros_like(original_dist).astype(np.float16)
#
#     query_num = q_feat.size(0)
#     all_num = original_dist.shape[0]
#     if eval_type:
#         dist[:, query_num:] = dist.max()
#     dist = dist.detach().cpu().numpy()
#     initial_rank = np.argsort(dist).astype(np.int32)
#
#     # print("start re-ranking")
#     for i in range(all_num):
#         # k-reciprocal neighbors
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = np.where(backward_k_neigh_index == i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
#                                                 :int(np.around(k1 / 2)) + 1]
#             # import pdb
#             # pdb.set_trace()
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
#                     candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#         weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float16)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append(np.where(V[:, i] != 0)[0])
#
#     jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
#
#
#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
#                                                                                 V[indImages[j], indNonZero[j]])
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num, query_num:]
#     return final_dist



# def pairwise_distance(query_features, gallery_features):
#     x = query_features
#     y = gallery_features
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
# def soft_reciprocal_rerank(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, expansion_thresh=0.5, eval_type=True):
#     feats = torch.cat([q_feat, g_feat], 0)
#     dist = pairwise_distance(feats, feats)
#     original_dist = dist.detach().cpu().numpy()
#     all_num = original_dist.shape[0]
#     original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
#     V = np.zeros_like(original_dist).astype(np.float32)
#
#     query_num = q_feat.size(0)
#     if eval_type:
#         dist[:, query_num:] = dist.max()
#     dist = dist.detach().cpu().numpy()
#     initial_rank = np.argsort(dist).astype(np.int32)
#
#     # Soft reciprocal re-ranking
#     for i in range(all_num):
#         forward_k_neigh_index = initial_rank[i, :k1+1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
#         fi = np.where(backward_k_neigh_index == i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate, :k1//2+1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :k1//2+1]
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#
#             # 加相似度过滤条件
#             if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2/3 * len(candidate_k_reciprocal_index):
#                 if np.mean(original_dist[i, candidate_k_reciprocal_index]) < expansion_thresh:
#                     k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#
#         weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
#
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append(np.where(V[:, i] != 0)[0])
#
#     jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
#
#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] += (V[i, indNonZero[j]] * V[indImages[j], indNonZero[j]]) ** 0.5  # soft-jaccard
#
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num, query_num:]
#     return final_dist

import numpy as np
import torch
from tqdm import tqdm


# def compute_pairwise_dist(x, y):
#     """Compute pairwise Euclidean distance."""
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
#
# def mutual_reciprocal(initial_rank, i, k1, threshold=0.7):
#     """Find mutual neighbors under similarity constraint."""
#     forward = initial_rank[i, :k1 + 1]
#     reciprocal = []
#     for candidate in forward:
#         candidate_forward = initial_rank[candidate, :k1 + 1]
#         overlap = len(np.intersect1d(forward, candidate_forward))
#         if overlap / k1 >= threshold:
#             reciprocal.append(candidate)
#     return np.array(reciprocal, dtype=np.int32)
#
#
# def local_reweight(dist_vec, scale=1.0):
#     """Locally adjust neighbor weights."""
#     mean_val = np.mean(dist_vec)
#     adjusted = np.exp(-(dist_vec - mean_val) / (scale + 1e-5))
#     return adjusted / np.sum(adjusted)
#
#
# def rerank(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, mutual_thresh=0.7, reweight_scale=0.5,
#                       eval_type=True):
#     """Adaptive Neighborhood Re-ranking."""
#     feats = torch.cat([q_feat, g_feat], dim=0)
#     print("Computing original pairwise distance...")
#     dist_mat = compute_pairwise_dist(feats, feats)
#     dist_np = dist_mat.detach().cpu().numpy()
#     dist_np = np.transpose(dist_np / np.max(dist_np, axis=0))
#     all_num = dist_np.shape[0]
#     V = np.zeros_like(dist_np, dtype=np.float32)
#
#     query_num = q_feat.size(0)
#     if eval_type:
#         dist_mat[:, query_num:] = dist_mat.max()
#     initial_rank = np.argsort(dist_mat.cpu().numpy())
#
#     print("Building k-reciprocal neighbors...")
#     for i in tqdm(range(all_num)):
#         reciprocal_neigh = mutual_reciprocal(initial_rank, i, k1, threshold=mutual_thresh)
#         expand_set = set(reciprocal_neigh)
#
#         for neighbor in reciprocal_neigh:
#             candidate_reciprocal = mutual_reciprocal(initial_rank, neighbor, k1 // 2, threshold=mutual_thresh)
#             if len(np.intersect1d(candidate_reciprocal, reciprocal_neigh)) > (2 / 3) * len(candidate_reciprocal):
#                 expand_set.update(candidate_reciprocal)
#
#         expand_indices = np.array(list(expand_set), dtype=np.int32)
#
#         if len(expand_indices) > 0:
#             try:
#                 weights = local_reweight(dist_np[i, expand_indices], scale=reweight_scale)
#             except Exception as e:
#                 # fallback: simple uniform weighting
#                 weights = np.ones(len(expand_indices)) / len(expand_indices)
#             V[i, expand_indices] = weights
#
#     if k2 != 1:
#         print("Applying feature expansion with k2...")
#         V_qe = np.zeros_like(V, dtype=np.float32)
#         for i in tqdm(range(all_num)):
#             V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
#         V = V_qe
#
#     print("Building inverted index...")
#     invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
#
#     print("Calculating Jaccard distance...")
#     jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
#
#     for i in tqdm(range(query_num)):
#         temp_min = np.zeros((1, all_num), dtype=np.float32)
#         nonzero_inds = np.where(V[i] != 0)[0]
#         for j, ind in enumerate(nonzero_inds):
#             related = invIndex[ind]
#             temp_min[0, related] += np.minimum(V[i, ind], V[related, ind])
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     print("Fusing Jaccard and original distance...")
#     adaptive_lambda = np.clip(np.mean(jaccard_dist, axis=1, keepdims=True), 0.2, 0.8)
#     final_dist = jaccard_dist[:, query_num:] * (1 - adaptive_lambda) + dist_np[:query_num, query_num:] * adaptive_lambda
#
#     return final_dist
## new
# def compute_pairwise_dist(x, y):
#     """Compute pairwise Euclidean distance."""
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
# def mutual_reciprocal(initial_rank, i, k1, threshold=0.7):
#     """Find mutual neighbors under similarity constraint."""
#     forward = initial_rank[i, :k1 + 1]
#     reciprocal = []
#     for candidate in forward:
#         candidate_forward = initial_rank[candidate, :k1 + 1]
#         overlap = len(np.intersect1d(forward, candidate_forward))
#         if overlap / k1 >= threshold:
#             reciprocal.append(candidate)
#     return np.array(reciprocal, dtype=np.int32)
#
# def local_reweight(dist_vec, scale=1.0):
#     """Locally adjust neighbor weights."""
#     mean_val = np.mean(dist_vec)
#     adjusted = np.exp(-(dist_vec - mean_val) / (scale + 1e-5))
#     return adjusted / np.sum(adjusted)
#
# def soft_min(a, b, eps=1e-6):
#     """Soft minimum instead of hard minimum."""
#     return (a * b) / (a + b + eps)
#
# def rerank(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, mutual_thresh=0.7, reweight_scale=0.5, eval_type=True):
#     """Double-Scale Adaptive Neighborhood Consistency Optimization."""
#     feats = torch.cat([q_feat, g_feat], dim=0)
#     dist_mat = compute_pairwise_dist(feats, feats)
#     dist_np = dist_mat.detach().cpu().numpy()
#     dist_np = np.transpose(dist_np / np.max(dist_np, axis=0))
#     all_num = dist_np.shape[0]
#     V = np.zeros_like(dist_np, dtype=np.float32)
#
#     query_num = q_feat.size(0)
#     if eval_type:
#         dist_mat[:, query_num:] = dist_mat.max()
#     initial_rank = np.argsort(dist_mat.cpu().numpy())
#
#     for i in range(all_num):
#         primary_neigh = mutual_reciprocal(initial_rank, i, k1, threshold=mutual_thresh)
#         secondary_neigh = mutual_reciprocal(initial_rank, i, k1 // 2, threshold=mutual_thresh)
#         expand_set = set(primary_neigh) | set(secondary_neigh)
#
#         for neighbor in primary_neigh:
#             candidate_reciprocal = mutual_reciprocal(initial_rank, neighbor, k1 // 2, threshold=mutual_thresh)
#             if len(np.intersect1d(candidate_reciprocal, primary_neigh)) > (2/3) * len(candidate_reciprocal):
#                 expand_set.update(candidate_reciprocal)
#
#         expand_indices = np.array(list(expand_set), dtype=np.int32)
#
#         if len(expand_indices) > 0:
#             try:
#                 weights = local_reweight(dist_np[i, expand_indices], scale=reweight_scale)
#             except Exception:
#                 weights = np.ones(len(expand_indices)) / len(expand_indices)
#             V[i, expand_indices] = weights
#
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
#         V = V_qe
#
#     invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
#
#     jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
#
#     for i in range(query_num):
#         temp_soft_min = np.zeros((1, all_num), dtype=np.float32)
#         nonzero_inds = np.where(V[i] != 0)[0]
#         for j, ind in enumerate(nonzero_inds):
#             related = invIndex[ind]
#             temp_soft_min[0, related] += soft_min(V[i, ind], V[related, ind])
#         jaccard_dist[i] = 1 - temp_soft_min / (2 - temp_soft_min)
#
#     adaptive_lambda = np.clip(np.mean(jaccard_dist, axis=1, keepdims=True), 0.2, 0.8)
#     final_dist = jaccard_dist[:, query_num:] * (1 - adaptive_lambda) + dist_np[:query_num, query_num:] * adaptive_lambda
#
#     return final_dist


# def rerank(query_feat, gallery_feat, k1=20, alpha=0.7):
#     """
#     Feature refinement guided by mutual neighbors, then compute pairwise distance.
#
#     Args:
#         query_feat (torch.Tensor): Query features [num_query, D]
#         gallery_feat (torch.Tensor): Gallery features [num_gallery, D]
#         k1 (int): Top-k neighbors for mutual neighbor selection
#         alpha (float): Self-feature fusion weight
#
#     Returns:
#         dist_mat (torch.Tensor): Final distance matrix [num_query, num_gallery]
#     """
#     all_features = torch.cat([query_feat, gallery_feat], dim=0)  # [N, D]
#     N, D = all_features.size()
#     device = all_features.device
#
#     # Step 1: Compute pairwise distance
#     dist_mat_full = torch.cdist(all_features, all_features, p=2)  # [N, N]
#     initial_rank = dist_mat_full.argsort(dim=1)  # ascending order
#
#     # Step 2: Feature refinement
#     refined_features = []
#     for i in range(N):
#         forward_neighbors = initial_rank[i, 1:k1+1]
#         reciprocal_set = []
#
#         for j in forward_neighbors:
#             backward_neighbors = initial_rank[j, 1:k1+1]
#             if i in backward_neighbors:
#                 reciprocal_set.append(j)
#
#         reciprocal_set = torch.tensor(reciprocal_set, device=device)
#
#         if len(reciprocal_set) == 0:
#             refined_features.append(all_features[i].unsqueeze(0))
#             continue
#
#         # Weighted neighbor fusion
#         sim_scores = 1.0 / (dist_mat_full[i, reciprocal_set] + 1e-6)
#         sim_scores = sim_scores / (sim_scores.sum() + 1e-6)
#
#         neighbor_feats = all_features[reciprocal_set]
#         weighted_neighbor = (sim_scores.unsqueeze(1) * neighbor_feats).sum(dim=0, keepdim=True)
#
#         enhanced_feat = alpha * all_features[i].unsqueeze(0) + (1 - alpha) * weighted_neighbor
#         refined_features.append(enhanced_feat)
#
#     refined_features = torch.cat(refined_features, dim=0)  # [N, D]
#     refined_features = torch.nn.functional.normalize(refined_features, dim=1)  # L2 normalize
#
#     # Step 3: Split refined query and gallery
#     refined_query = refined_features[:query_feat.size(0)]
#     refined_gallery = refined_features[query_feat.size(0):]
#
#     # Step 4: Compute final distance
#     dist_mat = torch.cdist(refined_query, refined_gallery, p=2)  # [num_query, num_gallery]
#
#     return dist_mat

# def compute_pairwise_dist(x, y):
#     """Compute pairwise Euclidean distance."""
#     m, n = x.size(0), y.size(0)
#     x = x.view(m, -1)
#     y = y.view(n, -1)
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist
#
#
# def soft_min(a, b, eps=1e-6):
#     """Soft minimum operation."""
#     return (a * b) / (a + b + eps)
#
#
# def local_reweight(dist_vec, scale=1.0):
#     """Local neighbor reweighting based on distance."""
#     mean_val = np.mean(dist_vec)
#     adjusted = np.exp(-(dist_vec - mean_val) / (scale + 1e-5))
#     return adjusted / np.sum(adjusted)
#
#
# def mutual_reciprocal(initial_rank, i, k1, threshold=0.7):
#     """Mutual neighbor selection."""
#     forward = initial_rank[i, :k1 + 1]
#     reciprocal = []
#     for candidate in forward:
#         candidate_forward = initial_rank[candidate, :k1 + 1]
#         overlap = len(np.intersect1d(forward, candidate_forward))
#         if overlap / k1 >= threshold:
#             reciprocal.append(candidate)
#     return np.array(reciprocal, dtype=np.int32)
#
#
# def rerank(q_feat, g_feat, k1=20, k2=6, mutual_thresh=0.7, reweight_scale=0.5, alpha=0.7):
#     """
#     Refine features based on mutual neighborhood consistency and soft aggregation.
#
#     Args:
#         q_feat: [num_query, D]
#         g_feat: [num_gallery, D]
#
#     Returns:
#         distance matrix: [num_query, num_gallery]
#     """
#     device = q_feat.device
#     feats = torch.cat([q_feat, g_feat], dim=0)  # [N, D]
#     all_num = feats.size(0)
#     query_num = q_feat.size(0)
#
#     # Compute pairwise distance
#     dist_mat = compute_pairwise_dist(feats, feats)
#     dist_np = dist_mat.detach().cpu().numpy()
#     dist_np = np.transpose(dist_np / np.max(dist_np, axis=0))
#     initial_rank = np.argsort(dist_np)
#
#     # Local neighbor affinity
#     V = np.zeros((all_num, all_num), dtype=np.float32)
#
#     for i in range(all_num):
#         primary_neigh = mutual_reciprocal(initial_rank, i, k1, threshold=mutual_thresh)
#         secondary_neigh = mutual_reciprocal(initial_rank, i, k1 // 2, threshold=mutual_thresh)
#         expand_set = set(primary_neigh) | set(secondary_neigh)
#
#         for neighbor in primary_neigh:
#             candidate_reciprocal = mutual_reciprocal(initial_rank, neighbor, k1 // 2, threshold=mutual_thresh)
#             if len(np.intersect1d(candidate_reciprocal, primary_neigh)) > (2 / 3) * len(candidate_reciprocal):
#                 expand_set.update(candidate_reciprocal)
#
#         expand_indices = np.array(list(expand_set), dtype=np.int32)
#         if len(expand_indices) > 0:
#             try:
#                 weights = local_reweight(dist_np[i, expand_indices], scale=reweight_scale)
#             except Exception:
#                 weights = np.ones(len(expand_indices)) / len(expand_indices)
#             V[i, expand_indices] = weights
#
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
#         V = V_qe
#
#     # --- 特征优化阶段 ---
#     feats_np = feats.detach().cpu().numpy()
#     refined_feats = np.zeros_like(feats_np)
#
#     for i in range(all_num):
#         weights = V[i]
#         nonzero_inds = np.where(weights > 0)[0]
#         if len(nonzero_inds) == 0:
#             refined_feats[i] = feats_np[i]
#             continue
#         neighbor_feats = feats_np[nonzero_inds]
#         neighbor_weights = weights[nonzero_inds]
#         weighted_neighbor = (neighbor_weights[:, None] * neighbor_feats).sum(axis=0)
#         weighted_neighbor = weighted_neighbor / (neighbor_weights.sum() + 1e-6)
#
#         refined_feats[i] = alpha * feats_np[i] + (1 - alpha) * weighted_neighbor
#
#     refined_feats = torch.from_numpy(refined_feats).to(device)
#     refined_feats = torch.nn.functional.normalize(refined_feats, dim=1)
#
#     refined_query = refined_feats[:query_num]
#     refined_gallery = refined_feats[query_num:]
#
#     # Final pairwise distance
#     final_dist = torch.cdist(refined_query, refined_gallery, p=2)
#
#     return final_dist

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def rerank(q_feat, g_feat, k1=20, k2=6, alpha=0.7, eval_type=True):
    """
    Fast Feature-level re-ranking optimization: directly output distance matrix.

    Args:
        q_feat: [num_query, D]
        g_feat: [num_gallery, D]

    Returns:
        final_dist: [num_query, num_gallery]
    """
    device = q_feat.device
    feats = torch.cat([q_feat, g_feat], dim=0)  # [N, D]
    N, D = feats.size()
    query_num = q_feat.size(0)

    # Step 1: Compute pairwise distance
    with torch.no_grad():
        dist = pairwise_distance(feats, feats)  # [N, N]
        original_dist = dist.detach().cpu().numpy()
        original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))

        if eval_type:
            dist[:, query_num:] = dist.max()
        dist = dist.cpu().numpy()

        initial_rank = np.argsort(dist).astype(np.int32)

        # Step 2: Build neighbor graph (V matrix)
        V = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index

            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, k1//2)
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (2/3)*len(candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-6)

        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(N):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2]], axis=0)
            V = V_qe

    # Step 3: Feature updating (矩阵乘法批处理，快速版)
    feats_np = feats.detach().cpu().numpy()  # [N, D]
    refined_feats = np.matmul(V, feats_np)  # [N, D]
    refined_feats = alpha * feats_np + (1 - alpha) * refined_feats
    refined_feats = torch.from_numpy(refined_feats).to(device)
    refined_feats = torch.nn.functional.normalize(refined_feats, dim=1)

    # Step 4: Compute final distance
    refined_query = refined_feats[:query_num]
    refined_gallery = refined_feats[query_num:]

    return refined_query,refined_gallery





