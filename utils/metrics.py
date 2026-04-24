from re import T
from time import time
import torch
import numpy as np
import os
import logging
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_hard_class(distmat, q_pids, g_pids, q_camids, g_camids,
                         q_classes, g_classes, max_rank=50):
    """Evaluation with per-query hard class filtering.
    
    For each query q, the gallery is truncated to:
        G_hard = {g_i in G | C(g_i) = C(q)}
    Then ranking and metric computation happen only within G_hard.
    
    Args:
        distmat: distance matrix [num_q, num_g]
        q_pids: query person IDs
        g_pids: gallery person IDs
        q_camids: query camera IDs
        g_camids: gallery camera IDs
        q_classes: query class labels (list of strings)
        g_classes: gallery class labels (list of strings)
        max_rank: max rank for CMC
    """
    logger = logging.getLogger('PAT')
    num_q, num_g = distmat.shape

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    num_skipped = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_class = q_classes[q_idx]

        # Hard class filtering: keep only gallery items with the same class as query
        class_mask = np.array([g_classes[i] == q_class for i in range(num_g)])
        g_indices_in_class = np.where(class_mask)[0]

        if len(g_indices_in_class) == 0:
            num_skipped += 1
            continue

        # Extract the sub-distance row for this query against same-class gallery only
        sub_distmat = distmat[q_idx, g_indices_in_class]
        sub_g_pids = g_pids[g_indices_in_class]
        sub_g_camids = g_camids[g_indices_in_class]

        # Sort within the filtered gallery
        sub_order = np.argsort(sub_distmat)
        sub_matches = (sub_g_pids[sub_order] == q_pid).astype(np.int32)

        # Remove gallery samples with same pid AND same camid as query
        remove = (sub_g_pids[sub_order] == q_pid) & (sub_g_camids[sub_order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = sub_matches[keep]
        if not np.any(orig_cmc):
            # query identity does not appear in filtered gallery
            continue

        local_max_rank = min(max_rank, len(orig_cmc))

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        # Pad to max_rank if shorter
        if len(cmc) < max_rank:
            cmc_padded = np.ones(max_rank)
            cmc_padded[:len(cmc)] = cmc[:len(cmc)]
            all_cmc.append(cmc_padded)
        else:
            all_cmc.append(cmc[:max_rank])

        num_valid_q += 1.

        # Average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_skipped > 0:
        logger.info("[HardClassFilter] Skipped {} queries (no same-class gallery items)".format(num_skipped))

    assert num_valid_q > 0, "Error: all query identities do not appear in same-class gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    logger.info("[HardClassFilter] Evaluated {} valid queries (per-query class-filtered gallery)".format(
        int(num_valid_q)))

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def update_with_paths(self, output):  # called once for each batch — includes img paths
        feat, pid, camid, imgpath = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(imgpath)

    def compute(self, imgpath_to_class=None):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        # If hard class filtering is enabled, use per-query class-aware evaluation
        if imgpath_to_class and self.img_paths:
            q_paths = self.img_paths[:self.num_query]
            g_paths = self.img_paths[self.num_query:]
            q_classes = [imgpath_to_class.get(p, '') for p in q_paths]
            g_classes = [imgpath_to_class.get(p, '') for p in g_paths]

            # Check if we actually have class labels
            has_classes = any(c != '' for c in q_classes) and any(c != '' for c in g_classes)
            if has_classes:
                cmc, mAP = eval_func_hard_class(distmat, q_pids, g_pids, q_camids, g_camids,
                                                q_classes, g_classes)
                return cmc, mAP, distmat, self.pids, self.camids, qf, gf

        # Standard evaluation (no class filtering)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



