# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
from collections import defaultdict
import ipdb

st = ipdb.set_trace


class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=False, thresholds=[0.25, 0.5], topks=[1, 5, 10], prefixes=[]):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes

        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        # self.dets = {
        #     (prefix, t, k, mode): 0 for prefix in self.prefixes for t in self.thresholds for k in self.topks for mode in ["bbs", "bbf"]
        # }  # Number of hit GT boxes, e.g. accuracy at IoU 0.5 for top-1
        # self.gts = dict(self.dets)  # Total number of GT boxes

        self.dets = defaultdict(int)
        self.gts = defaultdict(int)

        self.dets.update({"vd": 0, "vid": 0})
        self.dets.update({"hard": 0, "easy": 0})
        self.dets.update({"multi": 0, "unique": 0})
        self.gts.update({"vd": 1e-14, "vid": 1e-14})
        self.gts.update({"hard": 1e-14, "easy": 1e-14})
        self.gts.update({"multi": 1e-14, "unique": 1e-14})

        # Additional total_acc statistics
        self.dets.update({("total_acc", t, "bbf"): 0 for t in self.thresholds})
        self.gts.update({("total_acc", t, "bbf"): 1e-14 for t in self.thresholds})  # Prevent division by zero

        self.prediction_records = []

    def print_stats(self):
        """Print accumulated accuracies."""
        return_str = "\n"
        mode_str = {"bbs": "Box given span (soft-token)", "bbf": "Box given span (contrastive)"}
        for prefix in ["last_", "proposal_"]:  # self.prefixes: # 我改了，但是如果报错了，可以改回来
            for mode in ["bbs", "bbf"]:
                for t in self.thresholds:
                    line = f"{prefix} {mode_str[mode]} Acc{t:.2f}: " + ", ".join(
                        [f"Top-{k}: {self.dets[(prefix, t, k, mode)] / max(self.gts[(prefix, t, k, mode)], 1):.3f}" for k in self.topks]
                    )
                    # print(line)
                    return_str += line + "\n"

        return_str += "\n==Analysis==\n"

        for t in self.thresholds:
            acc = self.dets[("total_acc", t, "bbf")] / self.gts[("total_acc", t, "bbf")]
            return_str += f"Acc@{t} = {acc:.4f}  "

        return_str += "\n\n"

        # for field in ["easy", "hard", "vd", "vid", "unique", "multi"]:
        #     # print(field, self.dets[field] / self.gts[field])
        #     value = self.dets[field] / self.gts[field]
        #     return_str += f"{field}: {value:.3f}\n"

        return return_str

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    def evaluate(self, batch_data, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(end_points, prefix)
        self.evaluate_bbox_by_contrast(batch_data, end_points, prefix)

    def evaluate_bbox_by_span(self, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)
        # print(f"GT positive_map: {positive_map[0][0][:10]}\nGT gt_bboxes: {gt_bboxes}\n\n")
        # Parse predictions
        sem_scores = end_points[f"{prefix}sem_cls_scores"].softmax(-1)  # B, num_query=256, len_token=256

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(sem_scores.shape[0], sem_scores.shape[1], positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, : sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f"{prefix}center"]  # B, Q, 3
        pred_size = end_points[f"{prefix}pred_size"]  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)  # B, Q=256, 6, each query corresponds to a box

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points["box_label_mask"][bid].sum())  # 1
            pmap = positive_map[bid, :num_obj]
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)  # (1, Q, 256)  # (obj, 1, 256)  # (obj, Q) # Score of each query for target token

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10) # Sort each GT (only 1 here) and get top 10 queries
            pbox = pred_bbox[bid, top.reshape(-1)]  #  # Query indices, sorted by score from high to low

            # IoU
            ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]), box_cxcyczwhd_to_xyzxyz(pbox))  # IoU between 10 queries and this gt box
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks  # [1, 5, 10]
            for t in self.thresholds:  # 0.25, 0.5
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)  # Top-1: Check if any of first 1 has IoU > 0.5 # ious[:, :1] = [0.55] > 0.5
                    # NOTE bbs is "Box given span (soft-token)"
                    self.dets[(prefix, t, k, "bbs")] += found.sum().item()  # Number of hit GT boxes
                    self.gts[(prefix, t, k, "bbs")] += len(thresholded)  # Total number of GT boxes

    def evaluate_bbox_by_contrast(self, batch_data, end_points, prefix):
        """
        Evaluate bounding box IoU by contrasting with span features.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt
        positive_map, gt_bboxes = self._parse_gt(end_points)
        # print(f"GT positive_map: {positive_map[0][0][:10]}\nGT gt_bboxes: {gt_bboxes}\n\n")
        # Parse predictions
        pred_center = end_points[f"{prefix}center"]  # B, Q, 3
        pred_size = end_points[f"{prefix}pred_size"]  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)  # Predicted bbox

        proj_tokens = end_points["proj_tokens"]  # (B, tokens, 64) # NOTE Text features
        # After projection layer (fully connected layer) and L2 normalization, get token representation in contrastive learning space.
        # This is the key in contrastive learning, representing language semantic space.
        proj_queries = end_points[f"{prefix}proj_queries"]  # (B, Q, 64)
        # NOTE After each decoder layer, project query features (3D scene candidate box features) to contrastive space and normalize.
        # This is the query in contrastive learning, representing your model's understanding of point cloud regions/3D proposals.
        sem_scores = torch.matmul(
            proj_queries, proj_tokens.transpose(-1, -2)
        )  # NOTE Semantic similarity (dot product) between each 3D candidate box (query) and each language token
        sem_scores_ = (sem_scores / 0.07).softmax(-1)  # (B, Q, tokens) Here 0.07 is the common temperature parameter in contrastive learning to adjust distribution smoothness
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)  # Pad to 256 dimensions
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, : sem_scores_.size(1), : sem_scores_.size(2)] = sem_scores_

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points["box_label_mask"][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)  # [Q=1, 256] Same as span logic, filter scores corresponding to target tokens using positive_map
            # positive_map = tensor([[0., 1., 0., 0., 0., 0., ..., 0.]])  # Only focus on 2nd token

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]), box_cxcyczwhd_to_xyzxyz(pbox))  # (obj, 6)  # (obj*10, 6)  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            meta_path = batch_data["meta_path"][bid]
            dataset = meta_path.split("/")[-4]
            sequence = meta_path.split("/")[-3]
            frame = meta_path.split("/")[-2]

            record = {
                "id": f"{dataset}/{sequence}/{frame}",
                "utterance": batch_data["utterances"][bid],
                "gt_box": batch_data["gt_bboxes"][bid][:num_obj].cpu().numpy().tolist(),  # num_obj, 7
                "pred_box": pred_bbox[bid, top[:, 0]].cpu().numpy().tolist(),
                "ious": ious[:, 0].cpu().numpy().tolist(),
            }
            self.prediction_records.append(record)

            self.dets["iou"] += ious[:, 0].cpu().numpy().sum()
            self.dets["num_iou"] += num_obj

            # Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t  # num_objs

                for k in self.topks:
                    found = thresholded[:, :k].any(1)  # num_objs
                    all_found = found.all().item()

                    # NOTE bbf is "Box given span (contrastive)"
                    self.dets[(prefix, t, k, "bbf")] += all_found  # found.sum().item()
                    self.gts[(prefix, t, k, "bbf")] += 1  # len(thresholded)

                    # Only consider top-1 case (highest scoring prediction)
                    if prefix == "last_" and k == 1:  # NOTE Only last_ layer pred considered here
                        # found = found[0].item()
                        self.dets[("total_acc", t, "bbf")] += all_found  # found
                        self.gts[("total_acc", t, "bbf")] += 1

                        if t == self.thresholds[0]:  # iou 0.25
                            if end_points["is_view_dep"][bid]:
                                self.gts["vd"] += 1
                                self.dets["vd"] += found
                            else:
                                self.gts["vid"] += 1
                                self.dets["vid"] += found
                            if end_points["is_hard"][bid]:
                                self.gts["hard"] += 1
                                self.dets["hard"] += found
                            else:
                                self.gts["easy"] += 1
                                self.dets["easy"] += found
                            if end_points["is_unique"][bid]:
                                self.gts["unique"] += 1
                                self.dets["unique"] += found
                            else:
                                self.gts["multi"] += 1
                                self.dets["multi"] += found

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points["positive_map"])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points["center_label"][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points["size_gts"]  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        if self.only_root:  # MARK ony first object if true
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes
