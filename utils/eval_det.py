# ------------------------------------------------------------------------
# Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import ipdb

st = ipdb.set_trace

VALID_TEST_CLASSES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    75,
    76,
    77,
    79,
    80,
    81,
    82,
    83,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    107,
    108,
    109,
    111,
    113,
    115,
    116,
    117,
    118,
    119,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    129,
    131,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    141,
    142,
    143,
    144,
    146,
    147,
    148,
    150,
    151,
    152,
    154,
    155,
    156,
    157,
    158,
    160,
    161,
    163,
    165,
    166,
    167,
    168,
    172,
    174,
    175,
    177,
    178,
    179,
    181,
    182,
    183,
    185,
    187,
    188,
    191,
    192,
    193,
    194,
    195,
    198,
    201,
    203,
    204,
    205,
    206,
    208,
    209,
    210,
    214,
    215,
    216,
    218,
    222,
    223,
    225,
    230,
    231,
    234,
    235,
    237,
    238,
    240,
    241,
    250,
    255,
    256,
    258,
    260,
    262,
    263,
    264,
    267,
    270,
    276,
    280,
    282,
    284,
    285,
    286,
    288,
    290,
    304,
    306,
    312,
    313,
    314,
    315,
    321,
    323,
    324,
    325,
    326,
    327,
    328,
    329,
    330,
    331,
    333,
    334,
    335,
    337,
    338,
    339,
    341,
    342,
    343,
    344,
    346,
    347,
    348,
    349,
    350,
    351,
    353,
    354,
    355,
    357,
    360,
    361,
    363,
    364,
    368,
    369,
    371,
    372,
    373,
    381,
    384,
    389,
    396,
    398,
    402,
    413,
    414,
    422,
    426,
    427,
    446,
    455,
]


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
from metric_util import calc_iou  # axis-aligned 3D box IoU


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d), (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def corners_to_ends(box):
    min_xyz = torch.min(box, axis=0)[0]
    max_xyz = torch.max(box, axis=0)[0]
    return torch.cat((min_xyz, max_xyz))


def _volume_par(box):
    return (box[:, 3] - box[:, 0]) * (box[:, 4] - box[:, 1]) * (box[:, 5] - box[:, 2])


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return torch.clamp(xB - xA, 0) * torch.clamp(yB - yA, 0) * torch.clamp(zB - zA, 0)


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou = torch.zeros((N,M)).to(boxes1.device)
    union = torch.zeros((N,M)).to(boxes1.device)
    for n in range(N):
        for m in range(M):
            iou[n,m], union[n,m] = _iou3d(boxes1[n], boxes2[m])
    """
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


def get_iou(bb1, bb2):
    """Compute IoU of two bounding boxes.
    ** Define your bod IoU function HERE **
    """
    # pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d


from box_util import box3d_iou


def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection
    for a single class.
    Input:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # if npos==0:
    #     st()

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # if npos==0:
    #     rec = np.zeros(tp.shape, dtype=np.float64)
    #     # print(tp.shape)
    # else:
    rec = tp / float(npos + 1e-8)

    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        # print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
        # print(classname, ap[classname])

    return rec, prec, ap


from multiprocessing import Pool


def eval_det_multiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            # if classname not in VALID_TEST_CLASSES:
            #     continue
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            # if classname not in VALID_TEST_CLASSES:
            #     continue
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper, [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        # print(classname, ap[classname])

    return rec, prec, ap


def eval_grounding(pred_all, gt_all, ovthresh=0.25):
    """Generic functions to compute accuracy for grounding
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        acc: accuracy
    """
    k = (1, 5, 10)
    # k = ('exact', 3, 5)
    dataset2score = {k_: 0.0 for k_ in k}
    dataset2count = 0.0
    for img_id in pred_all.keys():
        target = gt_all[img_id]
        prediction = pred_all[img_id]
        assert prediction is not None
        # sort by scores
        sorted_scores_boxes = sorted(prediction, key=lambda x: x[2], reverse=True)
        _, sorted_boxes, sorted_scores = zip(*sorted_scores_boxes)
        sorted_boxes = torch.cat([corners_to_ends(torch.as_tensor(x)).view(1, 6) for x in sorted_boxes])
        target_box = torch.cat([corners_to_ends(torch.as_tensor(t[1])).view(-1, 6) for t in target[:1]])
        giou = generalized_box_iou3d(sorted_boxes, target_box)
        for g in range(giou.shape[1]):
            for k_ in k:
                if k_ == "exact":
                    if max(giou[:1, g]) >= ovthresh:
                        dataset2score[k_] += 1.0 / giou.shape[1]
                else:
                    if max(giou[:k_, g]) >= ovthresh:
                        dataset2score[k_] += 1.0 / giou.shape[1]
        dataset2count += 1.0

    for k_ in k:
        try:
            dataset2score[k_] /= dataset2count
        except:
            pass

    # results = sorted([v for k, v in dataset2score.items()])
    # print(f"Accuracy @ 1, 5, 10: {results} @IOU: {ovthresh} \n")

    return dataset2score


def bbox_7d9d_to_corners(bbox):
    """
    Convert 7/9D bbox to 8 corners (supports full 3D rotation)

    Args:
        bbox: (N, 7) or (N, 9) - [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, pitch, roll]
              or (7,) or (9,) - a single bbox
              Coordinate system: X-forward/backward(l), Y-left/right(w), Z-up/down(h)
              Rotations: yaw (around Z axis), pitch (around Y axis), roll (around X axis)
              Rotation order: Rz * Ry * Rx

    Returns:
        corners: (N, 8, 3) or (8, 3) - coordinates of 8 corners
                 First 4 points: Z=h/2 (top layer), last 4 points: Z=-h/2 (bottom layer)
    """
    
    is_numpy = isinstance(bbox, np.ndarray)
    is_torch = isinstance(bbox, torch.Tensor)
    
    if is_numpy:
        bbox = torch.from_numpy(bbox).float()
    
    # Handle single bbox
    single_bbox = False
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0)
        single_bbox = True
    
    # Extract parameters
    x, y, z = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    l, w, h = bbox[:, 3], bbox[:, 4], bbox[:, 5]
    yaw = bbox[:, 6]
    
    # Handle pitch and roll (for 9D bbox)
    if bbox.shape[1] >= 8:
        pitch = bbox[:, 7] if bbox.shape[1] >= 8 else torch.zeros_like(yaw)
        roll = bbox[:, 8] if bbox.shape[1] >= 9 else torch.zeros_like(yaw)
    else:
        pitch = torch.zeros_like(yaw)
        roll = torch.zeros_like(yaw)
    
    N = bbox.shape[0]
    
    # Local coordinates of 8 corners (in the object coordinate system)
    # X: ±l/2, Y: ±w/2, Z: ±h/2
    # First 4 points (index 0-3): Z = h/2 (top layer)
    # Last 4 points (index 4-7): Z = -h/2 (bottom layer)
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)  # (N, 8)
    y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)  # (N, 8)
    z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)  # (N, 8)
    
    # Stack corners to shape (N, 8, 3)
    corners_local = torch.stack([x_corners, y_corners, z_corners], dim=-1)  # (N, 8, 3)
    
    # Build full 3D rotation matrix: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # Rz: rotate around Z axis (yaw)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    Rz = torch.zeros(N, 3, 3).to(bbox.device)
    Rz[:, 0, 0] = cos_yaw
    Rz[:, 0, 1] = -sin_yaw
    Rz[:, 1, 0] = sin_yaw
    Rz[:, 1, 1] = cos_yaw
    Rz[:, 2, 2] = 1
    
    # Ry: rotate around Y axis (pitch)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    Ry = torch.zeros(N, 3, 3).to(bbox.device)
    Ry[:, 0, 0] = cos_pitch
    Ry[:, 0, 2] = sin_pitch
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_pitch
    Ry[:, 2, 2] = cos_pitch
    
    # Rx: rotate around X axis (roll)
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    Rx = torch.zeros(N, 3, 3).to(bbox.device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_roll
    Rx[:, 1, 2] = -sin_roll
    Rx[:, 2, 1] = sin_roll
    Rx[:, 2, 2] = cos_roll
    
    # Compose rotation matrix: R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))  # (N, 3, 3)
    
    # Apply rotation: corners_rotated = corners_local @ R^T
    corners_rotated = torch.bmm(corners_local, R.transpose(1, 2))  # (N, 8, 3)
    
    # Extract rotated coordinates
    x_rot = corners_rotated[..., 0]
    y_rot = corners_rotated[..., 1]
    z_rot = corners_rotated[..., 2]
    
    # Translate to world coordinates
    x_global = x_rot + x.unsqueeze(1)
    y_global = y_rot + y.unsqueeze(1)
    z_global = z_rot + z.unsqueeze(1)
    
    # Combine into corners (N, 8, 3)
    corners = torch.stack([x_global, y_global, z_global], dim=-1)
    
    if single_bbox:
        corners = corners.squeeze(0)
    
    if is_numpy:
        corners = corners.numpy()
    
    return corners


def bbox_6d_to_corners(bbox):
    """
    Convert 6D axis-aligned bbox to 8 corners

    Args:
        bbox: (N, 6) - [cx, cy, cz, l, w, h] or (6,) - a single bbox
              Coordinate system: length l along X, width w along Y, height h along Z, axis-aligned without rotation
              Exactly consistent with the first 6 dims of GT bbox!

    Returns:
        corners: (N, 8, 3) or (8, 3) - coordinates of 8 corners
                 First 4 points: Z=cz+h/2 (top layer), last 4 points: Z=cz-h/2 (bottom layer)
    """
    
    is_numpy = isinstance(bbox, np.ndarray)
    if is_numpy:
        bbox = torch.from_numpy(bbox).float()
    
    # Handle single bbox
    single_bbox = False
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0)
        single_bbox = True
    
    cx, cy, cz = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    l, w, h = bbox[:, 3], bbox[:, 4], bbox[:, 5]  # Consistent with GT bbox format!
    
    N = bbox.shape[0]
    
    # 8 corners (axis-aligned, no rotation)
    # Same corner ordering/definitions as 7/9D bbox
    # X: ±l/2, Y: ±w/2, Z: ±h/2
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)  # (N, 8)
    y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)  # (N, 8)
    z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)  # (N, 8)
    
    # Translate to world coordinates
    x_global = x_corners + cx.unsqueeze(1)
    y_global = y_corners + cy.unsqueeze(1)
    z_global = z_corners + cz.unsqueeze(1)
    
    # Combine into corners (N, 8, 3)
    corners = torch.stack([x_global, y_global, z_global], dim=-1)
    
    if single_bbox:
        corners = corners.squeeze(0)
    
    if is_numpy:
        corners = corners.numpy()
    
    return corners


def iou3d_rotated_vs_aligned(gt_bboxes_rotated, pred_bboxes_aligned):
    """
    Compute 3D IoU between rotated GT bbox (7/9D) and axis-aligned predicted bbox (6D)

    Args:
        gt_bboxes_rotated: (M, 7) or (M, 9) - GT bboxes [x,y,z,l,w,h,yaw(,pitch,roll)]
                           Coordinate system: X-forward/backward(l), Y-left/right(w), Z-up/down(h)
        pred_bboxes_aligned: (N, 6) - Predicted bboxes [cx,cy,cz,l,w,h] (axis-aligned)
                             Coordinate system: X-forward/backward(l), Y-left/right(w), Z-up/down(h)
                             Exactly consistent with the first 6 dims of GT bbox!

    Returns:
        ious: (M, N) - IoU matrix
        union: (M, N) - union volume matrix
    """
    
    # Convert to corners
    gt_corners = bbox_7d9d_to_corners(gt_bboxes_rotated)  # (M, 8, 3) - [X, Y, Z]
    pred_corners = bbox_6d_to_corners(pred_bboxes_aligned)  # (N, 8, 3) - [X, Y, Z]
    
    # box3d_iou expects coordinate system [X, Y, Z] where Y is the height axis
    # Although the comment says "up direction is negative Y", the logic requires corners[0].y > corners[4].y
    # Our coordinate system is [X, Y, Z] with Z as height, and corners[0-3].Z > corners[4-7].Z
    # We need to reorder coordinates: [X, Y, Z] -> [X, Z, Y]
    # So Z becomes the new Y (height), keeping corners[0].y > corners[4].y
    gt_corners_reorder = torch.stack([
        gt_corners[..., 0],  # X unchanged
        gt_corners[..., 2],  # new Y = Z (Z is height)
        gt_corners[..., 1],  # new Z = Y
    ], dim=-1)
    
    pred_corners_reorder = torch.stack([
        pred_corners[..., 0],  # X unchanged
        pred_corners[..., 2],  # new Y = Z
        pred_corners[..., 1],  # new Z = Y
    ], dim=-1)
    
    M = gt_corners_reorder.shape[0]
    N = pred_corners_reorder.shape[0]
    
    # Use existing box3d_iou function
    ious = torch.zeros(M, N).to(gt_corners.device)
    
    for i in range(M):
        for j in range(N):
            try:
                iou3d, iou_bev = box3d_iou(
                    gt_corners_reorder[i].cpu().numpy(), 
                    pred_corners_reorder[j].cpu().numpy()
                )
                ious[i, j] = torch.tensor(iou3d).to(gt_corners.device)
            except:
                # If computation fails, set to 0
                ious[i, j] = torch.tensor(0.0).to(gt_corners.device)
    
    # union can be approximated (for exact value, modify box3d_iou return)
    union = None
    
    return ious, union
