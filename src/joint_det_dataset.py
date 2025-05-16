# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

# ==================== Imports ====================
from collections import defaultdict
import json
import multiprocessing as mp
import os
import shutil
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
import open3d as o3d
from tqdm import tqdm

from utils.align_3eed import convert_boxes_from_n_to_vir, convert_points_to_virtual
from utils.visual import create_axis_aligned_bbox_with_cylindrical_edges, save_as_ply
from utils.transform_waymo import transform_to_front_view
from utils.pcds_in_bbox import get_points_in_bbox
from ops.teed_pointnet.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

import pickle
# ==================== Constants ====================
MAX_NUM_OBJ = 132

# ==================== Dataset Class ====================
class Joint3DDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(
        self,
        dataset_dict={"3eed": 1},
        test_dataset={"3eed": 1},
        split="train",
        overfit=False,
        data_path="./",
        use_color=False,
        use_height=False,
        use_multiview=False,
        detect_intermediate=False,
        butd=False,
        butd_gt=False,
        butd_cls=False,
        augment_det=False,
        debug=False,
    ):
        """Initialize dataset (here for ReferIt3D utterances)."""
        # Basic configuration
        self.debug = debug
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == "train"
        self.use_multiview = use_multiview
        self.data_path = data_path
        self.visualize = False
        self.butd = butd
        self.butd_gt = butd_gt
        self.butd_cls = butd_cls
        self.joint_det = "scannet" in dataset_dict and len(dataset_dict.keys()) > 1 and self.split == "train"
        self.augment_det = augment_det

        # Initialize tokenizer and other utilities
        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        self.tokenizer = RobertaTokenizerFast.from_pretrained("./data/roberta_base/")
        
        # Load classification results if available
        if os.path.exists("data/cls_results.json"):
            with open("data/cls_results.json") as fid:
                self.cls_results = json.load(fid)

        # Load annotations
        self.annos = []
        
        if self.split == "train":
            for dset in dataset_dict.keys():
                _annos = self.load_annos(dset)
                self.annos += _annos
        else:
            for dset in test_dataset.keys():
                _annos = self.load_annos(dset)
                self.annos += _annos
                
    # ==================== Data Loading Methods ====================
    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            "waymo": lambda: self.load_3eed_annos(dataset="waymo"),
            "m3ed-drone": lambda: self.load_3eed_annos(dataset="m3ed-drone"),
            "m3ed-quad": lambda: self.load_3eed_annos(dataset="m3ed-quad"),
            "waymo-multi": lambda: self.waymo_multi_annos(dataset="waymo-multi"),
        }
        annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos


    def waymo_multi_annos(self, dataset="waymo-multi"):
        def refine_frame_key(frame_key):
            frame_id, lidar_id = frame_key.split("_")[0], frame_key.split("_")[1]
            return f"{str(frame_id).zfill(4)}_{lidar_id}"

        """Load annotations of 3eed."""
        annos = []
        split = "train" if self.split == "train" else "val"

        class2id_dict = {
            "car": 0,
            "pedestrian": 1,
            "bus": 2,
            "othervehicle": 3,
            "truck": 4,
            "cyclist": 5,
        }

        synonyms_dict = {
            "car": ["car", "vehicle", "sedan", "van", "coupe", "automobile", "convertible", "hatchback", "SUV"],
            "truck": ["truck", "lorry", "freight", "pickup truck", "delivery truck", "cargo truck", "semi-truck"],
            "bus": ["bus", "coach", "minibus", "shuttle", "school bus", "public transport"],
            "othervehicle": ["vehicle", "van", "pickup", "minivan", "jeep", "SUV", "tractor", "trailer"],
            "pedestrian": ["pedestrian", "person", "man", "woman", "people", "child", "boy", "girl", "adult", "passerby", "walker"],
            "cyclist": ["cyclist", "biker", "bike rider", "rider"],
        }

        data_file = os.path.join(self.data_path,  f"waymo_multi_{split}_info.pkl")
        print(f"Loading {data_file}")
        assert os.path.exists(data_file), f"file not exist: {data_file}"

        with open(data_file, "rb") as f:
            frame_infos = pickle.load(f)

        for frame_info in frame_infos:
            utterance = frame_info["caption"]
            caption = " ".join(utterance.replace(",", " ,").split())
            caption = " " + caption + " "
            try:
                seg_name = frame_info["segment_name"]
            except:
                print("*" * 20)
                print(frame_info)

            frame_key = frame_info["frame_name"]
            frame_path = os.path.join(self.data_path, "waymo", seg_name, refine_frame_key(frame_key))

            # lidar path
            lidar_path = os.path.join(frame_path, "lidar.npy")
            # image path
            image_path = os.path.join(frame_path, "image.jpg")
            # bbox_3d & 2d
            box_info = defaultdict(list)
            box_id = 1
            while True:
                if f"bbox3d_obj_{box_id}" in frame_info:
                    box_info["bbox3d"].append(np.array(frame_info[f"bbox3d_obj_{box_id}"]))
                    box_info["bbox2d"].append(np.array(frame_info[f"bbox2d_obj_{box_id}"]))
                    box_info["class_names"].append(frame_info[f"class_obj_{box_id}"].lower())
                    box_info["class_id"].append(class2id_dict[frame_info[f"class_obj_{box_id}"].lower()])
                else:
                    break
                box_id += 1
            candidate_words = []
            class_names = box_info["class_names"]
            unique_class_names = set(class_names)
            class_names = list(unique_class_names)
            for class_name in class_names:
                candidate_words += synonyms_dict.get(class_name, [class_name])

            positions = []
            for word in candidate_words:
                matches = list(re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE))
                positions.extend([(match.start(), match.end(), word) for match in matches])

            # print(caption[positions[0][0] : positions[0][1]])
            all_positive = []
            if len(positions) > 0:
                for i in range(len(positions)):
                    tokens_positive = torch.tensor([positions[i][0], positions[i][1]], dtype=torch.long)  #
                    matched_cls = positions[i][2]  # e.g. van
                    all_positive.append(tokens_positive)
            else:
                # 记录哪些匹配失败
                with open("log.txt", "a") as f:
                    f.write(f"match failed: {utterance}\n")
                    f.write(f"class name: {class_names}\n\n")
                continue
            # caption
            tokenized = self.tokenizer.batch_encode_plus([" ".join(utterance.replace(",", " ,").split())], padding="longest", return_tensors="pt")
            gt_map = get_positive_map(tokenized, all_positive)  # MARK 多目标的 positive map
            anno_dict = {
                "scan_id": frame_key,
                "utterance": utterance,
                "pred_pos_map": gt_map,
                "dataset": dataset,  # "waymo-multi"
                "pcd_path": lidar_path,
                "image_path": image_path,
                "boxes_info": box_info,
            }
            annos.append(anno_dict)

        print(f"Loaded {len(annos)} annotations from {split}.")
        return annos


    def load_3eed_annos(self, dataset="waymo"):
        """Load annotations of 3eed."""

        split = "train" if self.split == "train" else "val"

        class2id_dict = {
            "car": 0,
            "pedestrian": 1,
            "bus": 2,
            "othervehicle": 3,
            "truck": 4,
            "cyclist": 5,
        }

        if dataset == "waymo":
            # Define synonym mapping table for Waymo dataset
            synonyms_dict = {
                "car": ["car", "vehicle", "sedan", "van", "coupe", "automobile", "convertible", "hatchback", "SUV"],
                "truck": ["truck", "lorry", "freight", "pickup truck", "delivery truck", "cargo truck", "semi-truck"],
                "bus": ["bus", "coach", "minibus", "shuttle", "school bus", "public transport"],
                "othervehicle": ["vehicle", "van", "pickup", "minivan", "jeep", "SUV", "tractor", "trailer"],
                "pedestrian": ["pedestrian", "person", "man", "woman", "people", "child", "boy", "girl", "adult", "passerby", "walker"],
                "cyclist": ["cyclist", "biker", "bike rider", "rider"],
            }
        else:
            # Define synonym mapping table for other datasets
            synonyms_dict = {
                "car": [
                    "car", "vehicle", "sedan", "van", "coupe", "automobile", "convertible", "hatchback", "SUV",
                    "truck", "bus", "coach", "minibus", "shuttle", "school bus", "public transport",
                    "lorry", "freight", "pickup truck", "delivery truck", "cargo truck", "semi-truck",
                    "bus", "coach", "minibus", "car", "shuttle", "school bus", "public transport",
                    "vehicle", "van", "pickup", "minivan", "jeep", "SUV", "tractor", "trailer",
                ],
                "pedestrian": ["pedestrian", "person", "man", "woman", "people", "child", "boy", "girl", "adult", "passerby", "walker", "cyclist", "biker", "bike rider", "rider"],
            }

        # Set data path based on dataset type
        if dataset == "waymo":
            data_path = os.path.join(self.data_path, "3eed", "waymo")
        elif dataset == "m3ed-drone":
            data_path = os.path.join(self.data_path, "3eed", "M3ED-Drone")
        elif dataset == "m3ed-quad":
            data_path = os.path.join(self.data_path, "3eed", "M3ED-Quadruped")
        else:
            raise NotImplementedError

        frames_names = []

        # Load sequence list from split file
        with open(f"data/splits/{dataset}_{split}.txt") as f:
            sequence_list = [line.rstrip() for line in f]

        for sequence in sequence_list:
            # List all frame directories in the sequence
            frame_list = [f for f in os.listdir(os.path.join(data_path, sequence)) if os.path.isdir(os.path.join(data_path, sequence, f))]
            for frame in frame_list:
                frames_names.append(os.path.join(sequence, frame))

        annos = []
        class_set = set()  # Store unique classes for statistics
        for frame_name in tqdm(frames_names):
            frame_path = os.path.join(data_path, frame_name)  # e.g. waymo/scene-0000/0000_0
            image_path = os.path.join(frame_path, "image.jpg")
            lidar_path = os.path.join(frame_path, "lidar.npy" if dataset == "waymo" else "lidar.bin")
            meta_path = os.path.join(frame_path, "meta_info.json")
            if not os.path.exists(meta_path):
                continue

            # Load metadata
            with open(meta_path) as f:
                meta = json.load(f)

            # Process each object in ground_info
            for obj_idx, obj in enumerate(meta["ground_info"]):
                class_set.add(obj["class"].lower())  # Add class to set for statistics

                # Get positive map
                try:
                    utterance = obj["caption"]
                except:
                    # Log error frame information
                    with open(f"error_frames_{dataset}.txt", "a") as log_file:
                        log_file.write(f"{frame_name} | obj: {obj}\n\n")
                    continue

                # Process utterance based on dataset type
                if dataset == "waymo":
                    utterance = utterance
                elif dataset == "m3ed-drone":
                    utterance = utterance.split("Summary:")[-1].strip()
                elif dataset == "m3ed-quad":
                    utterance = utterance.split("Summary:")[-1].strip()

                cat_names = obj["class"].lower()

                caption = " ".join(utterance.replace(",", " ,").split())
                caption = " " + caption + " "

                # Get current class's synonym list
                candidate_words = synonyms_dict.get(cat_names, [cat_names])

                # Find all candidate positions in utterance
                positions = []
                for word in candidate_words:
                    matches = list(re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE))
                    positions.extend([(match.start(), match.end(), word) for match in matches])

                if len(positions) > 0:
                    tokens_positive = torch.tensor([positions[0][0], positions[0][1]], dtype=torch.long)
                    matched_cls = positions[0][2]  # e.g. van
                else:
                    # Log failed matches
                    with open("log.txt", "a") as f:
                        f.write(f"match failed: {utterance}\n")
                        f.write(f"class name: {cat_names}\n\n")
                    continue

                tokenized = self.tokenizer.batch_encode_plus([" ".join(utterance.replace(",", " ,").split())], padding="longest", return_tensors="pt")
                gt_map = get_positive_map(tokenized, [tokens_positive])

                bbox_3d = obj["bbox_3d"]
                annos.append(
                    {
                        "scan_id": frame_name,
                        "target_id": class2id_dict[obj["class"].lower()],
                        "target": obj["class"].lower(),
                        "utterance": utterance,
                        "pred_pos_map": gt_map,  # TODO: Process span using VLM
                        "meta_path": meta_path,
                        "dataset": dataset,
                        "pcd_path": lidar_path,
                        "image_path": image_path,
                        "gt_bbox": bbox_3d,
                        "bbox_2d": obj["bbox_2d_proj"],
                        "pose": meta["pose"],
                    }
                )

        print(f"Loaded {len(annos)} annotations from {split}.")
        return annos

    # ==================== Data Processing Methods ====================
    def _get_3eed_pcd(self, anno):
        """Process point cloud data."""
        pcd_path = anno["pcd_path"]
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4) if pcd_path.endswith(".bin") else np.load(pcd_path)

        N = pcd.shape[0]
        TARGET_NUM_POINTS = 16384 

        if N >= TARGET_NUM_POINTS:  # Random downsampling
            indices = np.random.choice(N, TARGET_NUM_POINTS, replace=False)
        else:  # Random repeat sampling, fill up
            indices = np.random.choice(N, TARGET_NUM_POINTS, replace=True)

        pcd = pcd[indices]  # shape: (50000, 5)
        xyz = pcd[:, 0:3]

        reflectance = pcd[:, 3].reshape(-1, 1)  # shape: (n,)

        if anno["dataset"] == "waymo":
            # reflectance_3d = np.tanh(np.concatenate([pcd[:, 3:5], pcd[:, 3].reshape(-1, 1)], axis=1))  # (n_p, 3)
            reflectance = np.tanh(reflectance)

        elif anno["dataset"] == "m3ed-quad":
            # pass
            xyz, pose = convert_points_to_virtual(xyz, pose=np.asarray(anno["pose"]), drone=False)
            anno["pose"] = pose

        elif anno["dataset"] == "m3ed-drone":
            xyz[:, 2] += 1.8  # NOTE m3ed-drone dataset's z coordinate needs to be subtracted by 1.8 three platforms can be horizontally aligned
            # pass
            xyz, pose = convert_points_to_virtual(xyz, pose=np.asarray(anno["pose"]), drone=True)
            anno["pose"] = pose

        if self.visualize:
            save_pcd_name = f"{self.vis_save_dir}/{anno['scan_id']}_{anno['dataset']}_pcd.ply"
            save_dir = os.path.dirname(save_pcd_name)
            os.makedirs(save_dir, exist_ok=True)
            # Copy image cp to this path
            image_path = anno["image_path"]
            meta_path = anno["meta_path"]
            image_name = os.path.basename(image_path)
            meta_name = os.path.basename(meta_path)
            shutil.copyfile(image_path, save_pcd_name.replace("_pcd.ply", ".jpg"))
            shutil.copyfile(meta_path, save_pcd_name.replace("_pcd.ply", ".json"))
            save_as_ply(xyz, reflectance, save_path=f"{self.vis_save_dir}/{anno['scan_id']}_{anno['dataset']}_pcd.ply")

        reflectance = reflectance - 0.5  # self.mean_rgb[0] # from [0, 1] to [-1, 1]
        point_cloud = np.concatenate([xyz, reflectance], axis=1)

        return xyz  # point_cloud  

    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = " ".join(anno["utterance"].replace(",", " ,").split())
        caption = " " + caption + " "
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno["target"], list):
            cat_names = anno["target"]
        else:
            cat_names = [anno["target"]]
        if self.detect_intermediate:
            cat_names += anno["anchors"]
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(" " + cat_name + " ")
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(" " + cat_name)
                len_ = len(caption[start_span + 1 :].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != " ":
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != " ":
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus([" ".join(anno["utterance"].replace(",", " ,").split())], padding="longest", return_tensors="pt")
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[: len(cat_names)])
        positive_map[: len(cat_names)] = gt_map
        return tokens_positive, positive_map

    def _get_3eed_target_boxes(self, anno, xyz):
        """Return gt boxes to detect."""

        tids = [anno["target_id"]]  # waymo doesn't need to predict id

        # Generate instance label, default -1 (unmarked), if 3D point belongs to a target object, fill in target object ID
        xyz = xyz[:, :3]
        point_instance_label = -np.ones(len(xyz))
        # Find points inside bbox and mark as 0

        # Generate axis_align_bbox for 3D object
        bbox = np.array(anno["gt_bbox"])

        mask = get_points_in_bbox(xyz, bbox[:3], bbox[3:6], bbox[6])
        point_instance_label[mask] = 0

        if anno["dataset"] == "m3ed-drone":
            bbox = bbox.reshape(-1)
            bbox[2] += 1.8
            bbox = convert_boxes_from_n_to_vir(bbox, anno["pose"], drone=True)
        elif anno["dataset"] == "m3ed-quad":
            bbox = convert_boxes_from_n_to_vir(bbox, anno["pose"], drone=False)

        # Generate axis_align_bbox for debug
        if self.visualize:
            bbox_mesh = create_axis_aligned_bbox_with_cylindrical_edges(bbox, radius=0.02, color_rgb=(0, 180, 139))
            o3d.io.write_triangle_mesh(f"{self.vis_save_dir}/{anno['scan_id']}_{anno['dataset']}_bbox.ply", bbox_mesh)

        bbox = bbox.reshape(-1)
        bbox = bbox[:7]

        # Generate axis_align_bbox for 3D object
        bboxes = np.zeros((MAX_NUM_OBJ, 7))
        bboxes[: len(tids)] = bbox[:7]  # shape: (N, 6)

        bboxes[len(tids) :, :3] = 1000  # Fill bbox for non-target objects # First len(tids) are real targets, rest are padding
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[: len(tids)] = 1  # Mark which bboxes are valid

        # Visualization check, points in mask are red, other points are gray, save as .ply using open3d, can be viewed with meshlab
        # if anno["dataset"] == "3eed_debug":
            # visualize_3eed_pointcloud_with_bbox(xyz, point_instance_label, save_path=f"vis_3eed/{anno['scan_id']}.ply")
        return bboxes, box_label_mask, point_instance_label


    def _get_waymo_multi_target_boxes(self, anno, xyz):
        """Return gt boxes to detect."""
        boxes_info = anno["boxes_info"]
        tids = boxes_info["class_id"]
        gt_bbox = np.stack(boxes_info["bbox3d"], axis=0).astype(np.float32)  # shape: (N, 7)
        xyz = xyz[:, :3]
        point_instance_label = -np.ones(len(xyz)) # 找到 bbox 之内的点，并标记为 0

        point_indices = points_in_boxes_cpu(torch.from_numpy(xyz), torch.from_numpy(gt_bbox)).numpy()
        for i in range(gt_bbox.shape[0]):
            fg_mask = point_indices[i] > 0
            # point_instance_label[fg_mask] = i
            point_instance_label[fg_mask] = 0  

        bboxes = np.zeros((MAX_NUM_OBJ, 7))
        bboxes[: len(tids)] = gt_bbox[:, :7]  # shape: (N, 6)

        bboxes[len(tids) :, :3] = 1000  # 填充 无目标的 bbox # 前 len(tids) 个是真实目标，其余为 padding。
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[: len(tids)] = 1  # 标记 哪些 bbox 是有效的

        return bboxes, box_label_mask, point_instance_label

    # ==================== Data Augmentation Methods ====================
    def _augment(self, pc, color, rotate):
        """Apply data augmentation to point cloud."""
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90 * np.random.randint(0, 4) + 10 * np.random.rand() - 5
            # Flipping along the YZ plane
            augmentations["yz_flip"] = np.random.random() > 0.5
            if augmentations["yz_flip"]:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations["xz_flip"] = np.random.random() > 0.5
            if augmentations["xz_flip"]:
                pc[:, 1] = -pc[:, 1]
        else:
            theta_z = (2 * np.random.rand() - 1) * 5
        augmentations["theta_z"] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2 * np.random.rand() - 1) * 2.5
        augmentations["theta_x"] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2 * np.random.rand() - 1) * 2.5
        augmentations["theta_y"] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations["noise"] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations["shift"] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations["shift"]

        # Scale
        augmentations["scale"] = 0.98 + 0.04 * np.random.random()
        pc[:, :3] *= augmentations["scale"]

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04 * np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations

    def aug_points(
        self,
        xyz: np.array,
        if_flip: bool = False,
        if_scale: bool = False,
        scale_axis: str = "xyz",
        scale_range: list = [0.9, 1.1],
        if_jitter: bool = False,
        if_rotate: bool = False,
        if_tta: bool = False,
        num_vote: int = 0,
    ):
        """Apply various augmentations to points."""
        # aug (random rotate)
        if if_rotate:
            if if_tta:
                angle_vec = [0, 1, -1, 2, -2, 6, -6, 7, -7, 8]
                assert len(angle_vec) == 10
                angle_vec_new = [cnt * np.pi / 8.0 for cnt in angle_vec]
                theta = angle_vec_new[num_vote]
            else:
                theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            xyz = np.dot(xyz, rot_mat)

        # aug (random scale)
        if if_scale:
            # scale_range = [0.95, 1.05]
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            xyz = xyz * scale_factor

        # aug (random flip)
        if if_flip:
            if if_tta:
                flip_type = num_vote
            else:
                flip_type = np.random.choice(4, 1)

            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        # aug (random jitter)
        if if_jitter:
            noise_translate = np.array(
                [
                    np.random.normal(0, 0.1, 1),
                    np.random.normal(0, 0.1, 1),
                    np.random.normal(0, 0.1, 1),
                ]
            ).T
            xyz += noise_translate

        return xyz

    # ==================== Dataset Interface Methods ====================
    def __getitem__(self, index):
        """Get current batch for input index."""

        # Read annotation
        anno = self.annos[index]
        
        if anno["dataset"] == "waymo-multi":
            return self.getitem_waymo_multi(index)

        if self.debug:
            index = 0

        self.random_utt = False

        # Point cloud representation
        point_cloud = self._get_3eed_pcd(anno)
        gt_bboxes, box_label_mask, point_instance_label = self._get_3eed_target_boxes(anno, point_cloud)

        if anno["dataset"] == "waymo":
            lidar_id = int(anno["scan_id"].split("_")[-1])
            xyz = point_cloud[:, :3]
            WAYMO_VIEWS = ["F", "FL", "FR", "SL", "SR"]
            xyz, target_box = transform_to_front_view(xyz, gt_bboxes[0][None, :], WAYMO_VIEWS[lidar_id])
            point_cloud[:, :3] = xyz
            gt_bboxes[0] = target_box[0]

        positive_map = np.zeros((MAX_NUM_OBJ, 256))  #  1, 256
        positive_map_ = np.array(anno["pred_pos_map"]).reshape(-1, 256)
        positive_map[: len(positive_map_)] = positive_map_

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)  # 132

        ret_dict = {
            "box_label_mask": box_label_mask.astype(np.float32),  # NOTE Used in loss calculation 
            "center_label": gt_bboxes[:, :3].astype(np.float32),  # xyz
            "sem_cls_label": _labels.astype(np.int64),  # NOTE Used in loss calculation 
            "size_gts": gt_bboxes[:, 3:6].astype(np.float32),  # NOTE w h d
            "gt_bbox": gt_bboxes[0].astype(np.float32),
            'meta_path': anno['meta_path'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (" ".join(anno["utterance"].replace(",", " ,").split()) + " . not mentioned"),
            "positive_map": positive_map.astype(np.float32),
            "point_instance_label": point_instance_label.astype(np.int64),  # NOTE Used in loss calculation 
            "is_view_dep": self._is_view_dep(anno["utterance"]), 
            "is_hard": False, 
            "is_unique": False,  
        }

        return ret_dict

    def getitem_waymo_multi(self, index):
        anno = self.annos[index]
        self.random_utt = False
        anno["pcd_path"] = anno["pcd_path"].replace('data/', 'data/3eed/')
        anno["image_path"] = anno["image_path"].replace('data/', 'data/3eed/')
        point_cloud = self._get_3eed_pcd(anno)
        gt_bboxes, box_label_mask, point_instance_label = self._get_waymo_multi_target_boxes(anno, point_cloud)
        positive_map = np.zeros((MAX_NUM_OBJ, 256))  #  1, 256
        positive_map_ = np.array(anno["pred_pos_map"]).reshape(-1, 256)
        positive_map[: len(positive_map_)] = positive_map_
        _labels = np.zeros(MAX_NUM_OBJ)  # 132
        ret_dict = {
            "box_label_mask": box_label_mask.astype(np.float32), 
            "center_label": gt_bboxes[:, :3].astype(np.float32), 
            "sem_cls_label": _labels,  # NOTE 计算 loss 的时候用到 
            "size_gts": gt_bboxes[:, 3:6].astype(np.float32),  
            "gt_bboxes": gt_bboxes.astype(np.float32),
            "class_ids": ",".join(map(str, anno["boxes_info"]["class_id"])),  # e.g., "1,3,5"
            "utterance": anno["utterance"],
            "meta_path": anno["image_path"],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (" ".join(anno["utterance"].replace(",", " ,").split()) + " . not mentioned"),
            "positive_map": positive_map.astype(np.float32),
            "point_instance_label": point_instance_label.astype(np.int64),  
            "is_view_dep": self._is_view_dep(anno["utterance"]), 
            "is_hard":  False,  
            "is_unique":False
        }

        return ret_dict
    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = ["front", "behind", "back", "left", "right", "facing", "leftmost", "rightmost", "looking", "across"]
        words = set(utterance.split())
        return any(rel in words for rel in rels)

# ==================== Utility Functions ====================
def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos : end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()

def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[1.0, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]), pc.T).T

def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1.0, 0], [-np.sin(theta), 0, np.cos(theta)]]), pc.T).T

def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]]), pc.T).T

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack(
        (
            np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
            np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
            np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
            np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
            np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
            np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
            np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
            np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1),
        ),
        axis=1,
    )
