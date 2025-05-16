# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Main script for language modulation."""

import os
import json
import numpy as np
import torch
import torch.distributed as dist

from main_utils import parse_option, BaseTrainTester
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator#, GroundingGTEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths
from tqdm import tqdm

import ipdb

st = ipdb.set_trace


class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        
        test_dataset = {}
        for d in args.test_dataset:
            test_dataset[d] = 1
            
        train_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=test_dataset,
            split="train", 
            use_color=args.use_color,
            use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            augment_det=args.augment_det,
            debug=args.debug,
        )
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=test_dataset,
            split="val",  
            use_color=args.use_color,
            use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            debug=args.debug,
        )
        return train_dataset, test_dataset

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,  # TODO: Update this parameter
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            self_attend=args.self_attend,
        )
        return model

    @staticmethod
    def _get_inputs(batch_data):
        return {
            "point_clouds": batch_data["point_clouds"].float(),
            "text": batch_data["utterances"],
            # "det_boxes": batch_data["all_detected_boxes"],  # TODO: Check if this is only for butd
            # "det_bbox_label_mask": batch_data["all_detected_bbox_label_mask"],  # For butd?
            # "det_class_ids": batch_data["all_detected_class_ids"],  # For butd?
        }

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """

        if args.test_dataset == "scannet":
            return self.evaluate_one_epoch_det(epoch, test_loader, model, criterion, set_criterion, args)
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)

        if args.num_decoder_layers > 0:  # true, args.num_decoder_layers is 6
            prefixes = ["last_", "proposal_"]
            prefixes = ["last_"]
            prefixes.append("proposal_")
        else:
            prefixes = ["proposal_"]  # only proposal
        prefixes += [f"{i}head_" for i in range(args.num_decoder_layers - 1)]  # [0, 1, 2, 3, 4]

        assert args.butd_cls is False, "butd_cls not implemented"
        assert args.butd is False, "butd not implemented"
        assert args.butd_gt is False, "butd_gt not implemented"

        thres = [0.25, 0.5]  # [0.25, 0.5, 0.7, 0.9]

        evaluator = GroundingEvaluator(only_root=False, thresholds=thres, topks=[1, 5, 10], prefixes=prefixes)

        # Main eval branch
        for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Eval epoch {epoch}"):
            if self.debug and batch_idx > 10:
                self.logger.info("eval debug break")
                break

            stat_dict, end_points = self._main_eval_branch(epoch, batch_idx, batch_data, test_loader, model, stat_dict, criterion, set_criterion, args)
            if evaluator is not None:
                for prefix in prefixes:  # ['last_', 'proposal_', '0head_', '1head_', '2head_', '3head_', '4head_']
                    # evaluator.evaluate(end_points, prefix)
                    evaluator.evaluate(batch_data, end_points, prefix)

        evaluator.synchronize_between_processes()
        if dist.get_rank() == 0:
            if evaluator is not None:
                return_str = evaluator.print_stats()
                self.logger.info(return_str)

        # Record accuracy in tensorboard
        if self.tb_writer is not None:
            dets = evaluator.dets
            gts = evaluator.gts

            for t in thres:
                acc_bbf = dets[("total_acc", t, "bbf")] / gts[("total_acc", t, "bbf")]
                self.tb_writer.add_scalar(f"Eval/acc@{t}", acc_bbf, epoch)
        
        miou = evaluator.dets["iou"] / evaluator.dets["num_iou"]
        self.logger.info("mIoU: {:.4f}".format(miou))
        
        preds = evaluator.prediction_records 
        for i, pred in enumerate(preds):
            save_id = pred["id"]
            save_path = os.path.join(self.log_dir, f"predictions/{save_id}")
            os.makedirs(save_path, exist_ok=True)
            save_json = os.path.join(save_path, "prediction.json")
            with open(save_json, "w") as f:
                json.dump(pred, f, indent=4)
        print("Saved predictions at ", self.log_dir)

        return None

    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader, model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        import pdb

        pdb.set_trace()

        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            "remove_empty_box": False,
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.0,
            "dataset_config": dataset_config,
            "hungarian_loss": True,
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ["last_", "proposal_"]
            prefixes += [f"{i}head_" for i in range(args.num_decoder_layers - 1)]
        else:
            prefixes = ["proposal_"]  # only proposal
        prefixes = ["last_"]
        ap_calculator_list = [APCalculator(iou_thresh, dataset_config.class2type) for iou_thresh in args.ap_iou_thresholds]
        mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in args.ap_iou_thresholds]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        wordidx = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18])
        tokenidx = np.array([1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23, 25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45])
        for batch_idx, batch_data in enumerate(test_loader):
            stat_dict, end_points = self._main_eval_branch(epoch, batch_idx, batch_data, test_loader, model, stat_dict, criterion, set_criterion, args)
            # contrast
            proj_tokens = end_points["proj_tokens"]  # (B, tokens, 64)
            proj_queries = end_points["last_proj_queries"]  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, : sem_scores_.size(1), : sem_scores_.size(2)] = sem_scores_
            end_points["last_sem_cls_scores"] = sem_scores
            # end contrast
            sem_cls = torch.zeros_like(end_points["last_sem_cls_scores"])[..., :19]
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points["last_sem_cls_scores"][..., t]
            end_points["last_sem_cls_scores"] = sem_cls

            # Parse predictions
            # for prefix in prefixes:
            prefix = "last_"
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix, size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT, size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = "last_"
        for batch_pred_map_cls, batch_gt_map_cls in zip(batch_pred_map_cls_dict[prefix], batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info("=====================>" f"{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}" "<=====================")
            for key in metrics_dict:
                self.logger.info(f"{key} {metrics_dict[key]}")
            if prefix == "last_" and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict["mAP"]
            mAPs[i][1][prefix] = metrics_dict["mAP"]
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(f"IoU[{mAP[0]}]:\t" + "".join([f"{key}: {mAP[1][key]:.4f} \t" for key in sorted(mAP[1].keys())]))

        return None


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    opt = parse_option()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Changed this to enable debugging in PyCharm
    print(f"\n\nlocal_rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)

    ##########################################

    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # opt = parse_option()
    # torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # train_tester = TrainTester(opt)
    # ckpt_path = train_tester.main(opt)
