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
"""Shared utilities for all main scripts."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--num_target", type=int, default=256, help="Proposal number")
    parser.add_argument("--sampling", default="kps", type=str, help="Query points sampling method (kps, fps)")

    # Transformer
    parser.add_argument("--num_encoder_layers", default=3, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--self_position_embedding", default="loc_learned", type=str, help="(none, xyz_learned, loc_learned)")
    parser.add_argument("--self_attend", action="store_true")

    # Loss
    parser.add_argument("--query_points_obj_topk", default=4, type=int)
    parser.add_argument("--use_contrastive_align", action="store_true")
    parser.add_argument("--use_soft_token_loss", action="store_true")
    parser.add_argument("--detect_intermediate", action="store_true")
    parser.add_argument("--joint_det", action="store_true")

    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size during training")
    parser.add_argument("--dataset", type=str, default=["sr3d"], nargs="+", help="list of datasets to train on")
    parser.add_argument("--test_dataset", type=str, default=["sr3d"], nargs="+", )
    parser.add_argument("--data_root", default="./")
    parser.add_argument("--use_height", action="store_true", help="Use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--butd", action="store_true")
    parser.add_argument("--butd_gt", action="store_true")
    parser.add_argument("--butd_cls", action="store_true")
    parser.add_argument("--augment_det", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--optimizer", type=str, default="adamW")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument("--lr-scheduler", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--lr_decay_epochs", type=int, default=[280, 340], nargs="+", help="when to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="for step scheduler. decay rate for lr")
    parser.add_argument("--clip_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--syncbn", action="store_true")
    parser.add_argument("--warmup-epoch", type=int, default=-1)
    parser.add_argument("--warmup-multiplier", type=int, default=100)
    parser.add_argument("--flag", default=None, help="a flag to identify the experiment")

    # io
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        # default="/home/rongl/code/3eed/logs/bdetr/scanrefer/epch29_0645/ckpt_epoch_25.pth",
        help="Model checkpoint path",
    )
    parser.add_argument("--log_dir", default="log", help="Dump dir to save model checkpoint")
    parser.add_argument("--print_freq", type=int, default=10)  # batch-wise
    parser.add_argument("--save_freq", type=int, default=10)  # epoch-wise
    parser.add_argument("--val_freq", type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int, help="local rank for DistributedDataParallel")
    parser.add_argument("--ap_iou_thresholds", type=float, default=[0.25, 0.5], nargs="+", help="A list of AP IoU thresholds")
    parser.add_argument("--rng_seed", type=int, default=0, help="manual seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        # default=True,
        # default=False,
        help="try to overfit few samples",
    )
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--pp_checkpoint", default=None)
    parser.add_argument("--reduce_lr", action="store_true")

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train
    if args.eval:
        args.log_dir = args.log_dir + "_eval"

    if args.debug:
        args.num_workers = 1
        args.log_dir = args.log_dir + "_debug"

    return args


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    try:
        args.start_epoch = int(checkpoint["epoch"]) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint["model"], strict=True)
    if not args.eval and not args.reduce_lr:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    print("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint["epoch"]))

    del checkpoint
    torch.cuda.empty_cache()


def backup_python_files(self, backup_dirs, target_dir):
    """
    Backup specified python files/folders to target directory.

    Args:
        backup_dirs (list): List of directories or files to backup.
        target_dir (str): Where to store the backup.
    """
    for item in backup_dirs:
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(target_dir, item), dirs_exist_ok=True)
        elif item.endswith(".py"):
            shutil.copy2(item, target_dir)


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {"config": args, "save_path": "", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch}
        spath = os.path.join(args.log_dir, f"ckpt_epoch_{epoch}.pth")
        state["save_path"] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split("/")[-1]
        # Create log dir

        if args.flag is not None:
            args.log_dir = os.path.join(args.log_dir, ",".join(args.dataset), str(args.flag))
        else:
            args.log_dir = os.path.join(args.log_dir, ",".join(args.dataset), f"{int(time.time())}")

        os.makedirs(args.log_dir, exist_ok=True)

        # import pdb
        # pdb.set_trace()
        self.debug = args.debug

        # Create logger
        self.logger = setup_logger(output=args.log_dir, distributed_rank=dist.get_rank(), name=name)

        self.log_dir = args.log_dir
        
        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, "w") as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

        # Backup used python file
        # 主进程保存配置 + 备份代码
        if dist.get_rank() == 0:

            # 保存 config
            path = os.path.join(args.log_dir, "config.json")
            with open(path, "w") as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

            # 备份代码
            backup_files = ["main_utils.py", "prepare_data.py", "train_dist_mod.py"]
            backup_dirs = ["models", "src", "utils", "scripts"]
            backup_path = os.path.join(args.log_dir, "code_backup")
            os.makedirs(backup_path, exist_ok=True)
            self.backup_code(backup_files, backup_dirs, backup_path)
            self.logger.info(f"Code backup completed at {backup_path}")

        # import pdb

        # pdb.set_trace()
        # 只在主进程初始化 TensorBoard
        if dist.get_rank() == 0:
            tb_logdir = os.path.join(args.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_logdir)
            self.logger.info(f"TensorBoard logs at {tb_logdir}")
        else:
            self.tb_writer = None

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def backup_code(self, files, dirs, target_dir):
        def ignore_non_py_files(dir, files):
            return [f for f in files if not (f.endswith(".py") or f.endswith(".sh") or os.path.isdir(os.path.join(dir, f)))]

        # 复制单个 .py 文件
        for file in files:
            src_path = os.path.abspath(file)
            if os.path.exists(src_path) and src_path.endswith(".py"):
                shutil.copy2(src_path, target_dir)
            else:
                print(f"[Warning] File not found or not a .py file: {src_path}")

        # 复制目录下的 .py 文件
        for dir_ in dirs:
            src_path = os.path.abspath(dir_)
            dst_path = os.path.join(target_dir, os.path.basename(dir_))
            if os.path.exists(src_path):
                shutil.copytree(src_path, dst_path, ignore=ignore_non_py_files, dirs_exist_ok=True)
            else:
                print(f"[Warning] Directory not found: {src_path}")

    def get_loaders(self, args):
        """Initialize data loaders."""

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
        )
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g,
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ["boxes", "labels"]
        if args.use_contrastive_align:
            losses.append("contrastive_align")
        set_criterion = SetCriterion(matcher=matcher, losses=losses, eos_coef=0.1, temperature=0.07)
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone_net" not in n and "text_encoder" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone_net" in n and p.requires_grad], "lr": args.lr_backbone},
            {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": args.text_encoder_lr},
        ]
        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        n_data = len(train_loader.dataset)
        self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        scheduler = get_scheduler(optimizer, len(train_loader), args)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)  # , find_unused_parameters=True

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            self.logger.info("Loaded checkpoint from '{}'".format(args.checkpoint_path))

        # Just eval and end execution
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(args.start_epoch, test_loader, model, criterion, set_criterion, args)
            return

        # Training loop
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()
            self.train_one_epoch(epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args)
            self.logger.info(
                "epoch {}, total time {:.2f}, "
                "lr_base {:.5f}, lr_pointnet {:.5f}".format(epoch, (time.time() - tic), optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
            )
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:  # save model
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Test evaluation.......")
                self.evaluate_one_epoch(epoch, test_loader, model, criterion, set_criterion, args)

        # Training is over, evaluate
        save_checkpoint(args, "last", model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, "ckpt_epoch_last.pth")
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(args.max_epoch, test_loader, model, criterion, set_criterion, args)
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {"point_clouds": batch_data["point_clouds"].float(), "text": batch_data["utterances"]}

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(end_points, args.num_decoder_layers, set_criterion, query_points_obj_topk=args.query_points_obj_topk)
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    def train_one_epoch(self, epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        # Loop over batches
        for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train epoch {}".format(epoch)):

            if self.debug and batch_idx > 10:
                self.logger.info("debug mode")
                break

            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            # Forward pass
            end_points = model(inputs)

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert key not in end_points
                end_points[key] = batch_data[key]
            loss, end_points = self._compute_loss(end_points, criterion, set_criterion, args)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                stat_dict["grad_norm"] = grad_total_norm
            optimizer.step()
            scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if self.tb_writer:
                global_step = epoch * len(train_loader) + batch_idx
                for key in sorted(stat_dict.keys()):
                    if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key:
                        self.tb_writer.add_scalar(f"Train/{key}", stat_dict[key] / args.print_freq, global_step)

            if (batch_idx + 1) % args.print_freq == 0:

                # Terminal logs
                self.logger.info(f"Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ")
                self.logger.info(
                    "".join(
                        [
                            f"{key} {stat_dict[key] / args.print_freq:.4f} \t"
                            for key in sorted(stat_dict.keys())
                            if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key
                        ]
                    )
                )

                # Reset statistics
                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

    @torch.no_grad()
    def _main_eval_branch(self, epoch, batch_idx, batch_data, test_loader, model, stat_dict, criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        end_points = model(inputs)

        # Compute loss
        for key in batch_data:
            assert key not in end_points
            end_points[key] = batch_data[key]
        _, end_points = self._compute_loss(end_points, criterion, set_criterion, args)
        for key in end_points:
            if "pred_size" in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f"Eval: [{batch_idx + 1}/{len(test_loader)}]  ")
            self.logger.info(
                "".join(
                    [
                        f"{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t"
                        for key in sorted(stat_dict.keys())
                        if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key
                    ]
                )
            )

        # ====== 加入 TensorBoard 记录 loss ======
        if self.tb_writer:
            for key in sorted(stat_dict.keys()):
                if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key:
                    self.tb_writer.add_scalar(f"Eval/{key}", stat_dict[key] / (float(batch_idx + 1)), epoch * len(test_loader) + batch_idx)

        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
