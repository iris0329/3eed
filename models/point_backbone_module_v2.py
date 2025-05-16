import os
from easydict import EasyDict
import torch
import torch.nn as nn

# from ..ops.pointnet2.pointnet2_batch import pointnet2_modules
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append("/home/rongl/code/3eed/")
from ops.teed_pointnet.pointnet2_batch import pointnet2_modules
from pointnet2.pointnet2_modules import PointnetFPModule


class Point_Backbone_V2(nn.Module):

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get("AGGREGATION_MLPS", None)
        self.confidence_mlps = sa_config.get("CONFIDENCE_MLPS", None)
        self.max_translate_range = sa_config.get("MAX_TRANSLATE_RANGE", None)

        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list):  ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            if self.layer_types[k] == "SA_Layer":
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class=self.num_class,
                    )
                )

            elif self.layer_types[k] == "Vote_Layer":
                self.SA_modules.append(
                    pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k], pre_channel=channel_out_list[self.layer_inputs[k]], max_translate_range=self.max_translate_range)
                )

            channel_out_list.append(channel_out)

        self.num_point_features = channel_out

        output_dim = 288
        self.fp1 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 128, 256, output_dim])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, points):
        """
        Args:
            points:
                batch_size: [B, N, 3+C]

        """
        batch_dict = {}
        xyz, features = self._break_up_pc(points)

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], [xyz]

        li_cls_pred = None

        # down  sampling
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == "SA_Layer":
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred, sampled_idx_list = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            encoder_xyz.append(li_xyz)
            encoder_features.append(li_features)
            sa_ins_preds.append(sampled_idx_list)

        batch_dict["encoder_xyz"] = encoder_xyz
        batch_dict["encoder_features"] = encoder_features
        batch_dict["sa_ins"] = sa_ins_preds

        features = self.fp1(encoder_xyz[3], encoder_xyz[4], encoder_features[3], encoder_features[4])
        features = self.fp2(encoder_xyz[2], encoder_xyz[3], encoder_features[2], features)
        batch_dict["fp2_features"] = features  #  (B, 288, 1024)
        batch_dict["fp2_xyz"] = batch_dict["encoder_xyz"][2]  # (B, 1024, 3)
        num_seed = batch_dict["fp2_xyz"].shape[1]  # 1024
        batch_dict["fp2_inds"] = batch_dict["sa_ins"][1][:, 0:num_seed]  # indices among the entire input point clouds

        return batch_dict


def get_cfg():
    cfg = EasyDict()
    cfg.BACKBONE_3D = EasyDict()
    cfg.BACKBONE_3D.SA_CONFIG = EasyDict()

    # Sampling setting
    cfg.BACKBONE_3D.SA_CONFIG.NPOINT_LIST = [[2048], [1024], [512], [256]]  
    cfg.BACKBONE_3D.SA_CONFIG.SAMPLE_RANGE_LIST = [[-1], [-1], [-1], [-1]]
    cfg.BACKBONE_3D.SA_CONFIG.SAMPLE_METHOD_LIST = [["D-FPS"], ["D-FPS"], ["D-FPS"], ["D-FPS"]]

    # Group and Abstraction setting
    cfg.BACKBONE_3D.SA_CONFIG.RADIUS_LIST = [[0.2, 0.8], [0.8, 1.6], [1.6, 3.2], [1.6, 4.8]]
    cfg.BACKBONE_3D.SA_CONFIG.NSAMPLE_LIST = [[16, 32], [16, 32], [16, 32], [16, 32]]
    cfg.BACKBONE_3D.SA_CONFIG.MLPS = [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 128, 256], [128, 256, 256]],
        [[256, 256, 512], [256, 512, 1024]],
    ]

    cfg.BACKBONE_3D.SA_CONFIG.LAYER_TYPE = ["SA_Layer", "SA_Layer", "SA_Layer", "SA_Layer"]
    cfg.BACKBONE_3D.SA_CONFIG.DILATED_GROUP = [False, False, False, False]
    cfg.BACKBONE_3D.SA_CONFIG.AGGREGATION_MLPS = [[64], [128], [256], [512]]

    # Instance-aware setting
    cfg.BACKBONE_3D.SA_CONFIG.CONFIDENCE_MLPS = [[], [], [], [], [], []]

    cfg.BACKBONE_3D.SA_CONFIG.LAYER_INPUT = [0, 1, 2, 3, 4]
    cfg.BACKBONE_3D.SA_CONFIG.CTR_INDEX = [-1, -1, -1, -1]
    cfg.BACKBONE_3D.SA_CONFIG.MAX_TRANSLATE_RANGE = [3.0, 3.0, 2.0]
    return cfg



if __name__ == "__main__":
    cfg = get_cfg()
    model = Point_Backbone_V2(model_cfg=cfg.BACKBONE_3D, num_class=6, input_channels=3).to("cuda")
    points = torch.randn(4, 16384, 3).to("cuda")
    print(points.max(0))
    output_dict = model(points)
    print(output_dict["fp2_inds"].shape)
