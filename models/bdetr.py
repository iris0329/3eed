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

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .point_backbone_module_v2 import Point_Backbone_V2, get_cfg
from .backbone_module import Pointnet2Backbone
from .modules import PointsObjClsModule, GeneralSamplingModule, ClsAgnosticPredictHead, PositionEmbeddingLearned
from .encoder_decoder_layers import BiEncoder, BiEncoderLayer, BiDecoderLayer


class BeaUTyDETR(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(
        self,
        num_class=256,
        num_obj_class=485,
        input_feature_dim=3,
        num_queries=256,
        num_decoder_layers=6,
        self_position_embedding="loc_learned",
        contrastive_align_loss=True,
        d_model=288,
        butd=True,
        pointnet_ckpt=None,
        self_attend=True,
    ):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd

        # Visual encoder
        # self.backbone_net = Pointnet2Backbone(input_feature_dim=0, width=1)
        # if input_feature_dim == 3 and pointnet_ckpt is not None:
        # self.backbone_net.load_state_dict(torch.load(pointnet_ckpt), strict=False)
        cfg = get_cfg()
        self.backbone_net = Point_Backbone_V2(model_cfg=cfg.BACKBONE_3D, num_class=num_class, input_channels=3)

        # Text Encoder
        t_type = "./data/roberta_base/"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, d_model), nn.LayerNorm(d_model, eps=1e-12), nn.Dropout(0.1))

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load("data/class_embeddings3d.npy", allow_pickle=True))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu", n_heads=8, dim_feedforward=256, self_attend_lang=self_attend, self_attend_vis=self_attend, use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(num_class, 1, num_queries, d_model, objectness=False, heading=False, compute_sem_scores=True)

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(
                BiDecoderLayer(d_model, n_heads=8, dim_feedforward=256, dropout=0.1, activation="relu", self_position_embedding=self_position_embedding, butd=self.butd)
            )

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(num_class, 1, num_queries, d_model, objectness=False, heading=False, compute_sem_scores=True))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 64))
            self.contrastive_align_projection_text = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 64))

        # Init
        self.init_bn_momentum()

    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # Visual encoder
        end_points = self.backbone_net(inputs["point_clouds"])
        end_points["seed_inds"] = end_points["fp2_inds"]
        end_points["seed_xyz"] = end_points["fp2_xyz"]
        end_points["seed_features"] = end_points["fp2_features"]
        # Text encoder
        tokenized = self.tokenizer.batch_encode_plus(inputs["text"], padding="longest", return_tensors="pt").to(inputs["point_clouds"].device)
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        end_points["text_feats"] = text_feats
        end_points["text_attention_mask"] = text_attention_mask
        end_points["tokenized"] = tokenized
        return end_points

    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)
        end_points["seeds_obj_cls_logits"] = points_obj_cls_logits
        sample_inds = torch.topk(torch.sigmoid(points_obj_cls_logits).squeeze(1), self.num_queries)[1].int()
        xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
        end_points["query_points_xyz"] = xyz  # (B, V, 3)
        end_points["query_points_feature"] = features  # (B, F, V)
        end_points["query_points_sample_inds"] = sample_inds  # (B, V)
        return end_points

    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # Within-modality encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points["fp2_xyz"]  # (B, points, 3)
        points_features = end_points["fp2_features"]  # (B, F, points)
        text_feats = end_points["text_feats"]  # (B, L, F)
        text_padding_mask = end_points["text_attention_mask"]  # (B, L)

        # Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs["det_bbox_label_mask"]
            detected_feats = (
                torch.cat([self.box_embeddings(inputs["det_boxes"]), self.class_embeddings(self.butd_class_embeddings(inputs["det_class_ids"])).transpose(1, 2)], 1)  # 92.5, 84.9
                .transpose(1, 2)
                .contiguous()
            )
        else:
            detected_mask = None
            detected_feats = None

        # Cross-modality encoding
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(len(points_xyz), points_xyz.size(1)).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask,
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()  # (B, F, points)
        end_points["text_memory"] = text_feats
        end_points["seed_features"] = points_features
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(self.contrastive_align_projection_text(text_feats), p=2, dim=-1)
            end_points["proj_tokens"] = proj_tokens  # MARK 这个用来计算contrastive loss

        # Query Points Generation
        end_points = self._generate_queries(points_xyz, points_features, end_points)
        cluster_feature = end_points["query_points_feature"]  # (B, F, V)
        cluster_xyz = end_points["query_points_xyz"]  # (B, V, 3)
        query = self.decoder_query_proj(cluster_feature)
        query = query.transpose(1, 2).contiguous()  # (B, V, F)
        if self.contrastive_align_loss:
            end_points["proposal_proj_queries"] = F.normalize(self.contrastive_align_projection_image(query), p=2, dim=-1)

        # Proposals (one for each query)
        proposal_center, proposal_size = self.proposal_head(cluster_feature, base_xyz=cluster_xyz, end_points=end_points, prefix="proposal_")  # TODO read code
        base_xyz = proposal_center.detach().clone()  # (B, V, 3)
        base_size = proposal_size.detach().clone()  # (B, V, 3)
        query_mask = None

        # Decoder
        for i in range(self.num_decoder_layers):
            prefix = "last_" if i == self.num_decoder_layers - 1 else f"{i}head_"

            # Position Embedding for Self-Attention
            if self.self_position_embedding == "none":
                query_pos = None
            elif self.self_position_embedding == "xyz_learned":
                query_pos = base_xyz
            elif self.self_position_embedding == "loc_learned":
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # Transformer Decoder Layer
            query = self.decoder[i](
                query,
                points_features.transpose(1, 2).contiguous(),
                text_feats,
                query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(detected_feats if self.butd else None),
                detected_mask=detected_mask if self.butd else None,
            )  # (B, V, F)

            if self.contrastive_align_loss:
                end_points[f"{prefix}proj_queries"] = F.normalize(self.contrastive_align_projection_image(query), p=2, dim=-1)  # MARK 这个用来计算contrastive loss

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query.transpose(1, 2).contiguous(), base_xyz=cluster_xyz, end_points=end_points, prefix=prefix)  # (B, F, V)
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
