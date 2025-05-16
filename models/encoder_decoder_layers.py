# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Encoder-decoder transformer layers for self/cross attention."""

from copy import deepcopy

import torch
from torch import nn


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class CrossAttentionLayer(nn.Module):
    """Cross-attention between language and vision."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8,
                 dim_feedforward=256, use_butd_enc_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.use_butd_enc_attn = use_butd_enc_attn

        # Cross attention from lang to vision
        self.cross_lv = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_lv = nn.Dropout(dropout)
        self.norm_lv = nn.LayerNorm(d_model)
        self.ffn_lv = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm_lv2 = nn.LayerNorm(d_model)

        # Cross attention from vision to lang
        self.cross_vl = deepcopy(self.cross_lv)
        self.dropout_vl = nn.Dropout(dropout)
        self.norm_vl = nn.LayerNorm(d_model)
        self.ffn_vl = deepcopy(self.ffn_lv)
        self.norm_vl2 = nn.LayerNorm(d_model)

        if use_butd_enc_attn:
            self.cross_d = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

    def forward(self, vis_feats, vis_key_padding_mask, text_feats,
                text_key_padding_mask, pos_feats,
                detected_feats=None, detected_mask=None):
        """Forward pass, vis/pos_feats (B, V, F), lang_feats (B, L, F)."""
        # produce key, query, value for image
        qv = kv = vv = vis_feats
        qv = qv + pos_feats  # add pos. feats only on query

        # produce key, query, value for text
        qt = kt = vt = text_feats

        # cross attend language to vision
        text_feats2 = self.cross_lv(
            query=qt.transpose(0, 1),
            key=kv.transpose(0, 1),
            value=vv.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=vis_key_padding_mask  # (B, V)
        )[0].transpose(0, 1)
        text_feats = text_feats + self.dropout_lv(text_feats2)
        text_feats = self.norm_lv(text_feats)
        text_feats = self.norm_lv2(text_feats + self.ffn_lv(text_feats))

        # cross attend vision to language
        vis_feats2 = self.cross_vl(
            query=qv.transpose(0, 1),
            key=kt.transpose(0, 1),
            value=vt.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=text_key_padding_mask  # (B, L)
        )[0].transpose(0, 1)
        vis_feats = vis_feats + self.dropout_vl(vis_feats2)
        vis_feats = self.norm_vl(vis_feats)

        # cross attend vision to boxes
        if detected_feats is not None and self.use_butd_enc_attn:
            vis_feats2 = self.cross_d(
                query=vis_feats.transpose(0, 1),
                key=detected_feats.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=detected_mask
            )[0].transpose(0, 1)
            vis_feats = vis_feats + self.dropout_d(vis_feats2)
            vis_feats = self.norm_d(vis_feats)

        # FFN
        vis_feats = self.norm_vl2(vis_feats + self.ffn_vl(vis_feats))

        return vis_feats, text_feats


class TransformerEncoderLayerNoFFN(nn.Module):
    """TransformerEncoderLayer but without FFN."""

    def __init__(self, d_model, nhead, dropout):
        """Intialize same as Transformer (without FFN params)."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src


class PosTransformerEncoderLayerNoFFN(TransformerEncoderLayerNoFFN):
    """TransformerEncoderLayerNoFFN but additionaly add pos_embed in query."""

    def __init__(self, d_model, nhead, dropout):
        """Intialize same as parent class."""
        super().__init__(d_model, nhead, dropout)

    def forward(self, src, pos, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)
            pos: (S, B, F) positional embeddings
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src + pos, src + pos, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src


class BiEncoderLayer(nn.Module):
    """Self->cross layer for both modalities."""

    def __init__(self, d_model=256, dropout=0.1, activation="relu", n_heads=8,
                 dim_feedforward=256,
                 self_attend_lang=True, self_attend_vis=True,
                 use_butd_enc_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # self attention in language
        if self_attend_lang:
            self.self_attention_lang = TransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout
            )
        else:
            self.self_attention_lang = None

        # self attention in vision
        if self_attend_vis:
            self.self_attention_visual = PosTransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout
            )
        else:
            self.self_attention_visual = None

        # cross attention in language and vision
        self.cross_layer = CrossAttentionLayer(
            d_model, dropout, n_heads, dim_feedforward,
            use_butd_enc_attn
        )

    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={}, detected_feats=None,
                detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        # Self attention for image
        if self.self_attention_visual is not None:
            vis_feats = self.self_attention_visual(
                vis_feats.transpose(0, 1),
                pos_feats.transpose(0, 1),
                src_key_padding_mask=padding_mask
            ).transpose(0, 1)

        # Self attention for language
        if self.self_attention_lang is not None:
            text_feats = self.self_attention_lang(
                text_feats.transpose(0, 1),
                src_key_padding_mask=text_padding_mask
            ).transpose(0, 1)

        # Cross attention
        vis_feats, text_feats = self.cross_layer(
            vis_feats=vis_feats,
            vis_key_padding_mask=padding_mask,
            text_feats=text_feats,
            text_key_padding_mask=text_padding_mask,
            pos_feats=pos_feats,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )

        return vis_feats, text_feats


class BiEncoder(nn.Module):
    """Encode jointly language and vision."""

    def __init__(self, bi_layer, num_layers):
        """Pass initialized BiEncoderLayer and number of such layers."""
        super().__init__()
        self.layers = _get_clones(bi_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={},
                detected_feats=None, detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        for i, layer in enumerate(self.layers):
            vis_feats, text_feats = layer(
                vis_feats,
                pos_feats,
                padding_mask,
                text_feats,
                text_padding_mask,
                end_points,
                detected_feats=detected_feats,
                detected_mask=detected_mask
            )
            if 'lv_attention' in end_points:
                end_points['lv_attention%d' % i] = end_points['lv_attention']
        return vis_feats, text_feats


class BiDecoderLayer(nn.Module):
    """Self->cross_l->cross_v layer for proposals."""

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 self_position_embedding='loc_learned', butd=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross attention in language
        self.cross_l = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_l = nn.Dropout(dropout)
        self.norm_l = nn.LayerNorm(d_model)

        if butd:
            # Cross attention in detected boxes
            self.cross_d = deepcopy(self.cross_l)
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

        # Cross attention in vision
        self.cross_v = deepcopy(self.cross_l)
        self.dropout_v = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Positional embeddings
        if self_position_embedding == 'xyz_learned':
            self.self_posembed = PositionEmbeddingLearned(3, d_model)
        elif self_position_embedding == 'loc_learned':
            self.self_posembed = PositionEmbeddingLearned(6, d_model)
        else:
            self.self_posembed = None

    def forward(self, query, vis_feats, lang_feats, query_pos,
                padding_mask, text_key_padding_mask,
                detected_feats=None, detected_mask=None):
        """
        Forward pass.
        Args:
            query: (B, N, F)
            vis_feats: (B, V, F)
            lang_feats: (B, L, F)
            query_pos: (B, N, 3or6)
            padding_mask: (B, N) (for query)
            text_key_padding_mask: (B, L)
        Returns:
            query: (B, N, F)
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos = self.self_posembed(query_pos)
            query_pos = query_pos.transpose(1, 2).contiguous()
        else:
            query_pos = torch.zeros_like(query, device=query.device)
        query = query.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)

        # Self attention
        query2 = self.self_attn(
            query + query_pos, query + query_pos, query,
            attn_mask=None,
            key_padding_mask=padding_mask
        )[0]
        query = self.norm1(query + self.dropout1(query2))

        # Cross attend to language
        query2 = self.cross_l(
            query=query + query_pos,
            key=lang_feats.transpose(0, 1),
            value=lang_feats.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=text_key_padding_mask  # (B, L)
        )[0]
        query = self.norm_l(query + self.dropout_l(query2))

        # Cross attend to detected boxes
        if detected_feats is not None:
            query2 = self.cross_d(
                query=query + query_pos,
                key=detected_feats.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=detected_mask
            )[0]
            query = self.norm_d(query + self.dropout_d(query2))

        # Cross attend to vision
        query2 = self.cross_v(
            query=(query + query_pos),
            key=vis_feats.transpose(0, 1),
            value=vis_feats.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=None
        )[0]
        query = self.norm_v(query + self.dropout_v(query2))

        # FFN
        query = self.norm2(query + self.ffn(query))

        return query.transpose(0, 1).contiguous()
