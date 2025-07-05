"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
import functools
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .dfine_utils import distance2bbox, weighting_function
from .utils import (bias_init_with_prob, deformable_attention_core_func_v2,
                    get_activation, inverse_sigmoid)

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[Tuple[int, int]],
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2 or 4], range in [0, 1]
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor, optional): [bs, value_length], True for non-padding elements
        Returns:
            output (Tensor): [bs, query_length, C]
        """
        bs, query_length = query.shape[:2]
        value_length = value.shape[1]

        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.view(bs, value_length, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(bs, query_length, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).view(bs, query_length, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).view(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.view(bs, query_length, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
        return output
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = 'relu',
        n_levels: int = 4,
        n_points: int = 4,
        cross_attn_method: str = 'default',
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

        self.ffn = SwiGLUFFN(d_model, dim_feedforward, d_model)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[Tuple[int, int]],
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        query_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        target2 = self.ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)
        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[Tuple[int, int]],
        text_feats: torch.Tensor,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dec_out_bboxes = []
        dec_out_logits = []

        # FIXME Shihua: Share the same query position embedding across layers
        
        ref_points_detach = F.sigmoid(ref_points_unact.clamp(min=-500, max=500))
        query_pos_embed = query_pos_head(ref_points_detach)
        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            # layer: self_attn, cross_attn, ffn
            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, 
                           memory_mask, query_pos_embed)            

            inter_ref_bbox = F.sigmoid((bbox_head[i](output) + inverse_sigmoid(ref_points_detach)).clamp(min=-500, max=500))

            if self.training:
                dec_output = score_head[i](output)
                dec_out_logits.append(dec_output)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points).clamp(min=-500,max=500)))
            elif i == self.eval_idx:
                dec_output = score_head[i](output)
                dec_out_logits.append(dec_output)
                dec_out_bboxes.append(inter_ref_bbox)
                break
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
        
        queries = output
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), queries


@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        num_levels: int = 3,
        num_points: int = 4,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learn_query_content: bool = False,
        eval_spatial_size: Optional[Tuple[int, int]] = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
        cross_attn_method: str = 'default',
        query_select_method: str = 'default',
        text_dim: int = 256,
        text_out_dim: int = 256
    ) -> None:
        super().__init__()
        assert len(feat_channels) <= num_levels, f"Length of feat_channels ({len(feat_channels)}) must be <= num_levels ({num_levels})"
        assert len(feat_strides) == len(feat_channels), f"Length of feat_strides ({len(feat_strides)}) must match feat_channels ({len(feat_channels)})"
        assert query_select_method in ('default', 'one2many', 'agnostic'), f"Unknown query_select_method: {query_select_method}"
        assert cross_attn_method in ('default', 'discrete'), f"Unknown cross_attn_method: {cross_attn_method}"

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides + [feat_strides[-1] * 2] * (num_levels - len(feat_strides))
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method


        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
            # self.denoising_class_embed = nn.Linear(512, hidden_dim, bias=True)
            # nn.init.xavier_uniform_(self.denoising_class_embed.weight)
            # nn.init.constant_(self.denoising_class_embed.bias, 0.0)

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        # FIXME Shihua: Share the same query position embedding across layers
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3)

        # self.enc_output = nn.Sequential(OrderedDict([
        #     ('proj', nn.Linear(hidden_dim, hidden_dim)),
        #     ('norm', nn.LayerNorm(hidden_dim)),
        # ]))
        self.enc_output = nn.Identity()

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)])

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels) -> None:
        bias = bias_init_with_prob(0.01)
        if isinstance(self.enc_score_head, nn.Linear):
            init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            if isinstance(cls_head, nn.Linear):
                init.constant_(cls_head.bias, bias)
            init.constant_(reg_head.layers[-1].weight, 0)
            init.constant_(reg_head.layers[-1].bias, 0)

        # init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        # FIXME Shihua: Share the same query position embedding across layers
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)

        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                proj_feats.append(self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append((h, w))
        return torch.cat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Optional[List[Tuple[int, int]]] = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = [(int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)) for s in self.feat_strides]

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(dtype=dtype)
            grid_xy = (grid_xy + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.cat([grid_xy, wh], dim=-1).view(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.cat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        text_feats: torch.Tensor,
        denoising_logits: Optional[torch.Tensor] = None,
        denoising_bbox_unact: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device) if self.training or self.eval_spatial_size is None else (self.anchors, self.valid_mask)
        memory = memory * valid_mask.to(memory.dtype)
        
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)
        # enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)

        # FIXME Shihua: only do the bbox prediction for the topk queries
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact.clamp(min=-500, max=500))
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        content = self.tgt_embed.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1) if self.learn_query_content else enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.cat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.cat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_coords_unact: torch.Tensor,
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        else:
            raise ValueError(f"Unknown query_select_method: {self.query_select_method}")

        topk_coords = outputs_coords_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_coords

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
        text_feats: Optional[torch.Tensor] = None,
        text_feats_o: Optional[torch.Tensor] = None,
    ) -> dict:
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_contents, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets, self.num_classes, self.num_queries,  self.denoising_class_embed, num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_contents, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes, text_feats, denoising_contents, denoising_bbox_unact
        )

        out_bboxes, out_logits, queries = self.decoder(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes, text_feats, self.dec_bbox_head,
            self.dec_score_head, self.query_pos_head, attn_mask, None
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, queries = torch.split(queries, dn_meta['dn_num_split'], dim=1)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out, queries

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: List[torch.Tensor], outputs_coord: List[torch.Tensor]) -> List[dict]:
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]