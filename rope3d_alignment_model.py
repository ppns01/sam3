#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# small I/O helpers used by both forward / train scripts
# -------------------------------------------------
def load_pt(path: str) -> torch.Tensor:
    return torch.load(path, map_location='cpu')


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_query_aux_tensor(aux_npz: Dict[str, np.ndarray]) -> torch.Tensor:
    arr = np.stack([
        aux_npz['mask'],
        aux_npz['boundary'],
        aux_npz['depth'],
        aux_npz['depth_edge'],
        aux_npz['valid'],
        aux_npz['valid_edge'],
    ], axis=0).astype(np.float32)
    return torch.from_numpy(arr)


def build_anchor_aux_tensor(aux_npz: Dict[str, np.ndarray]) -> torch.Tensor:
    arr = np.stack([
        aux_npz['mask'].astype(np.float32),
        aux_npz['boundary'].astype(np.float32),
        aux_npz['depth'].astype(np.float32),
        aux_npz['depth_edge'].astype(np.float32),
        aux_npz['alpha'].astype(np.float32),
        aux_npz['alpha_edge'].astype(np.float32),
    ], axis=0).astype(np.float32)
    return torch.from_numpy(arr)


# -------------------------------------------------
# basic blocks
# -------------------------------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, groups_gn: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TokenProjector(nn.Module):
    def __init__(self, feat_ch: int = 384, aux_ch: int = 6, embed_dim: int = 256):
        super().__init__()
        self.feat_proj = nn.Conv2d(feat_ch, embed_dim, kernel_size=1, bias=False)
        self.aux_proj = nn.Sequential(
            ConvGNAct(aux_ch, 64, k=3, p=1),
            ConvGNAct(64, 64, k=3, p=1),
        )
        self.fuse = ConvGNAct(embed_dim + 64, embed_dim, k=3, p=1)

    def forward(self, feat: torch.Tensor, aux: torch.Tensor):
        if aux.shape[-2:] != feat.shape[-2:]:
            aux = F.interpolate(aux, size=feat.shape[-2:], mode='bilinear', align_corners=False)

        f = self.feat_proj(feat)
        a = self.aux_proj(aux)
        x = self.fuse(torch.cat([f, a], dim=1))
        token = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,D]
        return token, x


# -------------------------------------------------
# Continuous 3D RoPE
# -------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def rope_angles(coord: torch.Tensor, d_axis: int, base: float = 1000.0):
    # coord: [B,1,N] or [B,H,N]
    n_freq = d_axis // 2
    if n_freq < 1:
        raise ValueError(f'd_axis must be >=2, got {d_axis}')
    idx = torch.arange(n_freq, device=coord.device, dtype=coord.dtype)
    inv_freq = 1.0 / (float(base) ** (idx / max(n_freq - 1, 1)))
    ang = coord[..., None] * inv_freq  # [...,N,n_freq]
    ang = torch.repeat_interleave(ang, 2, dim=-1)  # [...,N,d_axis]
    return ang.cos(), ang.sin()


def apply_continuous_3d_rope(x: torch.Tensor, xyz: torch.Tensor, base: float = 1000.0) -> torch.Tensor:
    """
    x  : [B,H,N,Dh]
    xyz: [B,N,3]
    """
    if x.ndim != 4:
        raise ValueError(f'x must be [B,H,N,Dh], got {x.shape}')
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f'xyz must be [B,N,3], got {xyz.shape}')

    Dh = x.shape[-1]
    d_axis = (Dh // 6) * 2
    if d_axis < 2:
        return x
    d_used = 3 * d_axis

    x_main = x[..., :d_used]
    x_rest = x[..., d_used:]
    xx, xy, xz = torch.split(x_main, d_axis, dim=-1)

    px = xyz[:, None, :, 0]
    py = xyz[:, None, :, 1]
    pz = xyz[:, None, :, 2]

    cosx, sinx = rope_angles(px, d_axis, base=base)
    cosy, siny = rope_angles(py, d_axis, base=base)
    cosz, sinz = rope_angles(pz, d_axis, base=base)

    xx = xx * cosx + rotate_half(xx) * sinx
    xy = xy * cosy + rotate_half(xy) * siny
    xz = xz * cosz + rotate_half(xz) * sinz

    return torch.cat([xx, xy, xz, x_rest], dim=-1)


# -------------------------------------------------
# Attention with explicit Q/K/V projections and RoPE on Q,K only
# -------------------------------------------------
class RoPECrossAttentionBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, rope_base: float = 1000.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim({dim}) must be divisible by num_heads({num_heads})')
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_base = float(rope_base)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = MLP(dim, dim * 4)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, q_xyz: torch.Tensor, kv_xyz: torch.Tensor) -> torch.Tensor:
        B, N, D = q.shape
        M = kv.shape[1]

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        qh = self.q_proj(qn).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        kh = self.k_proj(kvn).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,M,Dh]
        vh = self.v_proj(kvn).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,M,Dh]

        qh = apply_continuous_3d_rope(qh, q_xyz, base=self.rope_base)
        kh = apply_continuous_3d_rope(kh, kv_xyz, base=self.rope_base)

        attn = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)  # [B,H,N,Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q


class TwoWayRoPECrossAttention(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, num_layers: int = 2, rope_base: float = 1000.0):
        super().__init__()
        self.layers_q_from_a = nn.ModuleList([
            RoPECrossAttentionBlock(dim=dim, num_heads=num_heads, rope_base=rope_base)
            for _ in range(num_layers)
        ])
        self.layers_a_from_q = nn.ModuleList([
            RoPECrossAttentionBlock(dim=dim, num_heads=num_heads, rope_base=rope_base)
            for _ in range(num_layers)
        ])

    def forward(self, q: torch.Tensor, a: torch.Tensor, q_xyz: torch.Tensor, a_xyz: torch.Tensor):
        for lq, la in zip(self.layers_q_from_a, self.layers_a_from_q):
            q = lq(q, a, q_xyz, a_xyz)
            a = la(a, q, a_xyz, q_xyz)
        return q, a
class PostSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = MLP(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
class TokenAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        x = self.norm(tokens)
        logits = self.score(x).squeeze(-1)      # [B, N]
        weights = torch.softmax(logits, dim=1)  # 토큰 축에서 softmax
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled

class AlignmentHead(nn.Module):
    def __init__(self, dim: int = 256, cond_dim: int = 21):
        super().__init__()
        self.q_pooler = TokenAttentionPool(dim)
        self.a_pooler = TokenAttentionPool(dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        pair_dim = dim * 5
        self.score_head = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.uncert_head = nn.Conv2d(dim, 1, kernel_size=1)
        self._init_small()

    def _init_small(self):
        last = self.score_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        nn.init.zeros_(self.uncert_head.weight)
        nn.init.zeros_(self.uncert_head.bias)

    def forward(
        self,
        q_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        q_fmap: torch.Tensor,
        delta_vec: torch.Tensor,
        base_pose_vec: torch.Tensor,
    ):
        q_pool = self.q_pooler(q_tokens)
        a_pool = self.a_pooler(a_tokens)
        cond = torch.cat([delta_vec, base_pose_vec], dim=1)
        cond_embed = self.cond_proj(cond)
        pair = torch.cat(
            [q_pool, a_pool, torch.abs(q_pool - a_pool), q_pool * a_pool, cond_embed],
            dim=1,
        )
        return {
            'score_logit': self.score_head(pair),
            'uncert_logit_map': self.uncert_head(q_fmap),
        }


def _xyz_map_to_tokens(xyz_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    입력 허용:
      [B,H,W,3] 또는 [B,3,H,W]
    출력:
      [B,H*W,3]
    """
    if xyz_map.ndim != 4:
        raise ValueError(f'xyz_map must be 4D, got {xyz_map.shape}')

    if xyz_map.shape[-1] == 3:
        xyz_hw3 = xyz_map
    elif xyz_map.shape[1] == 3:
        xyz_hw3 = xyz_map.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f'xyz_map must be [B,H,W,3] or [B,3,H,W], got {xyz_map.shape}')

    if xyz_hw3.shape[1] != H or xyz_hw3.shape[2] != W:
        xyz_chw = xyz_hw3.permute(0, 3, 1, 2).contiguous()
        # [핵심 수정 3] 모델 텐서 변환 중에도 3D 좌표는 섞이지 않도록 무조건 nearest 사용
        xyz_chw = F.interpolate(xyz_chw, size=(H, W), mode='nearest')
        xyz_hw3 = xyz_chw.permute(0, 2, 3, 1).contiguous()

    return xyz_hw3.view(xyz_hw3.shape[0], H * W, 3)


class CrossAttentionAlignmentModel3DRoPE(nn.Module):
    def __init__(
        self,
        feat_ch: int = 384,
        aux_ch: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        rope_base: float = 1000.0,
        post_self_layers: int = 1,
        post_self_dropout: float = 0.0,
    ):
        super().__init__()
        self.q_proj = TokenProjector(feat_ch=feat_ch, aux_ch=aux_ch, embed_dim=embed_dim)
        self.a_proj = TokenProjector(feat_ch=feat_ch, aux_ch=aux_ch, embed_dim=embed_dim)
        self.cross = TwoWayRoPECrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            rope_base=rope_base,
        )
                # cross-attention 뒤에 query 쪽 self-attention
        self.post_q_self = nn.ModuleList([
            PostSelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dropout=post_self_dropout,
            )
            for _ in range(post_self_layers)
        ])

        # cross-attention 뒤에 anchor 쪽 self-attention
        self.post_a_self = nn.ModuleList([
            PostSelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dropout=post_self_dropout,
            )
            for _ in range(post_self_layers)
        ])
        self.head = AlignmentHead(dim=embed_dim, cond_dim=21)

    def forward(
        self,
        q_feat,
        q_aux,
        q_xyz_map,
        a_feat,
        a_aux,
        a_xyz_map,
        delta_vec,
    base_pose_vec,
):
        q_tokens, q_fmap = self.q_proj(q_feat, q_aux)
        a_tokens, a_fmap = self.a_proj(a_feat, a_aux)
        H, W = q_fmap.shape[-2:]
        H2, W2 = a_fmap.shape[-2:]

        q_xyz = _xyz_map_to_tokens(q_xyz_map, H, W)
        a_xyz = _xyz_map_to_tokens(a_xyz_map, H2, W2)
        q_tokens, a_tokens = self.cross(q_tokens, a_tokens, q_xyz, a_xyz)
        for blk_q, blk_a in zip(self.post_q_self, self.post_a_self):
            q_tokens = blk_q(q_tokens)
            a_tokens = blk_a(a_tokens)
        return self.head(q_tokens, a_tokens, q_fmap, delta_vec, base_pose_vec)



def masked_patch_cosine_score(q_feat: torch.Tensor, a_feat: torch.Tensor, q_aux: torch.Tensor, a_aux: torch.Tensor):
    if q_feat.ndim == 3:
        q_feat = q_feat.unsqueeze(0)
    if a_feat.ndim == 3:
        a_feat = a_feat.unsqueeze(0)
    if q_aux.ndim == 3:
        q_aux = q_aux.unsqueeze(0)
    if a_aux.ndim == 3:
        a_aux = a_aux.unsqueeze(0)

    if q_aux.shape[-2:] != q_feat.shape[-2:]:
        q_aux = F.interpolate(q_aux, size=q_feat.shape[-2:], mode='nearest')
    if a_aux.shape[-2:] != a_feat.shape[-2:]:
        a_aux = F.interpolate(a_aux, size=a_feat.shape[-2:], mode='nearest')

    q_mask = q_aux[:, 0] > 0.5
    a_mask = a_aux[:, 0] > 0.5
    valid = q_mask & a_mask

    qn = F.normalize(q_feat, dim=1)
    an = F.normalize(a_feat, dim=1)
    sim = (qn * an).sum(dim=1)
    if valid.sum() < 1:
        return float(sim.mean().item())
    return float(sim[valid].mean().item())

