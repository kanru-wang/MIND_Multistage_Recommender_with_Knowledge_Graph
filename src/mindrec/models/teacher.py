from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] True for valid
        q = self.query.expand(x.size(0), 1, x.size(2))
        key_padding_mask = ~mask  # True for pad positions
        out, _ = self.attn(
            q, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out.squeeze(1)


class TransformerHistoryEncoder(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~mask
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TeacherTwoTower(nn.Module):
    def __init__(
        self,
        item_dim: int,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        self.item_mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.zeros_(self.item_proj.bias)
        self.history_encoder = TransformerHistoryEncoder(
            dim=hidden_dim, heads=heads, dropout=dropout
        )
        self.user_pool = HistoryAttentionPool(dim=hidden_dim, heads=heads)

    def encode_items(self, item_emb: torch.Tensor) -> torch.Tensor:
        base = self.item_proj(item_emb)
        refined = self.item_mlp(base)
        return F.normalize(base + refined, dim=-1)

    def encode_user_from_item_vectors(
        self, history_z: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        history_ctx = self.history_encoder(history_z, mask)
        user_z = self.user_pool(history_ctx, mask)
        return F.normalize(user_z, dim=-1)

    def encode_user(self, history_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hist_z = self.encode_items(history_emb)
        return self.encode_user_from_item_vectors(hist_z, mask)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return (a * b).sum(axis=-1)
