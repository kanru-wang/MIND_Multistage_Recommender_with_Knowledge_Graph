from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    in_dim: int,
    layer_dims: list[int],
    dropout: float = 0.0,
    last_activation: bool = True,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for i, od in enumerate(layer_dims):
        layers.append(nn.Linear(d, od))
        if i < len(layer_dims) - 1 or last_activation:
            layers.append(nn.ReLU())
        if dropout > 0.0 and i < len(layer_dims) - 1:
            layers.append(nn.Dropout(dropout))
        d = od
    return nn.Sequential(*layers)


class AttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.out = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B,D], kv: [B,T,D]
        q2 = self.q_proj(q).unsqueeze(1)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        out, _ = self.attn(q2, k, v, need_weights=False)
        return self.out(out.squeeze(1))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def mask_pooled_vector(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    has_history = mask.any(dim=1, keepdim=True).to(dtype=x.dtype)
    return x * has_history


class DLRMStudent(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_news: int,
        n_cats: int,
        n_subcats: int,
        dense_dim: int,
        item_base_dim: int,
        emb_dim: int = 64,
        id_emb_dim: int | None = None,
        bottom_mlp: list[int] | None = None,
        top_mlp: list[int] | None = None,
        dropout: float = 0.0,
        fusion_heads: int = 4,
        semantic_ff_mult: int = 1,
        semantic_dropout: float | None = None,
        news_id_warm_scale: float = 1.0,
        news_id_cold_scale: float = 1.0,
        use_time_features: bool = False,
        use_hour_feature: bool = True,
        use_weekday_feature: bool = False,
    ) -> None:
        super().__init__()
        bottom_mlp = bottom_mlp or [128, 64]
        top_mlp = top_mlp or [256, 128, 1]
        id_emb_dim = id_emb_dim or emb_dim
        semantic_dropout = dropout if semantic_dropout is None else semantic_dropout
        self.news_id_warm_scale = float(news_id_warm_scale)
        self.news_id_cold_scale = float(news_id_cold_scale)
        self.use_hour_feature = bool(use_time_features and use_hour_feature)
        self.use_weekday_feature = bool(use_time_features and use_weekday_feature)

        self.user_emb = nn.Embedding(n_users, id_emb_dim, padding_idx=0)
        self.news_emb = nn.Embedding(n_news, id_emb_dim, padding_idx=0)
        self.user_id_proj = nn.Linear(id_emb_dim, emb_dim)
        self.news_id_proj = nn.Linear(id_emb_dim, emb_dim)
        self.cat_emb = nn.Embedding(n_cats, emb_dim, padding_idx=0)
        self.subcat_emb = nn.Embedding(n_subcats, emb_dim, padding_idx=0)
        if self.use_hour_feature:
            self.hour_emb = nn.Embedding(24, emb_dim)
        if self.use_weekday_feature:
            self.weekday_emb = nn.Embedding(7, emb_dim)

        self.bottom = make_mlp(
            dense_dim, bottom_mlp, dropout=dropout, last_activation=True
        )
        d_bottom = bottom_mlp[-1]
        self.xd_proj = nn.Linear(d_bottom, emb_dim)

        self.item_base_proj = nn.Linear(item_base_dim, emb_dim)
        semantic_hidden = emb_dim * semantic_ff_mult
        self.item_base_mlp = nn.Sequential(
            nn.Linear(emb_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(semantic_dropout),
            nn.Linear(semantic_hidden, emb_dim),
        )
        self.hist_refine = nn.Sequential(
            nn.Linear(emb_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(semantic_dropout),
            nn.Linear(semantic_hidden, emb_dim),
        )
        self.user_sem_proj = nn.Linear(emb_dim, emb_dim)
        self.item_sem_proj = nn.Linear(emb_dim, emb_dim)
        self.semantic_fusion = AttentionFusion(dim=emb_dim, heads=fusion_heads)

        # DLRM interaction features:
        # xd_emb, user_id, news_id, category, subcategory, user_sem, item_sem, sem_fused
        self.n_feat = 8
        n_inter = self.n_feat * (self.n_feat - 1) // 2
        n_time_inter = 2 * int(self.use_hour_feature) + 2 * int(self.use_weekday_feature)
        n_concat_emb = self.n_feat
        top_in = d_bottom + n_concat_emb * emb_dim + n_inter + n_time_inter

        self.top = make_mlp(top_in, top_mlp, dropout=dropout, last_activation=False)

    def _encode_item_base(self, item_base: torch.Tensor) -> torch.Tensor:
        base = self.item_base_proj(item_base)
        refined = self.item_base_mlp(base)
        return F.normalize(base + refined, dim=-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        news_idx: torch.Tensor,
        cat_idx: torch.Tensor,
        subcat_idx: torch.Tensor,
        dense: torch.Tensor,
        item_base: torch.Tensor,
        history_item_base: torch.Tensor,
        history_mask: torch.Tensor,
        is_new_item: torch.Tensor | None = None,
        hour_idx: torch.Tensor | None = None,
        weekday_idx: torch.Tensor | None = None,
        return_repr: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        xd = self.bottom(dense)
        xd_emb = self.xd_proj(xd)

        item_sem = self.item_sem_proj(self._encode_item_base(item_base))
        # The same item embedding may play slightly different roles as a candidate
        # item vs. as part of a user’s past behavior.
        # hist_sem is a residual network which gives the model a chance to make
        # history item vectors more user-profile-friendly before aggregation.
        hist_sem = self._encode_item_base(history_item_base)
        hist_sem = hist_sem + self.hist_refine(hist_sem)
        user_hist = masked_mean(hist_sem, history_mask)
        user_sem = self.user_sem_proj(user_hist)
        user_sem = mask_pooled_vector(user_sem, history_mask)
        user_sem = F.normalize(user_sem, dim=-1)
        eu = self.user_id_proj(self.user_emb(user_idx))
        en = self.news_id_proj(self.news_emb(news_idx))
        if is_new_item is not None:
            # Down-weight the memorized news-ID branch for cold/new items so
            # scoring falls back to category, dense, and semantic item signals.
            cold = is_new_item.to(dtype=en.dtype).unsqueeze(1)
            scale = (1.0 - cold) * self.news_id_warm_scale + cold * self.news_id_cold_scale
            en = en * scale
        ec = self.cat_emb(cat_idx)
        es = self.subcat_emb(subcat_idx)
        query = xd_emb + eu + 0.5 * (ec + es)

        concat_feats = [xd_emb, eu, en, ec, es]

        sem_fused = self.semantic_fusion(
            q=query, kv=torch.stack([user_sem, item_sem], dim=1)
        )

        feats = concat_feats + [user_sem, item_sem, sem_fused]
        inter = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                inter.append((feats[i] * feats[j]).sum(dim=1, keepdim=True))
        inter_vec = (
            torch.cat(inter, dim=1)
            if inter
            else torch.zeros((xd.size(0), 0), device=xd.device)
        )
        time_inter = []
        if self.use_hour_feature:
            if hour_idx is None:
                hour_idx = torch.zeros_like(cat_idx)
            eh = self.hour_emb(hour_idx.clamp(min=0, max=23))
            time_inter.append((ec * eh).sum(dim=1, keepdim=True))
            time_inter.append((es * eh).sum(dim=1, keepdim=True))
        if self.use_weekday_feature:
            if weekday_idx is None:
                weekday_idx = torch.zeros_like(cat_idx)
            ew = self.weekday_emb(weekday_idx.clamp(min=0, max=6))
            time_inter.append((ec * ew).sum(dim=1, keepdim=True))
            time_inter.append((es * ew).sum(dim=1, keepdim=True))
        time_inter_vec = (
            torch.cat(time_inter, dim=1)
            if time_inter
            else torch.zeros((xd.size(0), 0), device=xd.device)
        )

        concat = torch.cat(
            [xd] + feats + [inter_vec, time_inter_vec], dim=1
        )
        logit = self.top(concat).squeeze(1)

        rep = None
        if return_repr:
            rep = torch.cat([user_sem, item_sem, sem_fused], dim=1)
        return logit, rep
