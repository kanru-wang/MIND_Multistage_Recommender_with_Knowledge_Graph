from __future__ import annotations

import math

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


def masked_softmax(
    scores: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    weights = torch.softmax(scores.masked_fill(~mask, -1.0e9), dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0)


class DLRMStudent(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_news: int,
        n_cats: int,
        n_subcats: int,
        dense_dim: int,
        item_text_dim: int,
        item_kg_dim: int = 0,
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
        kg_gate_init: float = 0.15,
        kg_gate_trainable: bool = False,
    ) -> None:
        super().__init__()
        bottom_mlp = bottom_mlp or [128, 64]
        top_mlp = top_mlp or [256, 128, 1]
        id_emb_dim = id_emb_dim or emb_dim
        semantic_dropout = dropout if semantic_dropout is None else semantic_dropout
        self.news_id_warm_scale = float(news_id_warm_scale)
        self.news_id_cold_scale = float(news_id_cold_scale)
        self.has_kg = int(item_kg_dim) > 0

        self.user_emb = nn.Embedding(n_users, id_emb_dim, padding_idx=0)
        self.news_emb = nn.Embedding(n_news, id_emb_dim, padding_idx=0)
        self.user_id_proj = nn.Linear(id_emb_dim, emb_dim)
        self.news_id_proj = nn.Linear(id_emb_dim, emb_dim)
        self.cat_emb = nn.Embedding(n_cats, emb_dim, padding_idx=0)
        self.subcat_emb = nn.Embedding(n_subcats, emb_dim, padding_idx=0)
        self.bottom = make_mlp(
            dense_dim, bottom_mlp, dropout=dropout, last_activation=True
        )
        d_bottom = bottom_mlp[-1]
        self.xd_proj = nn.Linear(d_bottom, emb_dim)

        semantic_hidden = emb_dim * semantic_ff_mult
        self.text_base_proj = nn.Linear(item_text_dim, emb_dim)
        self.text_base_mlp = nn.Sequential(
            nn.Linear(emb_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(semantic_dropout),
            nn.Linear(semantic_hidden, emb_dim),
        )
        self.text_hist_refine = nn.Sequential(
            nn.Linear(emb_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(semantic_dropout),
            nn.Linear(semantic_hidden, emb_dim),
        )
        self.user_text_proj = nn.Linear(emb_dim, emb_dim)
        self.item_text_proj = nn.Linear(emb_dim, emb_dim)

        if self.has_kg:
            self.kg_base_proj = nn.Linear(item_kg_dim, emb_dim, bias=False)
            self.kg_entity_attn = nn.Linear(emb_dim, emb_dim, bias=False)
            self.kg_text_attn = nn.Linear(emb_dim, emb_dim, bias=False)
            self.kg_attn_score = nn.Linear(emb_dim, 1, bias=False)
            self.user_kg_proj = nn.Linear(emb_dim, emb_dim)
            self.item_kg_proj = nn.Linear(emb_dim, emb_dim)
            if kg_gate_trainable:
                gate = min(max(float(kg_gate_init), 1.0e-4), 1.0 - 1.0e-4)
                self.kg_gate_logit = nn.Parameter(
                    torch.tensor(math.log(gate / (1.0 - gate)), dtype=torch.float32)
                )
            else:
                gate = min(max(float(kg_gate_init), 0.0), 1.0)
                self.register_buffer(
                    "kg_gate_fixed", torch.tensor(gate, dtype=torch.float32)
                )

        self.semantic_fusion = AttentionFusion(dim=emb_dim, heads=fusion_heads)

        # Base features: dense, user_id, news_id, category, subcategory,
        # user_text_sem, item_text_sem, sem_fused.
        # KG adds user_kg_sem and item_kg_sem as ordinary DLRM features.
        self.n_feat = 10 if self.has_kg else 8
        n_inter = self.n_feat * (self.n_feat - 1) // 2
        top_in = d_bottom + self.n_feat * emb_dim + n_inter

        self.top = make_mlp(top_in, top_mlp, dropout=dropout, last_activation=False)

    def _encode_text_base(self, item_text_base: torch.Tensor) -> torch.Tensor:
        base = self.text_base_proj(item_text_base)
        refined = self.text_base_mlp(base)
        return F.normalize(base + refined, dim=-1)

    def _encode_kg_article(
        self,
        entity_slots: torch.Tensor,
        entity_mask: torch.Tensor,
        text_sem: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_kg:
            raise RuntimeError("KG branch was not initialized")
        entities = F.normalize(self.kg_base_proj(entity_slots), dim=-1)
        attn_hidden = torch.tanh(
            self.kg_entity_attn(entities)
            + self.kg_text_attn(text_sem).unsqueeze(-2)
        )
        scores = self.kg_attn_score(attn_hidden).squeeze(-1)
        weights = masked_softmax(scores, entity_mask)
        return (entities * weights.unsqueeze(-1)).sum(dim=-2)

    def kg_gate(self) -> torch.Tensor:
        if not self.has_kg:
            return torch.tensor(0.0, device=self.user_emb.weight.device)
        if hasattr(self, "kg_gate_logit"):
            return torch.sigmoid(self.kg_gate_logit)
        return self.kg_gate_fixed

    def forward(
        self,
        user_idx: torch.Tensor,
        news_idx: torch.Tensor,
        cat_idx: torch.Tensor,
        subcat_idx: torch.Tensor,
        dense: torch.Tensor,
        item_text_base: torch.Tensor,
        history_item_text_base: torch.Tensor,
        history_mask: torch.Tensor,
        item_kg_base: torch.Tensor | None = None,
        history_item_kg_base: torch.Tensor | None = None,
        item_kg_mask: torch.Tensor | None = None,
        history_item_kg_mask: torch.Tensor | None = None,
        is_new_item: torch.Tensor | None = None,
        return_repr: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        xd = self.bottom(dense)
        xd_emb = self.xd_proj(xd)

        item_text_sem = self.item_text_proj(self._encode_text_base(item_text_base))
        hist_text_sem = self._encode_text_base(history_item_text_base)
        hist_text_sem = hist_text_sem + self.text_hist_refine(hist_text_sem)
        user_text = masked_mean(hist_text_sem, history_mask)
        user_text_sem = self.user_text_proj(user_text)
        user_text_sem = mask_pooled_vector(user_text_sem, history_mask)
        user_text_sem = F.normalize(user_text_sem, dim=-1)

        kg_feats: list[torch.Tensor] = []
        if self.has_kg:
            if (
                item_kg_base is None
                or history_item_kg_base is None
                or item_kg_mask is None
                or history_item_kg_mask is None
            ):
                raise ValueError("KG branch requires KG features and availability masks")
            gate = self.kg_gate()
            item_article_mask = item_kg_mask.any(dim=-1)
            history_article_mask = history_item_kg_mask.any(dim=-1)
            kg_history_mask = history_mask & history_article_mask

            item_kg_sem = F.normalize(
                self.item_kg_proj(
                    self._encode_kg_article(
                        item_kg_base, item_kg_mask, item_text_sem
                    )
                ),
                dim=-1,
            )
            item_kg_sem = (
                item_kg_sem
                * item_article_mask.unsqueeze(-1).to(dtype=item_kg_sem.dtype)
                * gate
            )

            hist_kg_sem = self._encode_kg_article(
                history_item_kg_base, history_item_kg_mask, hist_text_sem
            )
            user_kg = masked_mean(hist_kg_sem, kg_history_mask)
            user_kg_sem = F.normalize(self.user_kg_proj(user_kg), dim=-1)
            user_kg_sem = mask_pooled_vector(user_kg_sem, kg_history_mask)
            user_kg_sem = user_kg_sem * gate
            kg_feats = [user_kg_sem, item_kg_sem]

        eu = self.user_id_proj(self.user_emb(user_idx))
        en = self.news_id_proj(self.news_emb(news_idx))
        if is_new_item is not None:
            cold = is_new_item.to(dtype=en.dtype).unsqueeze(1)
            scale = (1.0 - cold) * self.news_id_warm_scale + cold * self.news_id_cold_scale
            en = en * scale
        ec = self.cat_emb(cat_idx)
        es = self.subcat_emb(subcat_idx)
        query = xd_emb + eu + 0.5 * (ec + es)

        sem_fused = self.semantic_fusion(
            q=query, kv=torch.stack([user_text_sem, item_text_sem], dim=1)
        )

        feats = [
            xd_emb,
            eu,
            en,
            ec,
            es,
            user_text_sem,
            item_text_sem,
        ]
        feats.extend(kg_feats)
        feats.append(sem_fused)

        inter = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                inter.append((feats[i] * feats[j]).sum(dim=1, keepdim=True))
        inter_vec = (
            torch.cat(inter, dim=1)
            if inter
            else torch.zeros((xd.size(0), 0), device=xd.device)
        )

        concat = torch.cat([xd] + feats + [inter_vec], dim=1)
        logit = self.top(concat).squeeze(1)

        rep = None
        if return_repr:
            rep = torch.cat([user_text_sem, item_text_sem], dim=1)
        return logit, rep
