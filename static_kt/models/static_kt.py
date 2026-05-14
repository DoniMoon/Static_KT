from __future__ import annotations

import math

import torch
import torch.nn as nn


def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class StaticKT(nn.Module):
    """
    Static KT in the log-odds domain.

    The model keeps a fixed item prior and adds attention-weighted evidence from
    previously answered items, without any temporal decay or explicit transition model.
    """

    def __init__(
        self,
        pi: torch.Tensor,
        rank: int = 128,
        pad_id: int = 0,
        init_embed_std: float = 1e-3,
    ):
        super().__init__()
        assert pi.ndim == 1

        self.num_items = int(pi.shape[0])
        self.rank = int(rank)
        self.pad_id = int(pad_id)
        self.emb_size = self.num_items + 1

        self.register_buffer("pi", pi.float(), persistent=True)

        self.beta_q = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.beta_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)

        self.delta_response = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.delta_plus_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.delta_minus_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)

        self._init_parameters(init_embed_std)

    def _init_parameters(self, init_embed_std: float = 1e-3) -> None:
        for mod in [self.beta_q, self.beta_k, self.delta_response, self.delta_plus_k, self.delta_minus_k]:
            nn.init.normal_(mod.weight, mean=0.0, std=init_embed_std)

    def history_mask(
        self,
        hist_indices: torch.LongTensor,
        target_items: torch.LongTensor,
    ) -> torch.Tensor:
        del target_items
        return hist_indices != self.pad_id

    def forward(
        self,
        hist_indices: torch.LongTensor,
        hist_values: torch.FloatTensor,
        target_items: torch.LongTensor,
    ) -> torch.Tensor:
        _, hist_len = hist_indices.shape
        device = target_items.device

        p_i = self.pi[target_items - 1]
        prior_term = safe_logit(p_i).to(device)

        if hist_len == 0:
            return prior_term

        mask = self.history_mask(hist_indices, target_items)
        if not torch.any(mask):
            return prior_term

        q_beta = self.beta_q(target_items)
        k_beta = self.beta_k(hist_indices)

        attn_scores = torch.einsum("br,bhr->bh", q_beta, k_beta) / math.sqrt(self.rank)
        attn_scores = attn_scores.masked_fill(~mask, -1e4)
        beta = torch.softmax(attn_scores, dim=1)
        beta = beta * mask.float()
        beta = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-12)

        q_delta = self.delta_response(target_items)
        k_delta_plus = self.delta_plus_k(hist_indices)
        k_delta_minus = self.delta_minus_k(hist_indices)

        delta_plus_val = torch.einsum("br,bhr->bh", q_delta, k_delta_plus)
        delta_minus_val = torch.einsum("br,bhr->bh", q_delta, k_delta_minus)

        is_correct = (hist_values > 0.5).float()
        is_wrong = (hist_values < -0.5).float()
        evidence_term = (is_correct * delta_plus_val) + (is_wrong * delta_minus_val)

        history_update = torch.sum(beta * evidence_term, dim=1)
        return prior_term + history_update

    @torch.no_grad()
    def diagnostics(self) -> dict:
        return {
            "num_items": self.num_items,
            "rank": self.rank,
            "architecture": "StaticKT",
        }
