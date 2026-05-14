from __future__ import annotations

import torch

from .static_kt import StaticKT


class IndependentKT(StaticKT):
    """
    Static KT with same-KC history masked out.

    Only history items whose KC set is disjoint from the target item's KC set
    are allowed to contribute to attention and evidence aggregation.
    """

    def __init__(
        self,
        pi: torch.Tensor,
        q_matrix: torch.Tensor,
        rank: int = 128,
        pad_id: int = 0,
        init_embed_std: float = 1e-3,
    ):
        super().__init__(
            pi=pi,
            rank=rank,
            pad_id=pad_id,
            init_embed_std=init_embed_std,
        )
        assert q_matrix.ndim == 2
        q_bool = q_matrix.bool()
        q_model_ids = torch.zeros((self.emb_size, q_bool.shape[1]), dtype=torch.bool)
        q_model_ids[1 : self.num_items + 1, :] = q_bool[: self.num_items, :]
        self.register_buffer("q_matrix", q_model_ids, persistent=False)

    def history_mask(
        self,
        hist_indices: torch.LongTensor,
        target_items: torch.LongTensor,
    ) -> torch.Tensor:
        valid_history = hist_indices != self.pad_id
        target_kcs = self.q_matrix[target_items]
        hist_kcs = self.q_matrix[hist_indices]
        shares_kc = torch.any(hist_kcs & target_kcs.unsqueeze(1), dim=-1)
        return valid_history & ~shares_kc

    @torch.no_grad()
    def diagnostics(self) -> dict:
        return {
            "num_items": self.num_items,
            "rank": self.rank,
            "num_kcs": int(self.q_matrix.shape[1]),
            "architecture": "IndependentKT",
        }
