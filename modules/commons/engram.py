import math
from typing import Iterable, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from utils.nn.fa import (
    RMSNorm,
)


class NGramEngram(nn.Module):
    """
    Engram-style hashed suffix n-gram memory with:
      - multi-head hashing lookup
      - contextualized gating (hidden as Query, memory as Key)
      - per-order projection + per-order gating
      - multi-branch integration (shared tables + shared Wv, branch-specific Wk)
      - optional lightweight dwconv (non-causal; can disable via conv_kernel<=1)

    token_ids:   [B,T] (int/long)
    token_mask:  [B,T] bool (True=valid)
    hidden_for_gate: [B,T,D] float (Query)
    return: delta [B,T,D]
    """
    def __init__(
        self,
        *,
        model_dim: int,
        orders: Tuple[int, ...] = (1, 2, 3),
        num_heads: int = 4,
        table_size: int = 262144,
        head_dim: int = 64,
        dropout: float = 0.0,
        gate_bias: float = -4.0,
        conv_kernel: int = 3,
        num_branches: int = 2,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.orders = tuple(int(x) for x in orders)
        self.num_heads = int(num_heads)
        self.table_size = int(table_size)
        self.head_dim = int(head_dim)
        self.num_branches = int(num_branches)

        # ===== changed: one big table per order =====
        # each order has a single embedding table of size:
        #   [num_heads * table_size, head_dim]
        # head-specific bucket is implemented by adding offset: head_id * table_size
        self.tables = nn.ModuleList([
            nn.Embedding(self.num_heads * self.table_size, self.head_dim)
            for _ in self.orders
        ])

        # per-order projection: [num_heads * head_dim] -> [model_dim]
        per_order_mem_in_dim = self.num_heads * self.head_dim
        self.proj = nn.ModuleList([
            nn.Linear(per_order_mem_in_dim, self.model_dim, bias=False)
            for _ in self.orders
        ])

        # shared Wv
        self.wv = nn.Linear(self.model_dim, self.model_dim, bias=False)

        # ===== changed: merge branch-specific Wk into one big linear =====
        self.wk = nn.Linear(
            self.model_dim,
            self.num_branches * self.model_dim,
            bias=False,
        )

        self.q_norm = RMSNorm(self.model_dim)
        self.k_norm = RMSNorm(self.model_dim)
        self.v_norm = RMSNorm(self.model_dim)

        self.gate_bias = nn.Parameter(torch.full((self.num_branches,), float(gate_bias)))

        self.use_conv = (conv_kernel is not None) and (int(conv_kernel) > 1)
        if self.use_conv:
            k = int(conv_kernel)
            self.dwconv = nn.Conv1d(
                self.model_dim,
                self.model_dim,
                kernel_size=k,
                padding=k // 2,
                groups=self.model_dim,
                bias=False,
            )
            nn.init.zeros_(self.dwconv.weight)

        self.drop = nn.Dropout(float(dropout))

        # hash seeds
        primes = [1000003, 1009837, 1019699, 1029533, 1039391, 1049297, 1059199, 1069127]
        self.register_buffer(
            "hash_mul",
            torch.tensor(primes[: max(1, self.num_heads)], dtype=torch.int64),
            persistent=False,
        )

        xor_mask = 0x7FFFFFFFFFFFFFFF
        xor_seeds = [
            (0x9E3779B97F4A7C15 + i * 0xBF58476D1CE4E5B9) & xor_mask
            for i in range(max(1, self.num_heads))
        ]
        self.register_buffer(
            "hash_xor",
            torch.tensor(xor_seeds, dtype=torch.int64),
            persistent=False,
        )

        # ===== new: offsets for head-specific bucket ranges =====
        # shape: [1, 1, H]
        self.register_buffer(
            "head_offsets",
            (torch.arange(self.num_heads, dtype=torch.long) * self.table_size).view(1, 1, self.num_heads),
            persistent=False,
        )

    @staticmethod
    def _sanitize_token_ids(
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        special_ids: Optional[Iterable[int]] = None,
        special_replace_id: int = 1,
    ) -> torch.Tensor:
        ids = token_ids.to(torch.long)
        mask = token_mask.to(torch.bool)

        ids = ids.clone()
        ids[~mask] = 0

        if special_ids is not None:
            sid_list = [int(sid) for sid in special_ids if sid is not None]
            if len(sid_list) > 0:
                sid_tensor = torch.tensor(sid_list, device=ids.device, dtype=ids.dtype)
                special_mask = torch.isin(ids, sid_tensor)
                ids = torch.where(
                    special_mask,
                    ids.new_full((), int(special_replace_id)),
                    ids,
                )

        return ids

    def _hash_suffix_ngram_all_heads(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        """
        ids: [B, T]
        return: [B, T, H]
        """
        B, T = ids.shape
        ids64 = ids.to(torch.int64)

        pad_left = n - 1
        ids_pad = F.pad(ids64, (pad_left, 0), value=0)   # [B, T+n-1]
        win = ids_pad.unfold(1, n, 1)                    # [B, T, n]

        mul = self.hash_mul[: self.num_heads].to(ids64.device).view(1, 1, self.num_heads)  # [1,1,H]
        xo = self.hash_xor[: self.num_heads].to(ids64.device).view(1, 1, self.num_heads)    # [1,1,H]

        h = torch.zeros((B, T, self.num_heads), device=ids64.device, dtype=torch.int64)
        for j in range(n):
            tok = win[:, :, j:j + 1]  # [B, T, 1]
            h = (h * mul) ^ (tok + xo + (j + 1) * 0x27D4EB2D)

        h = h & 0x7FFFFFFFFFFFFFFF
        return torch.remainder(h, self.table_size).to(torch.long)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        hidden_for_gate: torch.Tensor,
        *,
        special_ids: Optional[Iterable[int]] = None,
        special_replace_id: int = 1,
    ) -> torch.Tensor:
        """
        return delta: [B,T,D]
        """
        B, T = token_ids.shape
        device = hidden_for_gate.device
        mask = token_mask.to(torch.bool).to(device)
        mask_f = mask.unsqueeze(-1)

        ids = self._sanitize_token_ids(
            token_ids.to(device),
            mask,
            special_ids=special_ids,
            special_replace_id=special_replace_id,
        )

        # 按你的要求：norm 后保持 float
        q = self.q_norm(hidden_for_gate).float()  # [B,T,D]

        out = None

        # ===== per-order lookup -> per-order proj -> per-order gating =====
        for order_i, n in enumerate(self.orders):
            # [B,T,H]
            idx_all = self._hash_suffix_ngram_all_heads(ids, n=n)

            # ===== changed: single embedding lookup per order =====
            # shift each head to its own bucket range
            idx_all = idx_all + self.head_offsets.to(idx_all.device)    # [B,T,H]

            # one lookup -> [B,T,H,head_dim]
            emb = self.tables[order_i](idx_all)

            # flatten heads -> [B,T,H*head_dim]
            e = emb.reshape(B, T, self.num_heads * self.head_dim)

            e = self.proj[order_i](e)   # [B,T,D]
            e = e * mask_f

            v = self.wv(e)              # [B,T,D]

            # ===== changed: one big wk instead of ModuleList =====
            k = self.wk(e).view(B, T, self.num_branches, self.model_dim)   # [B,T,Br,D]
            k = self.k_norm(k).float()

            score = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.model_dim)   # [B,T,Br]
            alpha = torch.sigmoid(
                score + self.gate_bias.view(1, 1, self.num_branches)
            ).to(v.dtype)   # [B,T,Br]

            # broadcast v to branches, then average over branches
            order_out = (alpha.unsqueeze(-1) * v.unsqueeze(2)).mean(dim=2)   # [B,T,D]

            if out is None:
                out = order_out
            else:
                out = out + order_out

        out = out / float(max(1, len(self.orders)))

        # ===== optional conv fusion (non-causal) =====
        if self.use_conv:
            y = self.v_norm(out).to(out.dtype)
            y = self.dwconv(y.transpose(1, 2)).transpose(1, 2)
            out = out + F.silu(y)

        out = out * mask_f
        return self.drop(out)