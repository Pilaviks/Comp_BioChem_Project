from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNConfig:
    in_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    num_rbf: int = 12
    rbf_dmin: float = 2.0
    rbf_dmax: float = 10.0
    dropout: float = 0.10


class RBFEncoder(nn.Module):
    """
    Encodes distances d into K radial basis features:
      rbf_k(d) = exp(-gamma * (d - mu_k)^2)
    """
    def __init__(self, num_rbf: int, dmin: float, dmax: float):
        super().__init__()
        self.num_rbf = num_rbf
        mus = torch.linspace(dmin, dmax, steps=num_rbf)
        self.register_buffer("mus", mus)
        # gamma chosen so that neighboring bases overlap reasonably
        step = (dmax - dmin) / max(1, (num_rbf - 1))
        gamma = 1.0 / (step ** 2 + 1e-8)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: (E,)
        # returns (E, K)
        diff = d[:, None] - self.mus[None, :]
        return torch.exp(-self.gamma * diff * diff)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        h = hidden if hidden is not None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DistAttentionLayer(nn.Module):
    """
    Distance-aware attention message passing.

    Inputs:
      h: (N, H)
      edge_index: (2, E) with src=i, dst=j meaning message i <- j
      edge_dist: (E,) distances in Å

    Message:
      m_i = sum_{j in N(i)} alpha_ij * V h_j
    Attention:
      alpha_ij = softmax_i( a^T tanh( Wq h_i + Wk h_j + We rbf(d_ij) ) )
    """
    def __init__(self, hidden_dim: int, num_rbf: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.We = nn.Linear(num_rbf, hidden_dim, bias=False)

        self.att_vec = nn.Linear(hidden_dim, 1, bias=False)
        self.out_mlp = MLP(2 * hidden_dim, hidden_dim, hidden=hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        Compute softmax over edges grouped by index (target node).
        scores: (E,)
        index: (E,) target node id for each edge
        """
        # For numerical stability, subtract max per node
        max_per = torch.full((n_nodes,), -1e9, device=scores.device, dtype=scores.dtype)
        max_per.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)

        scores_exp = torch.exp(scores - max_per[index])
        sum_per = torch.zeros((n_nodes,), device=scores.device, dtype=scores.dtype)
        sum_per.scatter_add_(0, index, scores_exp)

        return scores_exp / (sum_per[index] + 1e-12)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        """
        h: (N, H)
        edge_index: (2, E) long
        edge_rbf: (E, K)
        """
        N = h.size(0)
        src = edge_index[0]  # receiving node i
        dst = edge_index[1]  # neighbor node j

        qi = self.Wq(h[src])          # (E, H)
        kj = self.Wk(h[dst])          # (E, H)
        ej = self.We(edge_rbf)        # (E, H)

        att_h = torch.tanh(qi + kj + ej)   # (E, H)
        att_score = self.att_vec(att_h).squeeze(-1)  # (E,)

        alpha = self._scatter_softmax(att_score, src, N)  # (E,)
        alpha = self.dropout(alpha)

        vj = self.Wv(h[dst])  # (E, H)
        msg = vj * alpha[:, None]  # (E, H)

        agg = torch.zeros((N, self.hidden_dim), device=h.device, dtype=h.dtype)
        agg.index_add_(0, src, msg)  # sum over incoming edges

        h_new = self.out_mlp(torch.cat([h, agg], dim=-1))  # (N, H)
        h_new = self.dropout(h_new)
        # residual + norm
        return self.norm(h + h_new)


class InterfaceGNN(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.rbf = RBFEncoder(cfg.num_rbf, cfg.rbf_dmin, cfg.rbf_dmax)

        self.in_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            DistAttentionLayer(cfg.hidden_dim, cfg.num_rbf, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor):
        """
        x: (N, in_dim)
        edge_index: (2, E) long, src=receiver, dst=neighbor
        edge_dist: (E,) float
        returns:
          logits: (N,)
          att_info: None (we’ll extract attention later by re-running layer-wise if needed)
        """
        h = self.in_proj(x)
        edge_rbf = self.rbf(edge_dist)

        for layer in self.layers:
            h = layer(h, edge_index, edge_rbf)

        logits = self.head(h).squeeze(-1)
        return logits
