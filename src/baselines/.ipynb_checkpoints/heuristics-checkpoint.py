from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class BaselineConfig:
    # graph edge radius used in preprocessing (10 Å)
    graph_radius_A: float = 10.0

    # heuristic knobs
    # lower local density -> more surface-like
    surface_density_quantile: float = 0.40  # keep the least-dense 40% as "surface candidates"
    # score uses inverse of local density (plus tiny eps)
    eps: float = 1e-6


def load_chain_graph(example_dir: Path, chain: str):
    """
    chain: "A" or "B"
    returns:
      edge_index: (2, E) int
      edge_dist: (E,) float
    """
    edge_index = np.load(example_dir / f"graph_chain{chain}_edges.npy")
    edge_dist = np.load(example_dir / f"graph_chain{chain}_edge_dist.npy")
    return edge_index, edge_dist


def local_contact_density(n_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    """
    Simple local density proxy: degree (number of neighbors).
    For undirected graphs stored as two directed edges, this is fine.
    """
    deg = np.zeros(n_nodes, dtype=np.float32)
    src = edge_index[0]
    for i in src:
        deg[i] += 1.0
    return deg


def baseline_surface_filtered_distance_score(
    n_nodes: int,
    edge_index: np.ndarray,
    cfg: BaselineConfig,
) -> np.ndarray:
    """
    Baseline score per residue using only single-chain geometry.
    Idea:
      - residues with LOWER local contact density are more surface-like
      - score = 1 / (density + eps)
      - zero out non-surface candidates using a quantile cutoff
    Returns: score (n_nodes,) higher = more likely interface
    """
    density = local_contact_density(n_nodes, edge_index)
    inv = 1.0 / (density + cfg.eps)

    # surface mask: low-density residues
    thr = np.quantile(density, cfg.surface_density_quantile)
    surface_mask = density <= thr

    score = inv.copy()
    score[~surface_mask] = 0.0
    return score


def scores_to_topk_predictions(score: np.ndarray, k: int) -> np.ndarray:
    """
    Returns binary predictions selecting top-k residues by score.
    """
    k = max(1, min(int(k), len(score)))
    idx = np.argsort(-score)[:k]
    pred = np.zeros(len(score), dtype=np.int8)
    pred[idx] = 1
    return pred
