from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Amino-acid mappings (locked)
# -----------------------------
AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

AA1_LIST = list("ARNDCQEGHILKMFPSTWYV")
AA1_TO_IDX = {a:i for i,a in enumerate(AA1_LIST)}

# Coarse groups (you locked these)
POSITIVE = set(["K","R","H"])
NEGATIVE = set(["D","E"])
AROMATIC = set(["F","W","Y"])
SULFUR   = set(["C","M"])
GLY      = set(["G"])
PRO      = set(["P"])
POLAR    = set(["S","T","N","Q","C","Y"])
HYDROPHOBIC = set(["A","V","I","L","M","F","W","P"])

# Physicochemical scalars (simple, interpretable)
# Kyte-Doolittle hydrophobicity (common scale)
KD_HYDRO = {
    "A": 1.8, "R":-4.5, "N":-3.5, "D":-3.5, "C": 2.5,
    "Q":-3.5, "E":-3.5, "G":-0.4, "H":-3.2, "I": 4.5,
    "L": 3.8, "K":-3.9, "M": 1.9, "F": 2.8, "P":-1.6,
    "S":-0.8, "T":-0.7, "W":-0.9, "Y":-1.3, "V": 4.2
}

# Side-chain volumes (Å^3) — standard approximate values
SC_VOLUME = {
    "A":  88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G":  60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S":  89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0
}

# Net side-chain charge at ~physiological pH (simple)
NET_CHARGE = {a: 0.0 for a in AA1_LIST}
for a in POSITIVE:
    NET_CHARGE[a] = 1.0
for a in NEGATIVE:
    NET_CHARGE[a] = -1.0


@dataclass
class FeatureConfig:
    # include coarse flags even though redundant with one-hot (locked)
    include_flags: bool = True
    include_position: bool = True

    # use float32 everywhere
    dtype: str = "float32"


def _aa1_from_resname(resname3: str) -> str:
    aa1 = AA3_TO_AA1.get(resname3.upper(), "X")
    return aa1


def load_residue_table(example_dir: Path, chain: str) -> pd.DataFrame:
    return pd.read_csv(example_dir / f"chain{chain}_residues.csv")


def load_graph(example_dir: Path, chain: str) -> Tuple[np.ndarray, np.ndarray]:
    edge_index = np.load(example_dir / f"graph_chain{chain}_edges.npy")  # (2,E)
    edge_dist  = np.load(example_dir / f"graph_chain{chain}_edge_dist.npy")  # (E,)
    return edge_index, edge_dist


def load_labels(example_dir: Path, chain: str, t_angstrom: int = 5) -> np.ndarray:
    return np.load(example_dir / f"labels_chain{chain}_t{t_angstrom}.npy").astype(np.int8)


def build_features_for_chain(
    example_dir: Path,
    chain: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      X: (N, D) float32 feature matrix for residues in chain
      feature_names: list of length D
    """
    df = load_residue_table(example_dir, chain)
    resnames = df["resname"].tolist()
    aa1 = np.array([_aa1_from_resname(r) for r in resnames], dtype=object)
    n = len(aa1)

    # --- Identity one-hot (20)
    onehot = np.zeros((n, 20), dtype=np.float32)
    for i, a in enumerate(aa1):
        if a in AA1_TO_IDX:
            onehot[i, AA1_TO_IDX[a]] = 1.0

    # --- Coarse flags (locked)
    flags = []
    flag_names = []
    if cfg.include_flags:
        def mk_flag(name: str, s: set):
            arr = np.array([1.0 if a in s else 0.0 for a in aa1], dtype=np.float32)
            flags.append(arr[:, None])
            flag_names.append(name)

        mk_flag("is_positive", POSITIVE)
        mk_flag("is_negative", NEGATIVE)
        # derived sets
        mk_flag("is_charged", POSITIVE | NEGATIVE)
        mk_flag("is_polar", POLAR)
        mk_flag("is_hydrophobic", HYDROPHOBIC)
        mk_flag("is_aromatic", AROMATIC)
        mk_flag("is_sulfur", SULFUR)
        mk_flag("is_gly", GLY)
        mk_flag("is_pro", PRO)

    flags_mat = np.concatenate(flags, axis=1) if flags else np.zeros((n, 0), dtype=np.float32)

    # --- Physicochemical scalars (locked)
    hydro = np.array([KD_HYDRO.get(a, 0.0) for a in aa1], dtype=np.float32)[:, None]
    vol   = np.array([SC_VOLUME.get(a, 0.0) for a in aa1], dtype=np.float32)[:, None]
    chg   = np.array([NET_CHARGE.get(a, 0.0) for a in aa1], dtype=np.float32)[:, None]

    # --- Graph features
    edge_index, edge_dist = load_graph(example_dir, chain)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    E = len(edge_dist)

    # degree (out-degree; with symmetric edges this equals undirected degree)
    degree = np.zeros(n, dtype=np.float32)
    np.add.at(degree, src, 1.0)
    degree_col = degree[:, None]

    # mean/min neighbor distance per node
    sum_d = np.zeros(n, dtype=np.float32)
    np.add.at(sum_d, src, edge_dist.astype(np.float32))
    mean_d = (sum_d / np.maximum(1.0, degree)).astype(np.float32)

    # min distance: initialize inf, take min over outgoing edges
    min_d = np.full(n, np.inf, dtype=np.float32)
    # vectorized min via loop over edges (E is manageable for residue graphs)
    for i in range(E):
        s = int(src[i])
        d = float(edge_dist[i])
        if d < min_d[s]:
            min_d[s] = d
    # residues with no neighbors -> set 0 (rare at R=10Å), keeps finite
    min_d[~np.isfinite(min_d)] = 0.0

    geo = np.stack([degree, mean_d, min_d], axis=1).astype(np.float32)
    geo_names = ["contact_degree", "mean_neighbor_dist", "min_neighbor_dist"]

    # --- Neighborhood composition + mean neighbor degree (locked)
    # Precompute per-node chemistry flags as 0/1 scalars
    is_pos = np.array([1.0 if a in POSITIVE else 0.0 for a in aa1], dtype=np.float32)
    is_neg = np.array([1.0 if a in NEGATIVE else 0.0 for a in aa1], dtype=np.float32)
    is_chg = (is_pos + is_neg).clip(0, 1)
    is_hyd = np.array([1.0 if a in HYDROPHOBIC else 0.0 for a in aa1], dtype=np.float32)
    is_pol = np.array([1.0 if a in POLAR else 0.0 for a in aa1], dtype=np.float32)
    is_aro = np.array([1.0 if a in AROMATIC else 0.0 for a in aa1], dtype=np.float32)

    # Sum neighbor properties over outgoing edges (i -> j contributes j's property to i)
    def neigh_mean(node_scalar: np.ndarray) -> np.ndarray:
        sums = np.zeros(n, dtype=np.float32)
        np.add.at(sums, src, node_scalar[dst])
        return (sums / np.maximum(1.0, degree)).astype(np.float32)

    frac_pos = neigh_mean(is_pos)
    frac_neg = neigh_mean(is_neg)
    frac_chg = neigh_mean(is_chg)
    frac_hyd = neigh_mean(is_hyd)
    frac_pol = neigh_mean(is_pol)
    frac_aro = neigh_mean(is_aro)
    mean_neigh_deg = neigh_mean(degree)

    neigh = np.stack(
        [frac_pos, frac_neg, frac_chg, frac_hyd, frac_pol, frac_aro, mean_neigh_deg],
        axis=1
    ).astype(np.float32)

    neigh_names = [
        "frac_positive_neighbors",
        "frac_negative_neighbors",
        "frac_charged_neighbors",
        "frac_hydrophobic_neighbors",
        "frac_polar_neighbors",
        "frac_aromatic_neighbors",
        "mean_neighbor_degree",
    ]

    # --- Relative position (locked)
    pos = np.zeros((n, 0), dtype=np.float32)
    pos_names: List[str] = []
    if cfg.include_position:
        if n <= 1:
            rel = np.zeros(n, dtype=np.float32)
        else:
            rel = (np.arange(n, dtype=np.float32) / float(n - 1)).astype(np.float32)
        pos = rel[:, None]
        pos_names = ["relative_position"]

    # Concatenate all
    X = np.concatenate([onehot, flags_mat, hydro, vol, chg, geo, neigh, pos], axis=1).astype(np.float32)

    feature_names = (
        [f"aa_{a}" for a in AA1_LIST]
        + flag_names
        + ["hydrophobicity_kd", "sidechain_volume", "net_charge"]
        + geo_names
        + neigh_names
        + pos_names
    )

    return X, feature_names
