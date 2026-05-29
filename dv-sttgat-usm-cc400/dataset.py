"""
dataset.py – DV-STTGAT  (NYU-only)
=====================================
Packs dual-view graph data into PyG Data objects and returns stratified
train/validation DataLoaders.

NYU enhancements:
  - Sliding window augmentation for training subjects (2 windows per subject):
      Window A : timepoints [  0 : 150 ]
      Window B : timepoints [ 25 : 175 ]
    This doubles the effective training set. Val subjects use the full signal.
  - Motion-aware sample weights via get_motion_weight().

Each Data object carries:
    x               : (N, T_win)  Z-scored BOLD window
    edge_index_pear : (2, E_pear)
    edge_attr_pear  : (E_pear,)
    edge_index_prec : (2, E_prec)
    edge_attr_prec  : (E_prec,)
    y               : (1,) float  – ASD=1, TD=0
    site            : (1,) long   – integer site code
    weight          : (1,) float  – motion-aware sample weight
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional

# ── NYU sliding-window config ────────────────────────────────────────────────
WIN_START_A = 0
WIN_END_A   = 150
WIN_START_B = 25
WIN_END_B   = 175


def get_motion_weight(mfd: float, sigma: float = 0.5) -> float:
    """Exponential decay: higher mFD = lower weight."""
    return float(np.exp(-mfd / sigma))


# ─────────────────────────────────────────────────────────────────────────────
def _make_data_object(
    x_window: torch.Tensor,         # (N, T_win)
    pear_ei, pear_ew,
    prec_ei, prec_ew,
    label: int,
    site: int,
    weight: float,
) -> Data:
    return Data(
        x               = x_window,
        edge_index_pear = pear_ei,
        edge_attr_pear  = pear_ew,
        edge_index_prec = prec_ei,
        edge_attr_prec  = prec_ew,
        y               = torch.tensor([label],  dtype=torch.float32),
        site            = torch.tensor([site],   dtype=torch.long),
        weight          = torch.tensor([weight], dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
def create_pyg_dataloaders(
    padded_bold_tensor,             # torch.Tensor  (B, N, T_max)
    pear_edge_indices,              # list[LongTensor (2, E)]
    pear_edge_weights,              # list[FloatTensor (E,)]
    prec_edge_indices,              # list[LongTensor (2, E)]
    prec_edge_weights,              # list[FloatTensor (E,)]
    labels,                        # list[int]
    site_labels: Optional[List[int]] = None,
    mean_fds:    Optional[List[float]] = None,
    batch_size: int = 16,
    train_split: float = 0.8,
    seed: int = 42,
    motion_sigma: float = 0.5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build stratified train / validation PyG DataLoaders.

    Training subjects receive two sliding-window augmented Data objects
    (window A and window B), doubling the effective training set.
    Validation subjects use the full-length padded signal.

    Parameters
    ----------
    padded_bold_tensor : (B, N, T_max)
    pear_edge_indices  : list[LongTensor]
    pear_edge_weights  : list[FloatTensor]
    prec_edge_indices  : list[LongTensor]
    prec_edge_weights  : list[FloatTensor]
    labels             : list[int]  – 1=ASD, 0=TD
    site_labels        : list[int], optional
    mean_fds           : list[float], optional  – per-subject mean FD (mm)
    batch_size         : int
    train_split        : float
    seed               : int
    motion_sigma       : float  – sigma for motion weight decay

    Returns
    -------
    train_loader, val_loader : PyG DataLoader pair
    """
    B, N, T_max = padded_bold_tensor.shape

    if site_labels is None:
        site_labels = [0] * B
    if mean_fds is None:
        mean_fds = [0.2] * B   # moderate motion default → weight ≈ 0.67

    # ── Stratified train / val split ──────────────────────────────────────────
    rng      = np.random.default_rng(seed)
    pos_idx  = [i for i, l in enumerate(labels) if l == 1]
    neg_idx  = [i for i, l in enumerate(labels) if l == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_train = int(len(pos_idx) * train_split)
    n_neg_train = int(len(neg_idx) * train_split)

    train_idx = pos_idx[:n_pos_train] + neg_idx[:n_neg_train]
    val_idx   = pos_idx[n_pos_train:] + neg_idx[n_neg_train:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # ── Build Training Data objects (with sliding-window augmentation) ─────────
    train_data = []
    for i in train_idx:
        bold_i  = padded_bold_tensor[i]          # (N, T_max)
        pei     = pear_edge_indices[i]
        pew     = pear_edge_weights[i]
        rei     = prec_edge_indices[i]
        rew     = prec_edge_weights[i]
        lbl     = labels[i]
        site    = site_labels[i]
        w       = get_motion_weight(mean_fds[i], sigma=motion_sigma)

        T = bold_i.shape[1]

        # Window A: [0:150]  (only if enough timepoints)
        if T >= WIN_END_A:
            train_data.append(_make_data_object(
                bold_i[:, WIN_START_A:WIN_END_A],
                pei, pew, rei, rew, lbl, site, w,
            ))
        else:
            train_data.append(_make_data_object(
                bold_i, pei, pew, rei, rew, lbl, site, w,
            ))

        # Window B: [25:175]  (only if enough timepoints)
        if T >= WIN_END_B:
            train_data.append(_make_data_object(
                bold_i[:, WIN_START_B:WIN_END_B],
                pei, pew, rei, rew, lbl, site, w,
            ))
        # else: skip second window; subject still gets window A above

    # ── Build Validation Data objects (full signal, no augmentation) ───────────
    val_data = []
    for i in val_idx:
        w = get_motion_weight(mean_fds[i], sigma=motion_sigma)
        val_data.append(_make_data_object(
            padded_bold_tensor[i],
            pear_edge_indices[i], pear_edge_weights[i],
            prec_edge_indices[i], prec_edge_weights[i],
            labels[i], site_labels[i], w,
        ))

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    n_pos_tr  = sum(labels[i] for i in train_idx)
    n_pos_val = sum(labels[i] for i in val_idx)
    print(
        f"[Dataset] Raw train subjects: {len(train_idx)} "
        f"→ augmented Data objects: {len(train_data)} (2 windows each)  "
        f"(ASD={int(n_pos_tr)}, TD={len(train_idx)-int(n_pos_tr)})  |  "
        f"Val: {len(val_data)} subjects "
        f"(ASD={int(n_pos_val)}, TD={len(val_idx)-int(n_pos_val)})  |  "
        f"Batch: {batch_size}"
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
def create_fold_dataloaders(
    padded_bold_tensor,
    pear_edge_indices,
    pear_edge_weights,
    prec_edge_indices,
    prec_edge_weights,
    labels,
    train_idx: List[int],
    val_idx:   List[int],
    site_labels: Optional[List[int]] = None,
    mean_fds:    Optional[List[float]] = None,
    batch_size: int = 16,
    motion_sigma: float = 0.5,
    fold: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Like create_pyg_dataloaders but receives explicit train/val index lists
    (from StratifiedKFold), so the CV loop drives the split.

    Sliding-window augmentation is applied only to training subjects.
    Validation subjects use the full-length padded signal.
    """
    if site_labels is None:
        site_labels = [0] * len(labels)
    if mean_fds is None:
        mean_fds = [0.2] * len(labels)

    # ── Training Data (with sliding-window aug) ──────────────────────────────
    train_data = []
    for i in train_idx:
        bold_i = padded_bold_tensor[i]
        pei    = pear_edge_indices[i]
        pew    = pear_edge_weights[i]
        rei    = prec_edge_indices[i]
        rew    = prec_edge_weights[i]
        lbl    = labels[i]
        site   = site_labels[i]
        w      = get_motion_weight(mean_fds[i], sigma=motion_sigma)
        T      = bold_i.shape[1]

        if T >= WIN_END_A:
            train_data.append(_make_data_object(
                bold_i[:, WIN_START_A:WIN_END_A],
                pei, pew, rei, rew, lbl, site, w,
            ))
        else:
            train_data.append(_make_data_object(
                bold_i, pei, pew, rei, rew, lbl, site, w,
            ))

        if T >= WIN_END_B:
            train_data.append(_make_data_object(
                bold_i[:, WIN_START_B:WIN_END_B],
                pei, pew, rei, rew, lbl, site, w,
            ))

    # ── Validation Data (full signal, no augmentation) ───────────────────────
    val_data = []
    for i in val_idx:
        w = get_motion_weight(mean_fds[i], sigma=motion_sigma)
        val_data.append(_make_data_object(
            padded_bold_tensor[i],
            pear_edge_indices[i], pear_edge_weights[i],
            prec_edge_indices[i], prec_edge_weights[i],
            labels[i], site_labels[i], w,
        ))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    n_pos_tr  = sum(labels[i] for i in train_idx)
    n_pos_val = sum(labels[i] for i in val_idx)
    print(
        f"[Fold {fold+1}] Train subj: {len(train_idx)} "
        f"→ {len(train_data)} aug samples "
        f"(ASD={int(n_pos_tr)}, TD={len(train_idx)-int(n_pos_tr)})  |  "
        f"Val: {len(val_data)} subj "
        f"(ASD={int(n_pos_val)}, TD={len(val_idx)-int(n_pos_val)})"
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, N, T = 32, 116, 180

    mock_bold   = torch.randn(B, N, T)
    mock_pear_i = [torch.randint(0, N, (2, 400)) for _ in range(B)]
    mock_pear_w = [torch.rand(400) * 2 - 1       for _ in range(B)]
    mock_prec_i = [torch.randint(0, N, (2, 200)) for _ in range(B)]
    mock_prec_w = [torch.rand(200) * 2 - 1       for _ in range(B)]
    mock_labels = (np.arange(B) % 2).tolist()
    mock_fds    = [np.random.uniform(0.1, 0.45) for _ in range(B)]

    print("=" * 60)
    print("Building DV-STTGAT NYU DataLoaders (stratified + sliding window) …")
    train_loader, val_loader = create_pyg_dataloaders(
        mock_bold,
        mock_pear_i, mock_pear_w,
        mock_prec_i, mock_prec_w,
        mock_labels,
        site_labels=[0] * B,
        mean_fds=mock_fds,
        batch_size=8,
    )

    for batch in train_loader:
        print(f"\n  x              : {batch.x.shape}   (expect [8, {N}, 150])")
        print(f"  edge_index_pear: {batch.edge_index_pear.shape}")
        print(f"  y              : {batch.y.shape}")
        print(f"  weight         : {batch.weight.shape}")
        break

    print("=" * 60)
    print("DataLoader smoke-test passed [OK]")
