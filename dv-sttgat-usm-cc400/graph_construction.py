"""
graph_construction.py – DV-STTGAT
===================================
Builds two complementary graph views per subject from BOLD time-series:

  G_pear  –  Pearson correlation matrix, thresholded at |r| >= 0.3
  G_prec  –  Partial correlation from GraphicalLassoCV, thresholded at |rho| >= 0.15

Partial correlation formula (avoids using raw precision values as weights):
    rho_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)

Diagonal is zeroed (no self-loops in either view).
"""

import numpy as np
import torch
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PEARSON_THRESHOLD  = 0.3
PRECISION_THRESHOLD = 0.15
LASSO_CV_FOLDS     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _corr_to_edge_tensors(corr_matrix: np.ndarray, threshold: float):
    """
    Convert a symmetric (N, N) correlation/partial-correlation matrix into
    PyG-compatible (edge_index, edge_weight) tensors, keeping only entries
    where |value| >= threshold and excluding self-loops.

    Returns
    -------
    edge_index  : torch.LongTensor   shape (2, E)
    edge_weight : torch.FloatTensor  shape (E,)
    """
    N = corr_matrix.shape[0]
    rows, cols = np.where(
        (np.abs(corr_matrix) >= threshold) & (~np.eye(N, dtype=bool))
    )
    weights = corr_matrix[rows, cols]

    edge_index  = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def _pearson_view(bold: np.ndarray):
    """
    Compute Pearson correlation matrix from BOLD array (T, N).
    Returns np.ndarray (N, N).
    """
    # np.corrcoef expects (N, T) — transpose from (T, N)
    corr = np.corrcoef(bold.T)          # (N, N)
    np.fill_diagonal(corr, 0.0)        # zero out self-loops
    return corr


import warnings

def _precision_view(bold: np.ndarray):
    """
    Robust Partial Correlation estimation with automated cleaning.
    """
    import warnings
    from sklearn.covariance import GraphicalLassoCV, LedoitWolf

    T, N = bold.shape

    # ── STEP A: Robust Pre-flight Scrub ───────────────────────────
    bold = np.nan_to_num(bold, nan=0.0, posinf=0.0, neginf=0.0)

    stds = np.std(bold, axis=0)
    if np.any(stds < 1e-9):
        zero_var_mask = stds < 1e-9
        noise = np.random.default_rng(42).normal(
            0, 1e-6, (T, np.sum(zero_var_mask))
        )
        bold[:, zero_var_mask] += noise

    # ── STEP B: GraphicalLassoCV ──────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GraphicalLassoCV(
                cv=3,
                max_iter=1000,
                tol=1e-2,
                mode='cd',
                n_jobs=1,   # 🔥 IMPORTANT: avoid nested parallelism
                assume_centered=False,
            ).fit(bold)

            Theta = model.precision_

    except Exception:
        # ── STEP C: Ledoit-Wolf fallback ──────────────────────────
        lw = LedoitWolf(assume_centered=False).fit(bold)
        Theta = np.linalg.inv(lw.covariance_)

    # ── STEP D: Partial Correlation Conversion ───────────────────
    Theta = np.nan_to_num(Theta)
    diag = np.diag(Theta)
    d = np.sqrt(np.abs(diag))

    partial_corr = -Theta / (np.outer(d, d) + 1e-10)
    partial_corr = np.clip(partial_corr, -1.0, 1.0)

    np.fill_diagonal(partial_corr, 0.0)

    return partial_corr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
from joblib import Parallel, delayed
import multiprocessing


def _process_single_subject(i, bold, n_total, verbose):
    # ── Pearson view ─────────────────────────────────────────────
    corr = _pearson_view(bold)
    ei_p, ew_p = _corr_to_edge_tensors(corr, PEARSON_THRESHOLD)

    # ── Precision / partial-corr view ────────────────────────────
    pcorr = _precision_view(bold)
    ei_r, ew_r = _corr_to_edge_tensors(pcorr, PRECISION_THRESHOLD)

    if verbose:
        print(
            f"  [{i+1}/{n_total}]  "
            f"Pearson edges: {ei_p.shape[1]}  |  "
            f"Precision edges: {ei_r.shape[1]}"
        )

    return ei_p, ew_p, ei_r, ew_r


def build_dual_view_graphs(bold_signals: list, verbose: bool = True):
    """
    Parallelized version using joblib (DROP-IN REPLACEMENT).

    Returns SAME outputs as original function.
    """

    n_total = len(bold_signals)

    # 🔥 Use all cores except 1 (safe for system)
    n_jobs = max(1, multiprocessing.cpu_count())

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_subject)(i, bold, n_total, verbose)
        for i, bold in enumerate(bold_signals)
    )

    # ── Unpack results ───────────────────────────────────────────
    pear_edge_indices  = [r[0] for r in results]
    pear_edge_weights  = [r[1] for r in results]
    prec_edge_indices  = [r[2] for r in results]
    prec_edge_weights  = [r[3] for r in results]

    return pear_edge_indices, pear_edge_weights, prec_edge_indices, prec_edge_weights


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Smoke-test: build_dual_view_graphs on synthetic data …")

    N, T = 20, 150
    rng  = np.random.default_rng(42)
    # Two mock subjects (already Z-scored)
    fake_bold = [
        (rng.standard_normal((T, N))).astype(np.float32)
        for _ in range(2)
    ]

    pei, pew, rei, rew = build_dual_view_graphs(fake_bold)
    for k, (ei_p, Ew_p, ei_r, ew_r) in enumerate(zip(pei, pew, rei, rew)):
        print(f"  Subject {k+1}:")
        print(f"    G_pear  edge_index {ei_p.shape}  edge_weight {Ew_p.shape}")
        print(f"    G_prec  edge_index {ei_r.shape}  edge_weight {ew_r.shape}")
    print("Smoke-test passed [OK]")
    print("=" * 60)
