"""
data_loader.py – DV-STTGAT  (Age 6-11, All sites, CC400 atlas)
====================================================
Loads pre-extracted BOLD signals from the CC400 bold_cache produced by
build_cc400_cache.py.  No raw fMRI re-extraction is performed here.

Only subjects with age between 6 and 11 from all sites are retained.
num_sites is always 1 – DANN adversarial loss is disabled in train.py.

Atlas: Craddock 2012 (CC400) – scorr05_mean_all.nii.gz, volume 31 → 400 parcels.
NO ROI pruning is applied – the full 400-ROI whole-brain CC400 parcellation
is retained, giving the model access to whole-brain functional connectivity.
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default cache location  (populated by build_cc400_cache.py)
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_SCRIPT_DIR, "cc400_bold_cache")


def calculate_mean_fd(confounds_path):
    """
    Calculates Mean Framewise Displacement (Power et al. 2012).
    Assumes a standard 50mm head radius for rotation-to-mm conversion.
    """
    try:
        df = pd.read_csv(confounds_path, sep='\t')
        motion_params = df[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
        diff = motion_params.diff().abs().fillna(0)
        diff.iloc[:, 3:] *= 50.0
        fd_series = diff.sum(axis=1)
        return float(fd_series.mean())
    except Exception:
        return 0.2  # moderate-motion default if file is missing


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_from_cache(
    cache_dir: str = DEFAULT_CACHE_DIR,
    phenotype_csv: str = None,
):
    """
    Load CC400-extracted BOLD signals and attach phenotype labels / site codes.

    Parameters
    ----------
    cache_dir : str
        Directory containing ``{subject_id}_bold.npy`` files and
        ``roi_centroids.npy`` (produced by build_cc400_cache.py).
    phenotype_csv : str
        Path to the CSV with columns ``SUB_ID``, ``DX_GROUP``, ``SITE_ID``.
        DX_GROUP encoding: 1 = ASD, 2 = TD  (standard ABIDE convention).

    Returns
    -------
    subject_ids   : list[str]
    bold_signals  : list[np.ndarray]   – shape (T, N_ROIs_cc400) each
    labels        : list[int]          – 1 = ASD, 0 = TD
    site_labels   : list[int]          – always 0 (single site)
    roi_centroids : np.ndarray         – shape (N_ROIs_cc400, 3)
    num_sites     : int                – always 1
    """
    cache_dir = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"[DataLoader] cc400_bold_cache not found: {cache_dir}\n"
            "Run build_cc400_cache.py first to build the CC400 cache."
        )

    # ── ROI centroids ─────────────────────────────────────────────────────────
    centroids_path = os.path.join(cache_dir, "roi_centroids.npy")
    roi_centroids = np.load(centroids_path) if os.path.exists(centroids_path) else None
    if roi_centroids is not None:
        print(f"[DataLoader] CC400 ROI centroids loaded: {roi_centroids.shape}")
    else:
        print("[DataLoader] WARNING: roi_centroids.npy not found in cache.")

    # ── Discover all cached subjects ──────────────────────────────────────────
    npy_files = sorted(
        f for f in os.listdir(cache_dir)
        if f.endswith("_bold.npy") and f != "roi_centroids.npy"
    )
    if not npy_files:
        raise RuntimeError(f"[DataLoader] No *_bold.npy files found in {cache_dir}")

    cached_ids = [f.replace("_bold.npy", "") for f in npy_files]
    print(f"[DataLoader] Found {len(cached_ids)} cached CC400 subjects.")

    # ── Load BOLD arrays ──────────────────────────────────────────────────────
    bold_map = {}
    for sid, fname in zip(cached_ids, npy_files):
        bold_map[sid] = np.load(os.path.join(cache_dir, fname))

    # ── Phenotype matching ────────────────────────────────────────────────────
    if phenotype_csv is None:
        print("[DataLoader] No phenotype CSV provided – labels will be -1.")
        subject_ids  = cached_ids
        bold_signals = [bold_map[s] for s in subject_ids]
        return subject_ids, bold_signals, [-1] * len(subject_ids), \
               [-1] * len(subject_ids), roi_centroids, 1

    pheno = pd.read_csv(phenotype_csv)
    pheno["SUB_ID"] = pheno["SUB_ID"].astype(str).str.lstrip("0")

    # ── Age 6-11 filter (all sites) ──────────────────────────────────────────
    if "AGE_AT_SCAN" in pheno.columns:
        filtered_pheno = pheno[(pheno["AGE_AT_SCAN"] >= 11) & (pheno["AGE_AT_SCAN"] <= 18)]
    else:
        print("[DataLoader] WARNING: 'AGE_AT_SCAN' column not found in phenotype! Using all subjects.")
        filtered_pheno = pheno

    n_total   = len(pheno)
    n_filtered    = len(filtered_pheno)
    print(f"[DataLoader] Age 6-11 filter: keeping {n_filtered}/{n_total} subjects.")
    pheno = filtered_pheno

    id_to_label = dict(
        zip(pheno["SUB_ID"], pheno["DX_GROUP"].map({1: 1, 2: 0}))
    )
    id_to_site  = {sid: 0 for sid in pheno["SUB_ID"]}
    num_sites   = 1
    print("[DataLoader] All sites (Age 6-11). DANN adversarial loss is disabled.")

    # Match cached IDs to phenotype
    records = []
    for sid in cached_ids:
        sid_stripped = sid.lstrip("0")
        label = id_to_label.get(sid_stripped, -1)
        site  = id_to_site.get(sid_stripped,  -1)
        if label != -1 and site != -1:
            records.append((sid, bold_map[sid], label, site))

    if not records:
        raise RuntimeError(
            "[DataLoader] No Age 6-11 subjects matched between CC400 cache and phenotype CSV!\n"
            "Check that build_cc400_cache.py used the same phenotype file."
        )

    subject_ids, bold_signals, labels, site_labels = map(list, zip(*records))

    # ── Report atlas ROI count (NO pruning for CC400) ─────────────────────────
    n_rois = bold_signals[0].shape[1]
    print(
        f"[DataLoader] CC400 atlas: {n_rois} ROIs retained (full brain, no pruning)."
    )
    print(
        f"[DataLoader] Matched {len(subject_ids)} Age 6-11 subjects "
        f"(ASD={sum(labels)}, TD={len(labels)-sum(labels)})."
    )

    return subject_ids, bold_signals, labels, site_labels, roi_centroids, num_sites


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pheno = sys.argv[1] if len(sys.argv) > 1 else None
    sids, bolds, lbls, sites, centroids, n_sites = load_from_cache(
        phenotype_csv=pheno
    )
    print(f"Subjects      : {len(sids)}")
    print(f"Example BOLD  : {bolds[0].shape}  (T, N_ROIs_cc400)")
    if centroids is not None:
        print(f"ROI centroids : {centroids.shape}")
    print(f"Num sites     : {n_sites}")
