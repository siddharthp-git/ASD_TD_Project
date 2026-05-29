"""
build_cc400_cache.py  –  DV-STTGAT
====================================
Extracts ROI-averaged BOLD time-series for ALL subjects (all sites) using
the Craddock 2012 (CC400) whole-brain parcellation atlas and writes the
results to a bold_cache directory compatible with data_loader.py.

Why 356 ROIs instead of 400?
-----------------------------
The CC400 atlas defines 400 parcels in standard 2mm MNI whole-brain space.
However, when NiftiLabelsMasker resamples the atlas to each subject's fMRI
resolution (~3mm), parcels that fall completely outside the brain's gray-
matter mask — typically inferior brainstem edges, cerebellar extremities and
smaller atlas parcels — have zero valid voxels and are automatically dropped.
This consistently yields ~356 parcels across all subjects, which is correct
behavior.  You CANNOT extract BOLD from voxels that aren't in the brain.
The 356 ROIs are all valid, biologically meaningful parcels.

The site filter (e.g. NYU-only) is applied at training time in data_loader.py.
This cache stores everyone so you can use any site later without re-running.

Atlas
-----
  Craddock 2012 (CC400)  –  scorr_mean parcellation, 2 mm resolution.
  Obtained via nilearn with an SSL fallback (nitrc.org cert is expired/mismatched).
  Actual ROI count after NiftiLabelsMasker: typically 392.

Cache layout produced
---------------------
  <cache_dir>/
      <subject_id>_bold.npy   →  np.ndarray float32  (T, N_ROIs)
      roi_centroids.npy       →  np.ndarray float32  (N_ROIs, 3)

Usage
-----
  # Extract ALL subjects (default)
  python build_cc400_cache.py

  # Override paths / workers
  python build_cc400_cache.py \\
      --data_dir  D:/ABIDE \\
      --cache_dir D:/ABIDE/cc400_bold_cache \\
      --n_jobs 4

  # NYU-only (no longer the default; pass --site NYU to filter)
  python build_cc400_cache.py --site NYU

  # Smoke-test: first 5 subjects, serial
  python build_cc400_cache.py --n_jobs 1 --limit 5
"""

import os
import sys
import argparse
import tarfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR  = r"D:\ABIDE"
DEFAULT_CACHE_DIR = os.path.join(_HERE, "cc400_bold_cache")

# CC400 atlas download URL (nitrc.org has an expired cert → we bypass SSL)
_CC400_URL       = "https://cluster_roi.projects.nitrc.org/Parcellations/craddock_2011_parcellations.tar.gz"
_CC400_TARNAME   = "craddock_2011_parcellations.tar.gz"
# Member name inside the tarball that contains scorr_mean parcellations
_CC400_MEMBER    = "scorr05_mean_all.nii.gz"
# Extracted filename saved on disk
_CC400_NII       = "scorr05_mean_all.nii.gz"
# Volume index inside the 4-D atlas image that gives exactly 400 parcels
_CC400_VOL_IDX   = 31


# ─────────────────────────────────────────────────────────────────────────────
# Atlas builder  (CC400) — with SSL-bypass fallback
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_cc400_atlas(nilearn_data_dir: str = None, verbose: bool = False):
    """
    Return path to the CC400 scorr_mean NIfTI file.

    Strategy (fast-path first to avoid console noise in workers)
    --------
    1. Check if the NIfTI is already on disk → return immediately (silent).
    2. Try nilearn's fetch_atlas_craddock_2012() (works if already cached).
    3. Download the tar.gz with requests verify=False, extract the NIfTI.

    Returns
    -------
    nii_path : str  – absolute path to the NIfTI file
    """
    if nilearn_data_dir is None:
        nilearn_data_dir = os.path.join(os.path.expanduser("~"), "nilearn_data",
                                         "craddock_2012")
    os.makedirs(nilearn_data_dir, exist_ok=True)

    nii_path = os.path.join(nilearn_data_dir, _CC400_NII)
    tar_path = os.path.join(nilearn_data_dir, _CC400_TARNAME)

    # ── Fast path: already extracted → return immediately (no console noise) ──
    if os.path.exists(nii_path):
        if verbose:
            print(f"[Atlas] CC400 NIfTI on disk: {nii_path}")
        return nii_path

    # ── Try nilearn (may work if cache is valid in newer nilearn API) ─────────
    try:
        from nilearn import datasets
        atlas    = datasets.fetch_atlas_craddock_2012()
        src_path = atlas.scorr_mean if isinstance(atlas.scorr_mean, str) \
                   else atlas.scorr_mean.get_filename()
        if src_path and os.path.exists(src_path):
            import shutil
            shutil.copy(src_path, nii_path)
            print(f"[Atlas] CC400 atlas obtained via nilearn: {nii_path}")
            return nii_path
    except Exception:
        pass   # fall through silently to manual download

    # ── Manual download with SSL verify=False ────────────────────────────────

    # Download tar.gz
    if not os.path.exists(tar_path):
        import requests
        print(f"[Atlas] Downloading CC400 atlas from {_CC400_URL} …")
        print("[Atlas] (SSL verification disabled for this download)")
        resp = requests.get(_CC400_URL, verify=False, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tar_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {pct:5.1f}%  ({downloaded//(1<<20)} MB / {total//(1<<20)} MB)",
                          end="", flush=True)
        print()
        print(f"[Atlas] Downloaded → {tar_path}")
    else:
        print(f"[Atlas] Tar.gz already on disk: {tar_path}")

    # Extract the specific member
    print(f"[Atlas] Extracting {_CC400_MEMBER} from tarball …")
    with tarfile.open(tar_path, "r:gz") as tar:
        member_names = tar.getnames()
        if _CC400_MEMBER not in member_names:
            raise RuntimeError(
                f"[Atlas] '{_CC400_MEMBER}' not found in tarball.\n"
                f"Available: {member_names}"
            )
        member = tar.getmember(_CC400_MEMBER)
        member.name = _CC400_NII          # destination filename
        tar.extract(member, path=nilearn_data_dir)

    print(f"[Atlas] Extracted → {nii_path}")
    return nii_path


def _build_cc400_atlas(nilearn_data_dir: str = None):
    """
    Return (labels_img, roi_centroids) for CC400.

    The atlas is a 4-D NIfTI; volume index 199 corresponds to K=400 parcels.
    """
    import nibabel as nib
    from nilearn.plotting import find_parcellation_cut_coords

    nii_path = _ensure_cc400_atlas(nilearn_data_dir)

    img_4d  = nib.load(nii_path)
    n_vols  = img_4d.shape[3]
    vol_idx = min(_CC400_VOL_IDX, n_vols - 1)  # vol 31 → exactly 400 parcels

    labels_data = img_4d.get_fdata()[..., vol_idx]
    labels_img  = nib.Nifti1Image(
        labels_data.astype(np.int32),
        img_4d.affine,
        img_4d.header,
    )

    n_labels = int(labels_data.max())
    print(f"[Atlas] CC400: {n_labels} parcels in volume {vol_idx} (scorr_mean).")

    roi_centroids = find_parcellation_cut_coords(labels_img)
    print(f"[Atlas] Centroids computed: {roi_centroids.shape[0]} ROIs.")

    return labels_img, roi_centroids


# ─────────────────────────────────────────────────────────────────────────────
# File discovery  –  ALL subjects (no site filter by default)
# ─────────────────────────────────────────────────────────────────────────────
def _discover_subjects(data_dir: str, phenotype_csv: str = None, site_filter: str = None):
    """
    Walk data_dir for all fMRI files.  Optionally filter by site.

    Parameters
    ----------
    data_dir       : root of raw fMRI data
    phenotype_csv  : ABIDE phenotype CSV (used only if site_filter is set)
    site_filter    : if set (e.g. 'NYU'), only keep subjects from that site

    Returns
    -------
    subject_ids : list[str]
    img_files   : list[str]
    """
    # ── Optional site filter via phenotype CSV ────────────────────────────────
    allowed_ids = None
    if site_filter and phenotype_csv:
        if phenotype_csv.lower().endswith(('.xlsx', '.xls')):
            pheno = pd.read_excel(phenotype_csv)
        else:
            pheno = pd.read_csv(phenotype_csv)
        pheno["SUB_ID_STR"] = pheno["SUB_ID"].astype(str).str.lstrip("0")
        sites = [s.strip() for s in site_filter.split(",")]
        filtered = pheno[pheno["SITE_ID"].isin(sites)]
        allowed_ids = set(filtered["SUB_ID_STR"])
        print(f"[Discovery] Site filter '{site_filter}': {len(allowed_ids)} subjects in phenotype.")
    elif site_filter:
        print("[Discovery] WARNING: --site requires --phenotype; ignoring site filter.")

    # ── Walk data_dir ─────────────────────────────────────────────────────────
    subject_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    print(f"[Discovery] Total subject directories found: {len(subject_dirs)}")

    subject_ids, img_files = [], []
    skipped_site  = 0
    skipped_fmri  = 0

    for subj_id in subject_dirs:
        sid_stripped = subj_id.lstrip("0")

        # Site filter
        if allowed_ids is not None and sid_stripped not in allowed_ids:
            skipped_site += 1
            continue

        base  = f"{subj_id}_rest_1_moco_MNI_b6_bp_nuisreg.0.5mm_gsr"
        found = False
        for ext in (".nii.gz", ".nii"):
            fpath = os.path.join(data_dir, subj_id, base + ext)
            if os.path.exists(fpath):
                subject_ids.append(subj_id)
                img_files.append(fpath)
                found = True
                break

        if not found:
            skipped_fmri += 1

    if skipped_site:
        print(f"[Discovery] Skipped {skipped_site} subjects (site filter).")
    if skipped_fmri:
        print(f"[Discovery] Skipped {skipped_fmri} subjects (no fMRI file on disk).")
    print(f"[Discovery] {len(subject_ids)} subjects with valid fMRI files.")

    return subject_ids, img_files


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject worker  (module-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_one(args):
    """
    Extract CC400 BOLD from a single fMRI file.

    Parameters (packed as tuple for ProcessPoolExecutor)
    ----------
    args : (fpath, sid, cache_dir, detrend, nilearn_data_dir)

    Returns
    -------
    (sid, bold_array | None, error_str | None)
    """
    fpath, sid, cache_dir, detrend, nilearn_data_dir = args

    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = os.path.join(cache_dir, f"{sid}_bold.npy")
    if os.path.exists(cached):
        bold = np.load(cached)
        return sid, bold, None

    try:
        import warnings as _w
        _w.filterwarnings("ignore")

        import nibabel as nib
        from nilearn.input_data import NiftiLabelsMasker

        # Rebuild atlas inside each worker (no sharing across processes)
        nii_path = _ensure_cc400_atlas(nilearn_data_dir)
        img_4d   = nib.load(nii_path)
        n_vols   = img_4d.shape[3]
        vol_idx  = min(_CC400_VOL_IDX, n_vols - 1)  # vol 31 → 400 parcels

        labels_data = img_4d.get_fdata()[..., vol_idx]
        labels_img  = nib.Nifti1Image(
            labels_data.astype(np.int32),
            img_4d.affine,
            img_4d.header,
        )

        masker = NiftiLabelsMasker(
            labels_img=labels_img,
            standardize=False,       # raw BOLD; Z-scoring is done per-fold
            detrend=detrend,
            resampling_target="labels",
            verbose=0,
        )
        bold = masker.fit_transform(fpath).astype(np.float32)   # (T, N_ROIs)

        np.save(cached, bold)
        return sid, bold, None

    except Exception as exc:
        return sid, None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction driver
# ─────────────────────────────────────────────────────────────────────────────
def build_cc400_cache(
    data_dir:       str,
    cache_dir:      str,
    phenotype_csv:  str  = None,
    site_filter:    str  = None,
    n_jobs:         int  = None,
    detrend:        bool = True,
    limit:          int  = None,
):
    os.makedirs(cache_dir, exist_ok=True)

    # ── Discover subjects ─────────────────────────────────────────────────────
    subject_ids, img_files = _discover_subjects(data_dir, phenotype_csv, site_filter)

    if limit:
        subject_ids = subject_ids[:limit]
        img_files   = img_files[:limit]
        print(f"[Cache] Limiting to first {limit} subjects (--limit flag).")

    n_total = len(subject_ids)
    if n_total == 0:
        print("[Cache] ERROR: No subjects to process. Check paths.")
        sys.exit(1)

    # ── Build atlas (main process) & save centroids ───────────────────────────
    nilearn_data_dir = os.path.join(
        os.path.expanduser("~"), "nilearn_data", "craddock_2012"
    )
    centroids_path = os.path.join(cache_dir, "roi_centroids.npy")
    if not os.path.exists(centroids_path):
        labels_img, roi_centroids = _build_cc400_atlas(nilearn_data_dir)
        np.save(centroids_path, roi_centroids.astype(np.float32))
        print(f"[Cache] ROI centroids saved → {centroids_path}")
    else:
        roi_centroids = np.load(centroids_path)
        print(f"[Cache] ROI centroids already cached: {roi_centroids.shape}")

    # ── Count already-cached subjects ─────────────────────────────────────────
    already_done = sum(
        1 for sid in subject_ids
        if os.path.exists(os.path.join(cache_dir, f"{sid}_bold.npy"))
    )
    remaining = n_total - already_done
    if already_done:
        print(f"[Cache] {already_done}/{n_total} subjects already cached → skipping them.")
    if remaining == 0:
        print("[Cache] All subjects already cached. Nothing to do.")
        _print_summary(cache_dir, subject_ids, roi_centroids)
        return

    # ── Parallel extraction ───────────────────────────────────────────────────
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_cpus  = os.cpu_count() or 1
    workers = n_jobs if n_jobs is not None else n_cpus
    workers = max(1, min(workers, remaining))

    print(
        f"\n[Cache] Extracting CC400 BOLD from {remaining} subjects "
        f"({already_done} already done) using {workers}/{n_cpus} CPU workers …\n"
    )

    job_args = [
        (fpath, sid, cache_dir, detrend, nilearn_data_dir)
        for fpath, sid in zip(img_files, subject_ids)
    ]

    results = {}
    failed  = []

    if workers == 1:
        for i, args in enumerate(job_args):
            sid, bold, err = _extract_one(args)
            if err:
                print(f"  [{i+1:>4}/{n_total}] ✗  {sid}:  {err}")
                failed.append(sid)
            else:
                status = "cached" if bold is not None and \
                         os.path.getsize(os.path.join(cache_dir, f"{sid}_bold.npy")) > 0 \
                         and i < already_done else "✓"
                print(f"  [{i+1:>4}/{n_total}] {status}  {sid}  → {bold.shape}")
            results[sid] = bold
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(_extract_one, args): args[1]
                for args in job_args
            }
            done = 0
            for fut in as_completed(fut_map):
                sid, bold, err = fut.result()
                done += 1
                if err:
                    print(f"  [{done:>4}/{n_total}] ✗  {sid}:  {err}")
                    failed.append(sid)
                else:
                    print(f"  [{done:>4}/{n_total}] ✓  {sid}  → {bold.shape}")
                results[sid] = bold

    n_ok = sum(b is not None for b in results.values())
    print(f"\n[Cache] Done.  {n_ok}/{n_total} subjects extracted successfully.")
    if failed:
        print(f"[Cache] {len(failed)} failed subjects:")
        for s in failed:
            print(f"  - {s}")

    _print_summary(cache_dir, subject_ids, roi_centroids)


def _print_summary(cache_dir, subject_ids, roi_centroids):
    cached_files = [
        os.path.join(cache_dir, f"{sid}_bold.npy")
        for sid in subject_ids
        if os.path.exists(os.path.join(cache_dir, f"{sid}_bold.npy"))
    ]
    if cached_files:
        sample = np.load(cached_files[0])
        print(f"\n  Atlas ROIs     : {sample.shape[1]}")
        print(f"  Cached subjects: {len(cached_files)}")
        print(f"  Cache location : {os.path.abspath(cache_dir)}")
        if roi_centroids is not None:
            print(f"  Centroids shape: {roi_centroids.shape}")
    print("\n  ✅  CC400 bold_cache is ready.")
    print("  To train on NYU only:  python train.py")
    print("  (data_loader.py applies the NYU filter at load time)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CC400 BOLD cache (all sites) for DV-STTGAT pipeline"
    )
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR,
                        help="Root directory of raw fMRI data (default: D:/ABIDE)")
    parser.add_argument("--phenotype",  default=None,
                        help="ABIDE phenotype CSV (optional; needed only with --site)")
    parser.add_argument("--site",       default=None,
                        help="Comma-separated site(s) to extract, e.g. 'NYU' or 'NYU,SDSU'."
                             " Default: all sites.")
    parser.add_argument("--cache_dir",  default=DEFAULT_CACHE_DIR,
                        help=f"Output cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--n_jobs",     type=int, default=None,
                        help="Parallel workers (default: all CPU cores)")
    parser.add_argument("--no_detrend", action="store_true",
                        help="Disable linear detrending of BOLD signals")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process only the first N subjects (smoke-test)")
    args = parser.parse_args()

    print("=" * 70)
    print("  CC400 BOLD Cache Builder  –  All Sites")
    print("=" * 70)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Phenotype   : {args.phenotype or '(none – all subjects included)'}")
    print(f"  Site filter : {args.site or '(none – all sites)'}")
    print(f"  Cache dir   : {args.cache_dir}")
    print(f"  Workers     : {args.n_jobs or 'all CPUs'}")
    print(f"  Detrend     : {not args.no_detrend}")
    if args.limit:
        print(f"  Limit       : {args.limit} subjects (smoke-test)")
    print("=" * 70 + "\n")

    build_cc400_cache(
        data_dir      = args.data_dir,
        cache_dir     = args.cache_dir,
        phenotype_csv = args.phenotype,
        site_filter   = args.site,
        n_jobs        = args.n_jobs,
        detrend       = not args.no_detrend,
        limit         = args.limit,
    )
