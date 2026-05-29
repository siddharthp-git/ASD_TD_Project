"""
ensemble_eval.py – DV-STTGAT  (ADOS 0-10)
==========================================
Loads all 5 fold checkpoints, runs every ADOS 0-10 subject through each model,
soft-votes the probability scores, and reports a single unified
Accuracy, Sensitivity, Specificity and AUC for the full dataset.

Usage
-----
    python ensemble_eval.py
or
    python ensemble_eval.py --phenotype D:/ABIDE/asd_717participants.csv
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from data_loader        import load_from_cache, DEFAULT_CACHE_DIR
from graph_construction import build_dual_view_graphs
from cnn                import bold_signals_to_tensor
from dataset            import create_fold_dataloaders
from model              import DVSTTGATModel

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"
N_ROIS    = 116
NODE_FEAT = 64
GAT_HIDDEN = 32
GAT_HEADS  = 4
N_FOLDS    = 5
BATCH_SIZE = 16


def load_fold_model(ckpt_path: str, num_sites: int, device: str) -> DVSTTGATModel:
    model = DVSTTGATModel(
        n_regions         = N_ROIS,
        temporal_out_feat = NODE_FEAT,
        gat_hidden        = GAT_HIDDEN,
        gat_heads         = GAT_HEADS,
        num_sites         = num_sites,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model: DVSTTGATModel, loader, device: str):
    """Return (targets, probs) as numpy arrays for all subjects in loader."""
    all_targets, all_probs = [], []
    for batch in loader:
        batch   = batch.to(device)
        targets = batch.y.view(-1).float().cpu().numpy()
        logits  = model(batch).view(-1)
        probs   = torch.sigmoid(logits).cpu().numpy()
        all_targets.extend(targets)
        all_probs.extend(probs)
    return np.array(all_targets), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DV-STTGAT Ensemble Evaluator (ADOS 0-10)")
    parser.add_argument("--phenotype",  type=str, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--cache_dir",  type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_folds",    type=int, default=N_FOLDS)
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR

    print("=" * 70)
    print("  DV-STTGAT — Ensemble Evaluation  (ADOS 0-10, 5-Fold Soft Voting)")
    print("=" * 70)

    # ── Pre-flight: check checkpoints ───────────────────────────────────────
    ckpt_paths = [
        os.path.join(_HERE, f"dv_sttgat_fold{k+1}.pt")
        for k in range(args.n_folds)
    ]
    missing = [p for p in ckpt_paths if not os.path.isfile(p)]
    if missing:
        print("\n[ERROR] Missing checkpoints:")
        for p in missing:
            print(f"  {p}")
        print("Run train.py first to generate all fold checkpoints.")
        sys.exit(1)

    # ── Step 1: Load & Z-score BOLD ─────────────────────────────────────────
    print("\nStep 1 – Loading ADOS 0-10 BOLD signals …")
    subject_ids, bold_signals, labels, site_labels, roi_centroids, num_sites = \
        load_from_cache(cache_dir=cache_dir, phenotype_csv=args.phenotype)

    global N_ROIS
    N_ROIS = bold_signals[0].shape[1]
    print(f"  Dynamic N_ROIS derived from BOLD data: {N_ROIS}")

    print("Step 2 – Z-scoring …")
    bold_signals = [
        (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
        for b in bold_signals
    ]

    # ── Step 2: Build graphs ─────────────────────────────────────────────────
    print("Step 3 – Building dual-view graphs …")
    pear_ei, pear_ew, prec_ei, prec_ew = build_dual_view_graphs(
        bold_signals, verbose=True
    )

    # ── Step 3: Padded tensor ────────────────────────────────────────────────
    print("Step 4 – Building BOLD tensor …")
    bold_tensor = bold_signals_to_tensor(bold_signals)
    B = bold_tensor.shape[0]
    print(f"  {B} subjects loaded.")

    # Use all subjects as a single "val" fold (no augmentation)
    all_idx = list(range(B))

    # We only need a loader over the full dataset — use fold loader with
    # empty train and all subjects as val
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    def build_full_loader(batch_size):
        data_list = []
        for i in all_idx:
            data_list.append(Data(
                x               = bold_tensor[i],
                edge_index_pear = pear_ei[i],
                edge_attr_pear  = pear_ew[i],
                edge_index_prec = prec_ei[i],
                edge_attr_prec  = prec_ew[i],
                y               = torch.tensor([labels[i]], dtype=torch.float32),
                site            = torch.tensor([site_labels[i]], dtype=torch.long),
            ))
        return DataLoader(data_list, batch_size=batch_size, shuffle=False)

    full_loader = build_full_loader(args.batch_size)

    # ── Step 4: Run all 5 models, collect probs ──────────────────────────────
    print(f"\nStep 5 – Running {args.n_folds} fold models …\n")
    fold_probs   = []
    true_targets = None

    for k, ckpt in enumerate(ckpt_paths):
        print(f"  Loading Fold {k+1}: {os.path.basename(ckpt)}")
        model = load_fold_model(ckpt, num_sites=num_sites, device=device)
        targets, probs = predict_probs(model, full_loader, device)
        fold_probs.append(probs)
        if true_targets is None:
            true_targets = targets
        print(f"    Fold {k+1} individual AUC: {roc_auc_score(targets, probs):.4f}")

    # ── Step 5: Soft-vote (average probabilities across all folds) ───────────
    ensemble_probs = np.mean(fold_probs, axis=0)    # shape (B,)

    # ── Step 6: Metrics ──────────────────────────────────────────────────────
    ens_auc = roc_auc_score(true_targets, ensemble_probs)

    # Youden's J optimal threshold on the ensemble
    fpr, tpr, thresholds = roc_curve(true_targets, ensemble_probs)
    J_stats           = tpr - fpr
    best_idx          = int(np.argmax(J_stats))
    optimal_threshold = float(thresholds[best_idx])
    preds             = (ensemble_probs >= optimal_threshold).astype(float)

    acc  = accuracy_score(true_targets, preds)
    tp   = float(np.sum((preds == 1) & (true_targets == 1)))
    fn   = float(np.sum((preds == 0) & (true_targets == 1)))
    tn   = float(np.sum((preds == 0) & (true_targets == 0)))
    fp   = float(np.sum((preds == 1) & (true_targets == 0)))
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    n_asd = int(true_targets.sum())
    n_td  = len(true_targets) - n_asd

    print("\n" + "=" * 70)
    print("  ENSEMBLE RESULTS  (5-Model Soft Voting, Full ADOS 0-10 Dataset)")
    print("=" * 70)
    print(f"  Subjects      : {len(true_targets)}  (ASD={n_asd}, TD={n_td})")
    print(f"  Threshold     : {optimal_threshold:.3f}  (Youden's J)")
    print(f"  ROC-AUC       : {ens_auc:.4f}")
    print(f"  Accuracy      : {acc:.4f}  ({int(acc*len(true_targets))}/{len(true_targets)})")
    print(f"  Sensitivity   : {sens:.4f}  (TP rate – ASD detected)")
    print(f"  Specificity   : {spec:.4f}  (TN rate – TD correctly excluded)")
    print("=" * 70)
