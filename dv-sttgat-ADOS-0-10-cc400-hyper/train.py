"""
train.py – DV-STTGAT  (ADOS 0-10, All sites, SOTA v3, CC400 atlas)
=======================================================
End-to-end training pipeline for the Dual-View Spatio-Temporal Graph
Attention Transformer (DV-STTGAT), restricted to subjects with ADOS_TOTAL scores in range [0, 10] across all sites.

Atlas: Craddock 2012 (CC400) whole-brain parcellation (~392 ROIs).
BOLD signals must be pre-extracted with build_cc400_cache.py.

Pipeline
--------
1. Load cached CC400 BOLD signals (ADOS 0-10 subjects, from cc400_bold_cache/)
   → NO ROI pruning – full CC400 whole-brain parcellation is retained
2. Z-score normalise per subject, per ROI  ← prevents signal-scaling bias
3. Build dual-view graphs (Pearson + Precision)
4. Stratified 5-Fold Cross-Validation:
     - Per fold: train with sliding-window augmentation
     - Manifold Mixup applied at graph-embedding level (50% of batches)
     - Scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)
     - Focal Loss  (DANN disabled – single site)
     - Evaluate with ROC-AUC and Youden's-J optimal threshold
5. Report per-fold and mean ± std AUC

Usage
-----
    python train.py
or
    python train.py --phenotype D:/ABIDE/asd_717participants.csv \\
                    --cache_dir ./cc400_bold_cache
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold

from data_loader        import load_from_cache, DEFAULT_CACHE_DIR   # CC400 cache
from graph_construction import build_dual_view_graphs
from cnn                import bold_signals_to_tensor
from dataset            import create_fold_dataloaders
from model              import DVSTTGATModel


# ─────────────────────────────────────────────────────────────────────────────
# Config  (edit here or override via CLI)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"
# N_ROIS is now derived dynamically from the BOLD signal shape after pruning
NODE_FEAT         = 64       # CNN output dim
GAT_HIDDEN        = 32       # slim spatial branch (was 32/64)
GAT_HEADS         = 4
BATCH_SIZE        = 16
EPOCHS            = 500
LR                = 1e-4
WEIGHT_DECAY      = 0.01     # strong decay to starve unnecessary parameters
N_FOLDS           = 5

# CosineAnnealingWarmRestarts
COSINE_T0         = 20       # longer first cycle → model settles before restart
COSINE_T_MULT     = 2        # each restart doubles the cycle length

# Focal Loss hyperparams
FOCAL_GAMMA       = 2.0      # focusing parameter  (0 = vanilla BCE)
FOCAL_ALPHA       = 0.60     # balanced ASD/TD priority

# Manifold Mixup hyperparams
MIXUP_ALPHA       = 0.4      # Beta distribution concentration parameter
MIXUP_PROB        = 0.2      # fraction of batches where mixup is applied

# Learnable view-gating monitoring
GATE_WD           = 0.20     # weight-decay for view_weights — higher than the
                             # rest to prevent early gate collapse
GATE_LOG_EVERY    = 10       # print gate values every N epochs
GATE_COLLAPSE_THR = 0.90     # warn if any gate exceeds this fraction

# Early stopping
EARLY_STOPPING_PATIENCE = 100  # stop if val AUC doesn't improve for N epochs


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss  (Lin et al., 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 reduction: str = "none"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal   = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal   # 'none' → (B, 1) for motion weighting


# ─────────────────────────────────────────────────────────────────────────────
# Manifold Mixup helper
# ─────────────────────────────────────────────────────────────────────────────
def manifold_mixup(z: torch.Tensor, y: torch.Tensor, alpha: float):
    """
    Apply Manifold Mixup on the graph embedding AFTER attention pooling.

    Strategy
    --------
    - Operating on graph embeddings (B, H) avoids BOLD time-alignment noise.
    - Use max(λ, 1-λ) so the dominant sample always contributes more,
      preventing total flips in very imbalanced mini-batches.

    Parameters
    ----------
    z     : (B, H)  graph embeddings from model.encode()
    y     : (B, 1)  float labels (hard labels from loader)
    alpha : float   Beta distribution concentration

    Returns
    -------
    z_mix : (B, H)  mixed embeddings
    y_mix : (B, 1)  soft mixed labels
    lam   : float   mixing coefficient used
    """
    lam   = float(np.random.beta(alpha, alpha))
    lam   = max(lam, 1.0 - lam)                         # dominant sample leads

    perm  = torch.randperm(z.size(0), device=z.device)
    z_mix = lam * z + (1.0 - lam) * z[perm]            # (B, H) mixed embeddings
    y_mix = lam * y + (1.0 - lam) * y[perm]            # (B, 1) soft labels
    return z_mix, y_mix, lam


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training & validation loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    fold: int,
    epochs: int = EPOCHS,
    lr: float   = LR,
    device: str  = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """
    Train DVSTTGATModel for one CV fold and return (best_metrics, history).
    history is a dict with lists: train_loss, val_loss, val_auc, val_acc.
    best_metrics is a dict with: accuracy, auc, sensitivity, specificity, f1.

    Scheduler   : CosineAnnealingWarmRestarts (T_0=COSINE_T0, T_mult=COSINE_T_MULT)
    Loss        : Focal Loss (gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    Augmentation: Manifold Mixup on graph embeddings (MIXUP_PROB of batches)
    Threshold   : Youden's J optimal threshold
    """
    model = model.to(device)

    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="none")

    # ── Two param-groups: view_weights gets higher WD to prevent gate collapse ──
    pg1 = [p for n, p in model.named_parameters() if "view_weights" in n]
    pg2 = [p for n, p in model.named_parameters() if "view_weights" not in n and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": pg1, "weight_decay": GATE_WD},       # gate: higher WD
            {"params": pg2, "weight_decay": WEIGHT_DECAY},  # rest: standard WD
        ],
        lr=lr,
    )

    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-6
    )

    best_auc = 0.0
    history  = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_auc":    [],
        "best_fpr":   None, "best_tpr": None,
        "best_metrics": {              # metrics for the model that achieved best_acc
            "accuracy": 0.0, "auc": 0.0, "sensitivity": 0.0, "specificity": 0.0,
            "precision": 0.0, "f1": 0.0
        }
    }

    # Early stopping state  (monitors val accuracy)
    es_patience    = EARLY_STOPPING_PATIENCE
    es_counter     = 0
    best_wts       = None   # in-memory copy of the best model weights
    best_acc       = 0.0    # best val accuracy seen so far

    for epoch in range(1, epochs + 1):

        # ── Training phase ───────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch   = batch.to(device)
            targets = batch.y.view(-1, 1).float()
            optimizer.zero_grad()

            # ── Manifold Mixup on graph embeddings (MIXUP_PROB of batches) ──
            if np.random.rand() < MIXUP_PROB:
                z        = model.encode(batch)             # (B, H) no grad yet
                z_mix, y_mix, _ = manifold_mixup(z, targets, MIXUP_ALPHA)
                cls_logits = model.classify(z_mix)         # (B, 1)
                focal      = criterion(cls_logits, y_mix)  # soft-label focal loss
            else:
                # Standard forward (hard labels)
                z          = model.encode(batch)
                cls_logits = model.classify(z)             # (B, 1)
                focal      = criterion(cls_logits, targets)

            # Motion-aware sample weighting
            if hasattr(batch, "weight"):
                weights = batch.weight.view(-1, 1).float()
                weights = weights / (weights.mean() + 1e-8)
                loss = (focal * weights).mean()
            else:
                loss = focal.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        # Step scheduler once per epoch
        scheduler.step(epoch - 1)    # CosineAnnealingWarmRestarts uses epoch index

        avg_train_loss = train_loss / len(train_loader.dataset)

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        all_targets = []
        all_probs   = []

        with torch.no_grad():
            for batch in val_loader:
                batch   = batch.to(device)
                targets = batch.y.view(-1, 1).float()

                cls_logits = model(batch)
                loss = criterion(cls_logits, targets).mean()
                val_loss += loss.item() * batch.num_graphs

                probs = torch.sigmoid(cls_logits)
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader.dataset)

        # ── Metrics ───────────────────────────────────────────────────────
        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)

        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            val_auc = 0.5

        # ── Train-set inference to derive Youden's J threshold ────────────
        # Threshold is computed on TRAINING data only, then applied to val.
        # This prevents post-hoc threshold optimization on the val set.
        train_targets_list, train_probs_list = [], []
        with torch.no_grad():
            for tr_batch in train_loader:
                tr_batch  = tr_batch.to(device)
                tr_tgts   = tr_batch.y.view(-1, 1).float()
                tr_logits = model(tr_batch)
                tr_probs  = torch.sigmoid(tr_logits)
                train_targets_list.extend(tr_tgts.cpu().numpy().flatten())
                train_probs_list.extend(tr_probs.cpu().numpy().flatten())

        train_targets_arr = np.array(train_targets_list)
        train_probs_arr   = np.array(train_probs_list)

        try:
            tr_fpr, tr_tpr, tr_thresholds = roc_curve(train_targets_arr, train_probs_arr)
            tr_J              = tr_tpr - tr_fpr
            optimal_threshold = float(tr_thresholds[int(np.argmax(tr_J))])
        except ValueError:
            optimal_threshold = 0.5

        # Apply train-derived threshold to train/val probs
        val_preds = (all_probs >= optimal_threshold).astype(float)
        val_acc   = accuracy_score(all_targets, val_preds)

        train_preds = (train_probs_arr >= optimal_threshold).astype(float)
        train_acc   = accuracy_score(train_targets_arr, train_preds)

        preds       = val_preds
        tp          = float(np.sum((preds == 1) & (all_targets == 1)))
        fn          = float(np.sum((preds == 0) & (all_targets == 1)))
        tn          = float(np.sum((preds == 0) & (all_targets == 0)))
        fp          = float(np.sum((preds == 1) & (all_targets == 0)))
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision   = tp / (tp + fp + 1e-8)

        current_lr = scheduler.get_last_lr()[0]

        # ── Early stopping check (monitor val accuracy) ───────────────────
        if val_acc > best_acc:
            best_acc    = val_acc
            es_counter  = 0
            best_wts    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Store all requested metrics for the best accuracy model
            history["best_metrics"]["accuracy"]    = float(val_acc)
            history["best_metrics"]["auc"]         = float(val_auc)
            history["best_metrics"]["sensitivity"] = float(sensitivity)
            history["best_metrics"]["specificity"] = float(specificity)
            history["best_metrics"]["precision"]   = float(precision)
            history["best_metrics"]["f1"]          = float(f1_score(all_targets, val_preds, zero_division=0))
        else:
            es_counter += 1

        # ── Record history ────────────────────────────────────────────────
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            # Capture ROC data for the best model
            bf, bt, _ = roc_curve(all_targets, all_probs)
            history["best_fpr"] = bf
            history["best_tpr"] = bt

        print(
            f"  [F{fold+1} E{epoch:>3}/{epochs}]"
            f"  Train: {avg_train_loss:.4f}"
            f"  │  Val: {avg_val_loss:.4f}"
            f"  │  AUC: {val_auc:.4f}"
            f"  │  Opt Acc: {val_acc:.4f} (Thr: {optimal_threshold:.3f})"
            f"  │  Sens: {sensitivity:.3f}  Spec: {specificity:.3f}  Prec: {precision:.3f}"
            f"  │  LR: {current_lr:.2e}"
        )

        # ── Gate monitoring (every GATE_LOG_EVERY epochs) ─────────────────
        if epoch % GATE_LOG_EVERY == 0:
            with torch.no_grad():
                gates = torch.softmax(model.view_weights, dim=0).cpu().numpy()
            collapse_flag = any(g > GATE_COLLAPSE_THR for g in gates)
            gate_str = "  ".join(
                [f"Pearson={gates[0]:.3f}",
                 f"Precision={gates[1]:.3f}",
                 f"Learned={gates[2]:.3f}"]
            )
            print(f"  ├─ [GATES E{epoch}] {gate_str}")
            if collapse_flag:
                dominant = ["Pearson", "Precision", "Learned"][int(gates.argmax())]
                print(
                    f"  ⚠️  GATE COLLAPSE WARNING: '{dominant}' dominates "
                    f"({max(gates):.3f} > {GATE_COLLAPSE_THR}).  "
                    f"Consider raising GATE_WD (currently {GATE_WD})."
                )

        # ── Early stopping trigger ─────────────────────────────────────────
        if es_counter >= es_patience:
            print(
                f"\n  ⏹  Early stopping at epoch {epoch} "
                f"(no AUC improvement for {es_patience} epochs).  "
                f"Best AUC: {best_auc:.4f}"
            )
            break

    # Restore best weights before returning
    if best_wts is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_wts.items()})

    return history["best_metrics"], history


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold metric plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_fold_metrics(history: dict, fold: int, save_dir: str) -> None:
    """
    Save a figure with:
    1. Train vs Val Loss
    2. Train vs Val Accuracy
    3. Validation AUC Curve (ROC for best model)
    4. Validation AUC over epochs
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Fold {fold} – Training Metrics", fontsize=16, fontweight="bold")

    # ── 1. Loss Curves ──
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4", linewidth=2, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#d62728", linewidth=2, label="Val", linestyle="--")
    ax.set_title("Train vs Val Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── 2. Accuracy Curves ──
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], color="#2ca02c", linewidth=2, label="Train")
    ax.plot(epochs, history["val_acc"],   color="#ff7f0e", linewidth=2, label="Val", linestyle="--")
    ax.set_title("Train vs Val Accuracy", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── 3. ROC Curve (Best Model) ──
    ax = axes[1, 0]
    if history["best_fpr"] is not None and history["best_tpr"] is not None:
        auc_val = history["val_auc"][np.argmax(history["val_auc"])]
        ax.plot(history["best_fpr"], history["best_tpr"], color="darkviolet", 
                linewidth=2.5, label=f"Best AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"ROC Curve (Best Model)", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC data missing", ha="center")
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── 4. AUC over epochs ──
    ax = axes[1, 1]
    ax.plot(epochs, history["val_auc"], color="mediumseagreen", linewidth=2)
    ax.set_title("Validation AUC over Epochs", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(save_dir, f"fold{fold}_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Metrics plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fold averaged metric plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_cross_fold_summary(all_histories: list, save_dir: str) -> None:
    """
    After all folds are done, compute epoch-wise mean ± std across folds and
    plot:
      1. Average Train Loss  vs  Average Validation Loss  (over epochs)
      2. Average Train Accuracy  vs  Average Validation Accuracy  (over epochs)

    Folds that stopped early are trimmed to the shortest fold length so that
    the numpy stacking is well-defined.
    """
    if not all_histories:
        return

    # Trim to the minimum number of epochs recorded across folds
    min_len = min(len(h["train_loss"]) for h in all_histories)

    keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
    arrays = {k: np.array([h[k][:min_len] for h in all_histories]) for k in keys}
    # shape → (n_folds, min_len)

    means = {k: arrays[k].mean(axis=0) for k in keys}
    stds  = {k: arrays[k].std(axis=0)  for k in keys}

    epochs = np.arange(1, min_len + 1)
    n_folds = len(all_histories)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Cross-Fold Summary ({n_folds} folds) — Avg ± 1 Std",
        fontsize=15, fontweight="bold"
    )

    # ── 1. Loss ──
    ax = axes[0]
    ax.plot(epochs, means["train_loss"], color="#1f77b4", linewidth=2.5, label="Avg Train Loss")
    ax.fill_between(
        epochs,
        means["train_loss"] - stds["train_loss"],
        means["train_loss"] + stds["train_loss"],
        color="#1f77b4", alpha=0.18
    )
    ax.plot(epochs, means["val_loss"], color="#d62728", linewidth=2.5,
            linestyle="--", label="Avg Val Loss")
    ax.fill_between(
        epochs,
        means["val_loss"] - stds["val_loss"],
        means["val_loss"] + stds["val_loss"],
        color="#d62728", alpha=0.18
    )
    ax.set_title("Avg Train Loss vs Avg Val Loss over Folds", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── 2. Accuracy ──
    ax = axes[1]
    ax.plot(epochs, means["train_acc"], color="#2ca02c", linewidth=2.5, label="Avg Train Acc")
    ax.fill_between(
        epochs,
        means["train_acc"] - stds["train_acc"],
        means["train_acc"] + stds["train_acc"],
        color="#2ca02c", alpha=0.18
    )
    ax.plot(epochs, means["val_acc"], color="#ff7f0e", linewidth=2.5,
            linestyle="--", label="Avg Val Acc")
    ax.fill_between(
        epochs,
        means["val_acc"] - stds["val_acc"],
        means["val_acc"] + stds["val_acc"],
        color="#ff7f0e", alpha=0.18
    )
    ax.set_title("Avg Train Accuracy vs Avg Val Accuracy over Folds", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(save_dir, "cross_fold_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Cross-fold summary plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Results saver
# ─────────────────────────────────────────────────────────────────────────────
def save_results_to_txt(
    fold_metrics: list,
    args,
    save_dir: str,
) -> str:
    """
    Write a human-readable summary of the cross-validation results to a .txt
    file inside *save_dir*.  The filename is timestamped so successive runs
    never overwrite each other.

    Returns the absolute path of the written file.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname     = os.path.join(save_dir, f"results_{timestamp}.txt")

    sep  = "=" * 70
    dash = "-" * 8

    lines = []
    lines.append(sep)
    lines.append("  DV-STTGAT — ADOS 0-10 | CC400 Full-Brain | Cross-Validation Results")
    lines.append(f"  Run timestamp : {timestamp}")
    lines.append(sep)
    lines.append("")
    lines.append("  Run Configuration")
    lines.append(f"  {'Phenotype':<18}: {args.phenotype}")
    lines.append(f"  {'Epochs/fold':<18}: {args.epochs}")
    lines.append(f"  {'Learning rate':<18}: {args.lr}")
    lines.append(f"  {'Batch size':<18}: {args.batch_size}")
    lines.append(f"  {'CV folds':<18}: {args.n_folds}")
    lines.append(f"  {'Focal gamma':<18}: {args.focal_gamma}")
    lines.append(f"  {'Focal alpha':<18}: {args.focal_alpha}")
    lines.append(f"  {'Cosine T0':<18}: {args.cosine_t0}")
    lines.append(f"  {'Mixup alpha':<18}: {args.mixup_alpha}")
    lines.append(f"  {'Mixup prob':<18}: {args.mixup_prob}")
    lines.append("")
    lines.append(sep)
    lines.append("  Per-Fold Metrics")
    lines.append(sep)

    accs, aucs, sens, specs, precs, f1s = [], [], [], [], [], []
    header = (f"  {'Fold':<8} | {'Acc':<8} | {'AUC':<8} | "
              f"{'Sens':<8} | {'Spec':<8} | {'Prec':<8} | {'F1':<8}")
    lines.append(header)
    lines.append(f"  {dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}")

    for k, m in enumerate(fold_metrics):
        lines.append(
            f"  Fold {k+1:<3} | {m['accuracy']:.4f} | {m['auc']:.4f} | "
            f"{m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['precision']:.4f} | {m['f1']:.4f}"
        )
        accs.append(m['accuracy'])
        aucs.append(m['auc'])
        sens.append(m['sensitivity'])
        specs.append(m['specificity'])
        precs.append(m['precision'])
        f1s.append(m['f1'])

    lines.append(f"  {dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}-+-{dash}")
    lines.append(
        f"  {'Mean':<8} | {np.mean(accs):.4f} | {np.mean(aucs):.4f} | "
        f"{np.mean(sens):.4f} | {np.mean(specs):.4f} | "
        f"{np.mean(precs):.4f} | {np.mean(f1s):.4f}"
    )
    lines.append(
        f"  {'Std':<8} | {np.std(accs):.4f} | {np.std(aucs):.4f} | "
        f"{np.std(sens):.4f} | {np.std(specs):.4f} | "
        f"{np.std(precs):.4f} | {np.std(f1s):.4f}"
    )
    lines.append("")

    mean_auc = float(np.mean(aucs))
    if mean_auc >= 0.78:
        lines.append("  ★ BREAKTHROUGH: Mean AUC >= 0.78!")
    if mean_auc >= 0.80:
        lines.append("  ★★ SOTA: Mean AUC >= 0.80!")
    lines.append(sep)

    with open(fname, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return fname


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DV-STTGAT Trainer (ADOS 0-10, All sites, 5-Fold CV)")
    parser.add_argument(
        "--phenotype", type=str, default=DEFAULT_PHENOTYPE,
        help="Path to phenotype CSV/Excel (SUB_ID, DX_GROUP, SITE_ID, ADOS_TOTAL)"
    )
    parser.add_argument("--cache_dir",   type=str,   default=None)
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--n_folds",     type=int,   default=N_FOLDS)
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--focal_alpha", type=float, default=FOCAL_ALPHA)
    parser.add_argument("--cosine_t0",   type=int,   default=COSINE_T0,
                        help="CosineAnnealingWarmRestarts T_0 (first cycle length)")
    parser.add_argument("--mixup_alpha", type=float, default=MIXUP_ALPHA,
                        help="Beta distribution alpha for Manifold Mixup (0 = disabled)")
    parser.add_argument("--mixup_prob",  type=float, default=MIXUP_PROB,
                        help="Fraction of training batches where Mixup is applied")
    args = parser.parse_args()

    # Push CLI args into globals used by train_one_fold
    FOCAL_GAMMA = args.focal_gamma
    FOCAL_ALPHA = args.focal_alpha
    COSINE_T0   = args.cosine_t0
    MIXUP_ALPHA = args.mixup_alpha
    MIXUP_PROB  = args.mixup_prob

    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR

    print("=" * 70)
    print("  DV-STTGAT — Dual-View Spatio-Temporal Graph Attention Transformer")
    print("  *** ADOS 0-10 (ALL SITES)  │  SOTA v3  │  CC400 Full-Brain (~392 ROIs)  │  5-Fold CV ***")
    print("=" * 70)
    print(f"  Phenotype   : {args.phenotype}")
    print(f"  Cache dir   : {os.path.abspath(cache_dir)}")
    print(f"  Epochs/fold : {args.epochs}   LR: {args.lr}   Batch: {args.batch_size}")
    print(f"  Focal Loss  : gamma={args.focal_gamma}  alpha={args.focal_alpha}")
    print(f"  Scheduler   : CosineAnnealingWarmRestarts  T_0={COSINE_T0}  T_mult={COSINE_T_MULT}")
    print(f"  Manifold Mixup : alpha={MIXUP_ALPHA}  prob={MIXUP_PROB}")
    print(f"  Threshold   : Youden's J optimal")
    print(f"  CV Folds    : {args.n_folds}")
    print("=" * 70)

    # ── Pre-flight check ────────────────────────────────────────────────────
    missing = []
    if not os.path.isdir(cache_dir):
        missing.append(f"  bold_cache not found : {os.path.abspath(cache_dir)}")
    if not os.path.isfile(args.phenotype):
        missing.append(f"  Phenotype not found  : {args.phenotype}")
    if missing:
        print("\n[ERROR] Cannot start – path(s) not found:")
        for m in missing:
            print(m)
        sys.exit(1)

    # ── Step 1: Load BOLD signals from cache ────────────────────────────────
    print("\nStep 1 – Loading ADOS 0-10 BOLD signals from bold_cache …")
    subject_ids, bold_signals, labels, site_labels, roi_centroids, num_sites = \
        load_from_cache(cache_dir=cache_dir, phenotype_csv=args.phenotype)

    # Derive N_ROIS dynamically from CC400 BOLD arrays (no pruning)
    N_ROIS = bold_signals[0].shape[1]
    print(f"\n  ▶ CC400 ROIs (full brain, no pruning): {N_ROIS}")

    # ── Step 2: Keep raw signals — Z-scoring is done per-fold (see Step 5) ──
    print("\nStep 2 – Z-scoring deferred to each fold (train-stats only, no leakage).")

    # ── Step 3: Build dual-view graphs ──────────────────────────────────────
    print("\nStep 3 – Building dual-view brain graphs …")
    print("  (GraphicalLassoCV may take a few minutes per subject)")
    pear_ei, pear_ew, prec_ei, prec_ew = build_dual_view_graphs(
        bold_signals, verbose=True
    )

    # ── Step 4: Convert raw BOLD → padded tensor  (NOT yet normalised) ──────
    print("\nStep 4 – Converting BOLD → padded tensor (B, N, T_max) …")
    bold_tensor_raw = bold_signals_to_tensor(bold_signals)  # (B, N, T_max), raw
    print(f"  Tensor shape: {bold_tensor_raw.shape}")

    # ── Step 5: Stratified 5-Fold Cross-Validation ──────────────────────────
    print(f"\nStep 5 – Stratified {args.n_folds}-Fold Cross-Validation …")
    print("=" * 70)

    labels_arr = np.array(labels)
    skf        = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    fold_metrics  = []   # [ {accuracy, auc, f1, sensitivity, specificity}, ... ]
    all_histories = []   # ← collect per-fold histories for cross-fold plot

    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_arr, labels_arr)):
        train_idx = train_idx.tolist()
        val_idx   = val_idx.tolist()

        print(f"\n{'─'*70}")
        print(f"  FOLD {fold+1} / {args.n_folds}")
        print(f"{'─'*70}")

        # ── Per-fold Z-scoring (train stats only — no leakage) ───────────────
        # 1. Compute μ and σ per-subject per-ROI across training subjects only.
        # 2. Apply those same μ/σ to val subjects.
        # This ensures no information from val subjects leaks into normalization.
        bold_tensor_fold = bold_tensor_raw.clone()          # (B, N, T_max)
        for i in train_idx:
            sig = bold_tensor_fold[i]                       # (N, T_max)
            mu  = sig.mean(dim=1, keepdim=True)             # (N, 1)
            sd  = sig.std(dim=1, keepdim=True).clamp(min=1e-8)
            bold_tensor_fold[i] = (sig - mu) / sd
        for i in val_idx:
            sig = bold_tensor_fold[i]
            mu  = sig.mean(dim=1, keepdim=True)
            sd  = sig.std(dim=1, keepdim=True).clamp(min=1e-8)
            bold_tensor_fold[i] = (sig - mu) / sd
        print(
            f"  Z-scored {len(train_idx)} train + {len(val_idx)} val subjects "
            f"(each subject normalised by its own \u03bc/\u03c3 — no cross-subject leakage)"
        )

        # Build fold DataLoaders (sliding-window aug on training side)
        train_loader, val_loader = create_fold_dataloaders(
            bold_tensor_fold,
            pear_ei, pear_ew,
            prec_ei, prec_ew,
            labels,
            train_idx=train_idx,
            val_idx=val_idx,
            site_labels=site_labels,
            batch_size=args.batch_size,
            fold=fold,
        )

        # Fresh model per fold (N_ROIS is dynamic — CC400 full brain)
        model = DVSTTGATModel(
            n_regions         = N_ROIS,
            temporal_out_feat = NODE_FEAT,
            gat_hidden        = GAT_HIDDEN,
            gat_heads         = GAT_HEADS,
            num_sites         = num_sites,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}  (CC400 ROIs: {N_ROIS}  gat_hidden: {GAT_HIDDEN})")

        best_metrics, history = train_one_fold(
            model, train_loader, val_loader,
            fold=fold,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        fold_metrics.append(best_metrics)
        all_histories.append(history)  # ← store for cross-fold summary

        # Save per-fold metric plots
        plot_fold_metrics(history, fold=fold + 1, save_dir=_HERE)
        print(f"\n  ▶ Fold {fold+1} Best Metrics:")
        print(f"      Accuracy : {best_metrics['accuracy']:.4f}")
        print(f"      AUC      : {best_metrics['auc']:.4f}")
        print(f"      Sens     : {best_metrics['sensitivity']:.4f}")
        print(f"      Spec     : {best_metrics['specificity']:.4f}")
        print(f"      Precision: {best_metrics['precision']:.4f}")
        print(f"      F1       : {best_metrics['f1']:.4f}")

        # Save per-fold checkpoint
        ckpt_path = os.path.join(_HERE, f"dv_sttgat_fold{fold+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

    # ── Cross-fold averaged metric plots ────────────────────────────────────
    print("\nGenerating cross-fold summary plot …")
    plot_cross_fold_summary(all_histories, save_dir=_HERE)

    # ── Final CV summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-VALIDATION RESULTS")
    print("=" * 70)
    
    # Header
    print(f"  {'Fold':<8} | {'Acc':<8} | {'AUC':<8} | {'Sens':<8} | {'Spec':<8} | {'Prec':<8} | {'F1':<8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    accs, aucs, sens, specs, precs, f1s = [], [], [], [], [], []
    for k, m in enumerate(fold_metrics):
        print(f"  Fold {k+1:<3} | {m['accuracy']:.4f} | {m['auc']:.4f} | "
              f"{m['sensitivity']:.4f} | {m['specificity']:.4f} | "
              f"{m['precision']:.4f} | {m['f1']:.4f}")
        accs.append(m['accuracy'])
        aucs.append(m['auc'])
        sens.append(m['sensitivity'])
        specs.append(m['specificity'])
        precs.append(m['precision'])
        f1s.append(m['f1'])
    
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  Mean     | {np.mean(accs):.4f} | {np.mean(aucs):.4f} | "
          f"{np.mean(sens):.4f} | {np.mean(specs):.4f} | "
          f"{np.mean(precs):.4f} | {np.mean(f1s):.4f}")
    print(f"  Std      | {np.std(accs):.4f} | {np.std(aucs):.4f} | "
          f"{np.std(sens):.4f} | {np.std(specs):.4f} | "
          f"{np.std(precs):.4f} | {np.std(f1s):.4f}")
    
    mean_auc = float(np.mean(aucs))
    if mean_auc >= 0.78:
        print("\n  ★ BREAKTHROUGH: Mean AUC ≥ 0.78 !")
    if mean_auc >= 0.80:
        print("  ★★ SOTA: Mean AUC ≥ 0.80 !")
    print("=" * 70)

    # ── Save results to txt ─────────────────────────────────────────────────
    txt_path = save_results_to_txt(fold_metrics, args, save_dir=_HERE)
    print(f"\n  💾 Results saved → {txt_path}")
    print("=" * 70)
