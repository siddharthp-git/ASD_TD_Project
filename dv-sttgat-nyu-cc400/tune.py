"""
tune.py – Optuna Hyperparameter Tuning for DV-STTGAT (NYU-only, CC400)
=======================================================================
Efficiently searches the hyperparameter space using Bayesian optimisation
(TPE sampler) with epoch-level pruning (MedianPruner).

Key optimisations
-----------------
• Graphs are built ONCE before the Optuna loop and reused across ALL trials.
• BOLD tensor is pre-normalised ONCE (per-subject Z-score is fold-independent).
• Fold splits are pre-computed ONCE (StratifiedKFold with random_state=42).
• DataLoaders are created per-trial (batch_size is a tunable parameter).
• Study is persisted to SQLite → restart safely without losing completed trials.

Search space
------------
  lr            : log-uniform  [1e-5, 5e-3]
  weight_decay  : log-uniform  [1e-4, 0.10]
  focal_gamma   : uniform      [1.0,  4.0]
  focal_alpha   : uniform      [0.40, 0.75]
  mixup_alpha   : uniform      [0.10, 0.80]
  mixup_prob    : uniform      [0.05, 0.50]
  cosine_t0     : int          [10,   40]
  batch_size    : categorical  {8, 16, 32}

Objective
---------
• Maximise mean val Accuracy (Youden's-J optimal threshold from train set,
  identical to the threshold used in train.py — no val-set leakage).

Stopping rules
--------------
• Per-fold early stopping : patience = 100 epochs (no Accuracy improvement).
• Optuna early stopping   : stop study after 100 consecutive non-improving trials.
• Epoch-level pruning     : MedianPruner kills unpromising trials mid-fold.

Usage
-----
    python tune.py
    python tune.py --n_trials 500 --tune_epochs 300 --n_folds 5
    python tune.py --phenotype D:/ABIDE/asd_717participants.csv
"""

import sys
import os
import json
import argparse
import datetime
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold

# ── Project modules ────────────────────────────────────────────────────────────
from data_loader        import load_from_cache, DEFAULT_CACHE_DIR
from graph_construction import build_dual_view_graphs
from cnn                import bold_signals_to_tensor
from dataset            import create_fold_dataloaders
from model              import DVSTTGATModel

# ── Re-use loss + mixup helpers from train.py (no code duplication) ────────────
from train import FocalLoss, manifold_mixup


# ─────────────────────────────────────────────────────────────────────────────
# Fixed architecture constants  (not tuned — keep model comparable to train.py)
# ─────────────────────────────────────────────────────────────────────────────
NODE_FEAT        = 64
GAT_HIDDEN       = 32
GAT_HEADS        = 4
COSINE_T_MULT    = 2
GATE_WD          = 0.20   # weight-decay for view_weights (gate-collapse guard)
ES_PATIENCE      = 100    # per-fold early-stopping patience (epochs)

DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Optuna early-stopping callback
# ─────────────────────────────────────────────────────────────────────────────
class OptunaEarlyStopping:
    """
    Stops the Optuna study when no improvement has been seen for
    `patience` consecutive *completed* (non-pruned) trials.
    """

    def __init__(self, patience: int = 100):
        self.patience   = patience
        self._best      = float("-inf")
        self._no_improv = 0

    def __call__(self, study, trial):
        import optuna
        # Only count finished trials (skip pruned ones)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if study.best_value > self._best:
            self._best      = study.best_value
            self._no_improv = 0
        else:
            self._no_improv += 1

        if self._no_improv >= self.patience:
            print(
                f"\n  ⏹  Optuna early stop: no improvement for "
                f"{self.patience} completed trials  "
                f"(best Accuracy = {self._best:.4f})."
            )
            study.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training for one Optuna trial
# ─────────────────────────────────────────────────────────────────────────────
def _run_tune_fold(
    model:        nn.Module,
    train_loader,
    val_loader,
    *,
    lr:           float,
    weight_decay: float,
    focal_gamma:  float,
    focal_alpha:  float,
    mixup_alpha:  float,
    mixup_prob:   float,
    cosine_t0:    int,
    epochs:       int,
    device:       str,
    trial,          # optuna.Trial  – used for epoch-level pruning
    fold_idx:     int,
) -> float:
    """
    Train DVSTTGATModel for one fold using the supplied hyperparams.
    Reports intermediate Accuracy (Youden's-J threshold from train set) to
    Optuna after every epoch so the MedianPruner can kill bad trials early.

    Returns
    -------
    best_val_acc : float
    """
    model = model.to(device)

    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")

    # Two param-groups: view_weights get a fixed higher WD (gate-collapse guard)
    pg1 = [p for n, p in model.named_parameters() if "view_weights" in n]
    pg2 = [p for n, p in model.named_parameters()
           if "view_weights" not in n and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": pg1, "weight_decay": GATE_WD},
            {"params": pg2, "weight_decay": weight_decay},
        ],
        lr=lr,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_t0, T_mult=COSINE_T_MULT, eta_min=1e-6
    )

    best_acc   = 0.0
    es_counter = 0

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            batch   = batch.to(device)
            targets = batch.y.view(-1, 1).float()
            optimizer.zero_grad()

            if np.random.rand() < mixup_prob:
                z             = model.encode(batch)
                z_mix, y_mix, _ = manifold_mixup(z, targets, mixup_alpha)
                cls_logits    = model.classify(z_mix)
                focal         = criterion(cls_logits, y_mix)
            else:
                z          = model.encode(batch)
                cls_logits = model.classify(z)
                focal      = criterion(cls_logits, targets)

            if hasattr(batch, "weight"):
                w    = batch.weight.view(-1, 1).float()
                w    = w / (w.mean() + 1e-8)
                loss = (focal * w).mean()
            else:
                loss = focal.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step(epoch - 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_targets, all_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch   = batch.to(device)
                targets = batch.y.view(-1, 1).float()
                probs   = torch.sigmoid(model(batch))
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)

        # ── Youden's-J threshold derived from TRAINING data (no val leakage) ──
        train_tgts_list, train_probs_list = [], []
        with torch.no_grad():
            for tr_batch in train_loader:
                tr_batch  = tr_batch.to(device)
                tr_tgts   = tr_batch.y.view(-1, 1).float()
                tr_probs  = torch.sigmoid(model(tr_batch))
                train_tgts_list.extend(tr_tgts.cpu().numpy().flatten())
                train_probs_list.extend(tr_probs.cpu().numpy().flatten())

        train_tgts_arr  = np.array(train_tgts_list)
        train_probs_arr = np.array(train_probs_list)

        try:
            tr_fpr, tr_tpr, tr_thr = roc_curve(train_tgts_arr, train_probs_arr)
            optimal_thr = float(tr_thr[int(np.argmax(tr_tpr - tr_fpr))])
        except ValueError:
            optimal_thr = 0.5

        val_preds = (all_probs >= optimal_thr).astype(float)
        val_acc   = accuracy_score(all_targets, val_preds)

        # ── Report to Optuna (enables epoch-level pruning) ────────────────────
        # Each fold gets its own "step" range; offset by fold_idx * epochs
        report_step = fold_idx * epochs + epoch
        trial.report(val_acc, step=report_step)
        if trial.should_prune():
            import optuna
            raise optuna.exceptions.TrialPruned()

        # ── Per-fold early stopping (monitors accuracy) ───────────────────────
        if val_acc > best_acc:
            best_acc   = val_acc
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= ES_PATIENCE:
            break

    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Save best hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def save_best_params(study, args, save_dir: str):
    """
    Write best hyperparameters to:
      • best_hyperparams.json  (machine-readable, always overwritten)
      • best_hyperparams_<timestamp>.txt  (human-readable, timestamped)

    The TXT file includes a ready-to-run `python train.py ...` command with
    the best parameters pre-filled.
    """
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val   = study.best_value
    best_p     = study.best_params
    best_trial = study.best_trial.number

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(save_dir, "best_hyperparams.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_accuracy":   round(float(best_val), 6),
                "best_trial":      best_trial,
                "n_trials_run":    len(study.trials),
                "timestamp":       ts,
                "fixed": {
                    "node_feat":   NODE_FEAT,
                    "gat_hidden":  GAT_HIDDEN,
                    "gat_heads":   GAT_HEADS,
                    "tune_epochs": args.tune_epochs,
                    "n_folds":     args.n_folds,
                },
                "params": best_p,
            },
            fh,
            indent=2,
        )

    # ── TXT ───────────────────────────────────────────────────────────────────
    sep  = "=" * 70
    dash = "-" * 70
    txt_path = os.path.join(save_dir, f"best_hyperparams_{ts}.txt")

    lines = [
        sep,
        "  DV-STTGAT — Optuna Hyperparameter Tuning Results",
        f"  Timestamp     : {ts}",
        f"  Best mean Acc : {best_val:.6f}   (Trial #{best_trial})",
        f"  Trials run    : {len(study.trials)}",
        sep,
        "",
        "  Tuning Configuration",
        f"  {'n_trials':<20}: {args.n_trials}",
        f"  {'tune_epochs':<20}: {args.tune_epochs}",
        f"  {'n_folds':<20}: {args.n_folds}",
        f"  {'ES patience':<20}: {ES_PATIENCE} epochs / 100 trials",
        "",
        dash,
        "  Best Hyperparameters",
        dash,
    ]
    for k, v in best_p.items():
        if isinstance(v, float):
            lines.append(f"    {k:<20} : {v:.6g}")
        else:
            lines.append(f"    {k:<20} : {v}")

    lines += [
        "",
        dash,
        "  Suggested train.py command (copy-paste ready)",
        dash,
        "    python train.py \\",
        f"        --lr           {best_p['lr']:.2e}          \\",
        f"        --focal_gamma  {best_p['focal_gamma']:.4f} \\",
        f"        --focal_alpha  {best_p['focal_alpha']:.4f} \\",
        f"        --cosine_t0    {best_p['cosine_t0']}       \\",
        f"        --mixup_alpha  {best_p['mixup_alpha']:.4f} \\",
        f"        --mixup_prob   {best_p['mixup_prob']:.4f}  \\",
        f"        --batch_size   {best_p['batch_size']}",
        "",
        "  NOTE: weight_decay must be set manually (not exposed in train.py CLI).",
        f"        Set WEIGHT_DECAY = {best_p['weight_decay']:.6g} in train.py config block.",
        sep,
    ]

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return json_path, txt_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for DV-STTGAT (NYU-only, CC400)"
    )
    parser.add_argument("--phenotype",   type=str, default=DEFAULT_PHENOTYPE,
                        help="Path to phenotype CSV (SUB_ID, DX_GROUP, SITE_ID)")
    parser.add_argument("--cache_dir",   type=str, default=None,
                        help="Path to cc400_bold_cache directory")
    parser.add_argument("--n_trials",    type=int, default=500,
                        help="Max Optuna trials (default 500)")
    parser.add_argument("--tune_epochs", type=int, default=300,
                        help="Max epochs per fold per trial (default 300)")
    parser.add_argument("--n_folds",     type=int, default=5,
                        help="CV folds per trial (default 5)")
    parser.add_argument("--study_name",  type=str, default="dv_sttgat_cc400_tune",
                        help="Optuna study name (used for SQLite persistence)")
    args = parser.parse_args()

    # ── Check Optuna is installed ─────────────────────────────────────────────
    try:
        import optuna
    except ImportError:
        print("\n[ERROR] optuna is not installed.")
        print("  Install with:  pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress verbose output

    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  DV-STTGAT — Optuna Hyperparameter Tuning  (NYU-only | CC400)")
    print(f"  Trials      : {args.n_trials}  (Optuna ES: 100 non-improving trials)")
    print(f"  CV folds    : {args.n_folds}  per trial")
    print(f"  Epochs/fold : {args.tune_epochs}  (per-fold ES patience: {ES_PATIENCE})")
    print(f"  Device      : {device}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – Load data (ONCE)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nStep 1 – Loading NYU BOLD signals from cache …")
    subject_ids, bold_signals, labels, site_labels, roi_centroids, num_sites = \
        load_from_cache(cache_dir=cache_dir, phenotype_csv=args.phenotype)

    N_ROIS = bold_signals[0].shape[1]
    print(f"  ▶ Subjects : {len(labels)}   CC400 ROIs : {N_ROIS}")
    print(f"  ▶ ASD={sum(labels)}   TD={len(labels)-sum(labels)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – Build dual-view graphs ONCE  (reused across all trials)
    # ─────────────────────────────────────────────────────────────────────────
    print(
        "\nStep 2 – Building dual-view graphs (Pearson + Precision) …\n"
        "  ⏳ This runs ONCE and is reused for all 500 trials."
    )
    pear_ei, pear_ew, prec_ei, prec_ew = build_dual_view_graphs(
        bold_signals, verbose=True
    )
    print("  ✅ Graphs built and cached in memory.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – Pre-normalise BOLD tensor ONCE
    # ─────────────────────────────────────────────────────────────────────────
    # Per-subject Z-score (each subject normalised by its own μ/σ per ROI).
    # This is fold-independent → safe to do once before the Optuna loop.
    print("\nStep 3 – Pre-normalising BOLD tensor (per-subject Z-score, once) …")
    bold_tensor_raw = bold_signals_to_tensor(bold_signals)   # (B, N, T_max)
    bold_tensor_norm = bold_tensor_raw.clone()
    for i in range(bold_tensor_norm.shape[0]):
        sig = bold_tensor_norm[i]                            # (N, T_max)
        mu  = sig.mean(dim=1, keepdim=True)
        sd  = sig.std(dim=1, keepdim=True).clamp(min=1e-8)
        bold_tensor_norm[i] = (sig - mu) / sd
    print(f"  ✅ Normalised tensor shape: {bold_tensor_norm.shape}  (stored in RAM, reused)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – Pre-compute fold splits ONCE
    # ─────────────────────────────────────────────────────────────────────────
    print("\nStep 4 – Pre-computing 5-fold CV splits (random_state=42) …")
    labels_arr = np.array(labels)
    skf        = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_splits = [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in skf.split(labels_arr, labels_arr)
    ]
    for fi, (tr, vl) in enumerate(fold_splits):
        print(f"  Fold {fi+1}: train={len(tr)}  val={len(vl)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – Define Optuna objective
    # ─────────────────────────────────────────────────────────────────────────
    def objective(trial):
        # ── Sample hyperparameters ────────────────────────────────────────────
        lr           = trial.suggest_float("lr",           1e-5, 5e-3,  log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.10,  log=True)
        focal_gamma  = trial.suggest_float("focal_gamma",  1.0,  4.0)
        focal_alpha  = trial.suggest_float("focal_alpha",  0.40, 0.75)
        mixup_alpha  = trial.suggest_float("mixup_alpha",  0.10, 0.80)
        mixup_prob   = trial.suggest_float("mixup_prob",   0.05, 0.50)
        cosine_t0    = trial.suggest_int(  "cosine_t0",    10,   40)
        batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])

        fold_accs = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):

            # Build DataLoaders (batch_size is tunable → must recreate per trial)
            train_loader, val_loader = create_fold_dataloaders(
                bold_tensor_norm,          # pre-normalised, never mutated
                pear_ei, pear_ew,
                prec_ei, prec_ew,
                labels,
                train_idx    = train_idx,
                val_idx      = val_idx,
                site_labels  = site_labels,
                batch_size   = batch_size,
                fold         = fold_idx,
            )

            # Fresh model each fold
            model = DVSTTGATModel(
                n_regions         = N_ROIS,
                temporal_out_feat = NODE_FEAT,
                gat_hidden        = GAT_HIDDEN,
                gat_heads         = GAT_HEADS,
                num_sites         = num_sites,
            )

            fold_acc = _run_tune_fold(
                model, train_loader, val_loader,
                lr           = lr,
                weight_decay = weight_decay,
                focal_gamma  = focal_gamma,
                focal_alpha  = focal_alpha,
                mixup_alpha  = mixup_alpha,
                mixup_prob   = mixup_prob,
                cosine_t0    = cosine_t0,
                epochs       = args.tune_epochs,
                device       = device,
                trial        = trial,
                fold_idx     = fold_idx,
            )
            fold_accs.append(fold_acc)

        mean_acc = float(np.mean(fold_accs))

        # ── Per-trial console summary ─────────────────────────────────────────
        print(
            f"  [T{trial.number:>4}] Acc={mean_acc:.4f}"
            f"  lr={lr:.2e}  wd={weight_decay:.2e}"
            f"  fg={focal_gamma:.2f}  fa={focal_alpha:.2f}"
            f"  ma={mixup_alpha:.2f}  mp={mixup_prob:.2f}"
            f"  T0={cosine_t0:>2}  bs={batch_size}"
        )
        return mean_acc

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 – Create / resume Optuna study  (SQLite → survives restarts)
    # ─────────────────────────────────────────────────────────────────────────
    db_path     = os.path.join(_HERE, f"{args.study_name}.db")
    storage_url = f"sqlite:///{db_path}"

    print(f"\nStep 6 – Creating/resuming Optuna study …")
    print(f"  Study name  : {args.study_name}")
    print(f"  Storage     : {db_path}")

    study = optuna.create_study(
        study_name    = args.study_name,
        storage       = storage_url,
        direction     = "maximize",          # maximise mean val Accuracy
        load_if_exists= True,                # resume if already exists
        sampler       = optuna.samplers.TPESampler(seed=42),
        pruner        = optuna.pruners.MedianPruner(
            n_startup_trials = 10,           # don't prune first 10 trials
            n_warmup_steps   = 2,            # skip first 2 folds before pruning
        ),
    )

    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Completed trials already in DB: {n_done}")
    print(f"  Running up to {args.n_trials} total trials …\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 – Run study
    # ─────────────────────────────────────────────────────────────────────────
    es_callback = OptunaEarlyStopping(patience=100)

    study.optimize(
        objective,
        n_trials  = args.n_trials,
        callbacks = [es_callback],
        show_progress_bar = True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 – Report results
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]

    print(f"  Trials completed : {len(completed)}")
    print(f"  Trials pruned    : {len(pruned)}")
    print(f"  Best trial       : #{study.best_trial.number}")
    print(f"  Best mean Acc    : {study.best_value:.6f}")
    print("\n  Best hyperparameters:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k:<20} : {v:.6g}")
        else:
            print(f"    {k:<20} : {v}")

    json_path, txt_path = save_best_params(study, args, save_dir=_HERE)

    print(f"\n  💾 Results saved:")
    print(f"     JSON : {json_path}")
    print(f"     TXT  : {txt_path}")
    print(f"     DB   : {db_path}  ← resume from here if interrupted")
    print("=" * 70)
