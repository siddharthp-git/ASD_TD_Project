# DV-STTGAT — NYU-Only, SOTA v3

**Dual-View Spatio-Temporal Graph Attention Transformer** for ASD vs. TD classification from resting-state fMRI (rs-fMRI), restricted to the NYU site of the ABIDE dataset.

---

## Overview

This model classifies subjects as **ASD** (Autism Spectrum Disorder) or **TD** (Typical Development) from BOLD time-series extracted using the Harvard-Oxford atlas. It combines a **temporal CNN** for per-ROI feature extraction with a **triple-branch GATv2** that fuses three complementary graph views of functional connectivity.

---

## Architecture

```
BOLD signals (N×T)
      │
      ▼
MultiScaleInceptionCNN          ← Temporal branch: captures BOLD dynamics
      │  (B×N, F=64)
      ▼
Triple-Branch GATv2             ← Spatial branch: three graph views
  ├── Path A: Pearson graph     ← Static functional connectivity
  ├── Path B: Precision graph   ← Partial correlations (GraphicalLassoCV)
  └── Path C: LearnableGraph    ← Cosine-similarity, top-k sparsified, SLAMP-style
      │  Fused via learnable softmax gating (view_weights)
      ▼
Attention Global Pooling        ← Per-ROI importance scores → weighted sum
      │  (B, H=16)
      ▼
Classifier (Linear→BN→ReLU→Dropout→Linear)
      │  (B, 1)
      ▼
  Binary logit (ASD=1, TD=0)
```

---

## SOTA Upgrades (v3)

| # | Upgrade | Where |
|---|---|---|
| 1 | Social-Brain ROI Pruning (111 → 28 ROIs) | `data_loader.py` |
| 2 | Learnable Graph (cosine-sim + top-k) | `model.py` – `LearnableGraph` |
| 3 | Learnable View-Gating (3-view softmax) | `model.py` – `view_weights` |
| 4 | Slim Spatial Branch (`gat_hidden=16`) + Temporal MaxPool | `model.py`, `cnn.py` |
| 5 | Manifold Mixup on graph embeddings | `train.py` – `manifold_mixup()` |

---

## Training Pipeline

### Step 1 — Load BOLD Signals
- Load cached `.npy` BOLD arrays from `gnn_cnn_harmonised/bold_cache`
- NYU subjects only (single-site; DANN/GRL removed)
- Social-brain ROI pruning applied in `data_loader.py`

### Step 2 — Z-Score Normalisation *(per-fold, no leakage)*
- Each subject is Z-scored per-ROI using its **own** mean and std (self-normalisation)
- Normalisation is applied **inside each fold** after the train/val split — val subjects never influence train statistics

### Step 3 — Dual-View Graph Construction
- **Pearson graph**: thresholded Pearson correlation matrix
- **Precision graph**: sparse inverse covariance via `GraphicalLassoCV`
- Both computed on the (raw) BOLD signals before normalisation

### Step 4 — Padded BOLD Tensor
- Variable-length BOLD arrays are padded to `T_max` and packed into a `(B, N, T_max)` tensor

### Step 5 — Stratified 5-Fold Cross-Validation
For each fold:
1. **Sliding-window augmentation** on training subjects only (Window A: `[0:150]`, Window B: `[25:175]`) — doubles effective training set; split is **subject-wise** to prevent leakage
2. **Manifold Mixup** applied to graph embeddings on 50% of training batches
3. **Focal Loss** (`γ=2.0`, `α=0.60`) — handles class imbalance
4. **AdamW** with two parameter groups:
   - `view_weights`: `weight_decay=0.20` (prevents gate collapse)
   - All other params: `weight_decay=0.05`
5. **CosineAnnealingWarmRestarts** (`T_0=20`, `T_mult=2`)
6. **Early stopping** (patience=10, monitors val accuracy)
7. **Youden's J threshold**: computed on **training** ROC curve each epoch, then applied to val predictions — no post-hoc optimization

### Monitoring
- Every 10 epochs: gate weights are printed (`Pearson`, `Precision`, `Learned`)
- If any gate > 0.90: collapse warning printed (raise `GATE_WD` if it persists)
- Per-fold loss/AUC/accuracy plots saved as `fold{N}_metrics.png`

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `NODE_FEAT` | 64 | CNN output dim per ROI |
| `GAT_HIDDEN` | 16 | Slim spatial branch |
| `GAT_HEADS` | 4 | GATv2 attention heads |
| `EPOCHS` | 500 | Max epochs (early stopping applies) |
| `LR` | 5e-5 | AdamW learning rate |
| `WEIGHT_DECAY` | 0.05 | Standard L2 |
| `GATE_WD` | 0.20 | L2 for `view_weights` only |
| `FOCAL_GAMMA` | 2.0 | Focal loss focusing |
| `FOCAL_ALPHA` | 0.60 | Focal loss balance |
| `MIXUP_ALPHA` | 0.4 | Beta distribution for Mixup |
| `MIXUP_PROB` | 0.5 | Fraction of batches with Mixup |
| `EARLY_STOPPING_PATIENCE` | 100 | Patience on val accuracy |
| `N_FOLDS` | 5 | Stratified K-Fold |

---

## Usage

```bash
# Default run
python train.py

# Custom paths
python train.py --phenotype D:/ABIDE/asd_717participants.csv --cache_dir path/to/bold_cache

# Override hyperparameters
python train.py --epochs 200 --lr 1e-4 --focal_gamma 3.0 --mixup_prob 0.3
```

---

## Output

| File | Description |
|---|---|
| `dv_sttgat_fold{N}.pt` | Best model weights for fold N |
| `fold{N}_metrics.png` | Train loss / Val loss / AUC / Accuracy curves |

---

## File Structure

```
dv-sttgat-nyu -v2/
├── train.py              # Training pipeline + CV loop
├── model.py              # DVSTTGATModel (LearnableGraph, GATv2, gating)
├── cnn.py                # MultiScaleInceptionCNN (temporal branch)
├── dataset.py            # PyG DataLoaders + sliding-window augmentation
├── graph_construction.py # Pearson + Precision graph builders
├── data_loader.py        # BOLD cache loader + social-brain ROI pruning
└── ensemble_eval.py      # Ensemble evaluation across saved fold checkpoints
```

---

## Methodology Notes

- **No DANN/GRL**: single-site NYU data requires no domain adaptation
- **No data leakage**: Z-scoring is self-normalisation (per-subject, inside fold); sliding windows are subject-wise; threshold is train-derived
- **ROI selection**: 28 social-brain ROIs chosen from canonical neuroscience literature (amygdala, mPFC, STS, TPJ, etc.) — not from preliminary model results
