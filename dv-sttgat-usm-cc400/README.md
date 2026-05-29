# Dual-View Spatio-Temporal Graph Attention Transformer (DV-STTGAT) — usm-Only, CC400 Atlas

This directory contains the implementation of the **Dual-View Spatio-Temporal Graph Attention Transformer (DV-STTGAT)** model tailored for **ASD** (Autism Spectrum Disorder) vs. **TD** (Typical Development) classification using resting-state fMRI (rs-fMRI) data, specifically restricted to the usm site of the ABIDE dataset. 

This model extracts temporal characteristics from blood-oxygen-level-dependent (BOLD) signals using a multi-scale temporal CNN, constructs three distinct spatial graphs (static Pearson correlation, sparse partial correlation, and an end-to-end learnable cosine-similarity graph), and fuses them using a learnable softmax gating mechanism.

---

## 📂 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Step-by-Step Installation Guide](#2-step-by-step-installation-guide)
3. [Data Directory & Files Setup](#3-data-directory--files-setup)
4. [Pipeline Step 1: Pre-Extract BOLD Time-Series (Caching)](#4-pipeline-step-1-pre-extract-bold-time-series-caching)
5. [Pipeline Step 2: Model Training & 5-Fold Cross-Validation](#5-pipeline-step-2-model-training--5-fold-cross-validation)
6. [Pipeline Step 3: Run Soft-Voting Ensemble Evaluation](#6-pipeline-step-3-run-soft-voting-ensemble-evaluation)
7. [Optional Step: Hyperparameter Tuning (Optuna)](#7-optional-step-hyperparameter-tuning-optuna)
8. [Codebase Architecture & File Roles](#8-codebase-architecture--file-roles)
9. [Troubleshooting & Common Issues](#9-troubleshooting--common-issues)

---

## 1. System Prerequisites

Before starting, ensure your system has the following installed:
* **Python:** Version **3.8, 3.9, or 3.10** is required. 
  * *Do not use Python 3.11 or 3.12 yet, as PyTorch Geometric (PyG) binaries can sometimes be unstable or unavailable for newer releases.*
* **Hardware:** A CUDA-compatible NVIDIA GPU (e.g., RTX 30/40 series) is highly recommended for faster training, but the code will run on CPU automatically if CUDA is unavailable.

---

## 2. Step-by-Step Installation Guide

Follow these commands line-by-line.

### Step 2.1: Open Terminal & Clone/Navigate to Directory
Open PowerShell (on Windows) or Terminal, and navigate to the project directory:
```powershell
cd "e:\ASD_TD_Project\dv-sttgat-usm-cc400"
```

### Step 2.2: Create and Activate a Virtual Environment
Using a virtual environment prevents conflicts with other Python libraries installed on your machine.

**On Windows (PowerShell):**
```powershell
# Create the environment named 'venv'
python -m venv venv

# Activate the environment
.\venv\Scripts\Activate.ps1
```
*(If you get a permission error about running scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, then run the activate command again.)*

**On Windows (Command Prompt - CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, your command prompt should show `(venv)` at the beginning.

### Step 2.3: Install Dependencies

You can choose either of the following two methods to install the required libraries:

#### Option A: Quick Installation via `requirements.txt` (Recommended for CPU or Default Setup)
We have provided a [requirements.txt](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/requirements.txt) file that lists all dependencies (including PyTorch and PyTorch Geometric). To install everything with a single command, run:
```bash
pip install -r requirements.txt
```

#### Option B: Custom Manual Installation (Recommended for CUDA GPU Setup)
If you have a dedicated NVIDIA GPU and want to ensure PyTorch matches your system's CUDA version, install the packages manually in this order:

1. **Install PyTorch:**
   * **For GPU Support (NVIDIA CUDA 11.8):**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   * **For GPU Support (NVIDIA CUDA 12.1 / 12.4):**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   * **For CPU Only:**
     ```bash
     pip install torch torchvision torchaudio
     ```

2. **Install PyTorch Geometric (PyG):**
   ```bash
   pip install torch-geometric
   ```

3. **Install Other Scientific & Neuroimaging Libraries:**
   ```bash
   pip install numpy pandas scikit-learn nilearn nibabel matplotlib joblib requests optuna tqdm
   ```

### Step 2.4: Verify the Installation
Run this simple command to verify that PyTorch, CUDA (GPU detection), and PyTorch Geometric are correctly installed and integrated:
```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| PyG:', torch_geometric.__version__)"
```
If you see the version numbers printed without error, your environment is ready!

---

## 3. Data Directory & Files Setup

The code expects a structured directory layout for the ABIDE datasets. Ensure your data folders match the structure shown below.

### Expected Directory Layout
```
D:/ABIDE/
├── asd_717participants.csv         <-- Phenotype metadata file
├── 0050001/                        <-- Example Subject Folder (using SUB_ID)
│   └── 0050001_rest_1_moco_MNI_b6_bp_nuisreg.0.5mm_gsr.nii.gz   <-- fMRI Scan
├── 0050002/
│   └── 0050002_rest_1_moco_MNI_b6_bp_nuisreg.0.5mm_gsr.nii.gz
└── ...
```

### 1. Phenotype CSV Requirements
The phenotype metadata file (e.g., `asd_717participants.csv`) must contain at least the following column headers:
* `SUB_ID`: Subject ID matching the directory folder names (either as integers or strings).
* `DX_GROUP`: Diagnosis group where `1` = ASD (Autism Spectrum Disorder) and `2` = TD (Typical Development).
* `SITE_ID`: Site label. The usm loader looks specifically for `"usm"` or `"ABIDEII-usm_1"`.

### 2. fMRI File Naming Convention
The raw fMRI image must be in NIfTI format (either `.nii` or `.nii.gz`) and be named:
`{SUB_ID}_rest_1_moco_MNI_b6_bp_nuisreg.0.5mm_gsr.nii.gz`

### 3. Configuring Directory Paths

By default, the scripts point to local development directories (e.g., `D:\ABIDE\`). Since you are setting this up on your own machine, you will need to specify where your data is located. You can do this in **two ways**:

#### Option A: Use Command Line Arguments (Recommended)
You do not need to modify any source code. Simply pass your paths as arguments when running the scripts:
* **For caching BOLD signals:**
  ```bash
  python build_cc400_cache.py --data_dir "/path/to/your/ABIDE/folder" --cache_dir "./cc400_bold_cache"
  ```
* **For training the model:**
  ```bash
  python train.py --phenotype "/path/to/your/asd_717participants.csv" --cache_dir "./cc400_bold_cache"
  ```
* **For evaluation:**
  ```bash
  python ensemble_eval.py --phenotype "/path/to/your/asd_717participants.csv" --cache_dir "./cc400_bold_cache"
  ```
* **For hyperparameter tuning:**
  ```bash
  python tune.py --phenotype "/path/to/your/asd_717participants.csv" --cache_dir "./cc400_bold_cache"
  ```

#### Option B: Modify the Defaults in the Code
If you want to run the scripts without typing the paths every time, open the following files and edit the default path variables located near the top of the files:

1. **[build_cc400_cache.py](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/build_cc400_cache.py)**
   * Locate **Line 67**:
     ```python
     DEFAULT_DATA_DIR  = r"D:\ABIDE"  # <-- Change this to your raw fMRI data directory
     ```
2. **[train.py](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/train.py)**
   * Locate **Line 61**:
     ```python
     DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"  # <-- Change to your phenotype CSV file path
     ```
3. **[ensemble_eval.py](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/ensemble_eval.py)**
   * Locate **Line 36**:
     ```python
     DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"  # <-- Change to your phenotype CSV file path
     ```
4. **[tune.py](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/tune.py)**
   * Locate **Line 86**:
     ```python
     DEFAULT_PHENOTYPE = r"D:\ABIDE\asd_717participants.csv"  # <-- Change to your phenotype CSV file path
     ```

### 4. Changing the Site ID Filter

By default, the data loader filters specifically for the usm site. If you wish to filter for a different site, open **[data_loader.py](file:///e:/ASD_TD_Project/dv-sttgat-usm-cc400/data_loader.py)**:
* Locate **Line 114**:
  ```python
  usm_pheno = pheno[(pheno["SITE_ID"] == "usm") | (pheno["SITE_ID"] == "ABIDEII-usm_1")]
  ```
  Change `"usm"` and `"ABIDEII-usm_1"` to your target site ID(s) from your phenotype file.

---

## 4. Pipeline Step 1: Pre-Extract BOLD Time-Series (Caching)

Processing raw 4D fMRI files is CPU and RAM intensive. Therefore, we run `build_cc400_cache.py` first. This script:
1. Downloads the **Craddock 2012 (CC400)** parcellation atlas (if not cached locally).
2. Uses `nilearn.input_data.NiftiLabelsMasker` to extract the average BOLD signal for each of the ~392 parcels in standard MNI space.
3. Automatically applies linear detrending to clean slow drift noises.
4. Saves raw `.npy` signal matrices of shape `(Timepoints, ROIs)` and coordinates to `./cc400_bold_cache/`.

> [!TIP]
> **Automatic Skipping & Resuming:** `build_cc400_cache.py` is smart. If you run it and the destination cache directory already contains extracted `*_bold.npy` files for some subjects, the script will **automatically detect and skip them**, processing only the remaining/new subjects. This allows you to safely restart or resume caching if it gets interrupted.

### Run Caching for All Sites:
By default, the script caches all subjects found in the directory so you can reuse this cache for other projects later:
```bash
python build_cc400_cache.py --data_dir D:/ABIDE --cache_dir ./cc400_bold_cache
```

### Run a Fast Smoke-Test (Process only 5 subjects):
To quickly check that your environment and file paths work without waiting for the whole dataset:
```bash
python build_cc400_cache.py --data_dir D:/ABIDE --cache_dir ./cc400_bold_cache --n_jobs 1 --limit 5
```

### Command Options:
* `--data_dir`: Path to the raw ABIDE dataset directory (default: `D:/ABIDE`).
* `--cache_dir`: Destination directory to output the cached matrices (default: `./cc400_bold_cache`).
* `--site`: (Optional) Comma-separated sites to limit caching to (e.g., `usm`).
* `--n_jobs`: Number of parallel CPU workers to use. Omitting this uses all available cores.
* `--limit`: Stop after processing N subjects (useful for debugging).

---

## 5. Pipeline Step 2: Model Training & 5-Fold Cross-Validation

Once you have built your cache, you can train the model. The training script (`train.py`):
1. Loads cached BOLD signals for usm-site subjects only.
2. Lazily computes a Pearson correlation matrix and a sparse partial covariance (Precision matrix via `GraphicalLassoCV` with Ledoit-Wolf fallback) for every subject.
3. Executes a **Stratified 5-Fold Cross-Validation**.
4. Inside each fold, it Z-scores BOLD signals *subject-wise* to prevent data leakage from validation/testing subjects.
5. Performs data augmentation on the training set using two sliding windows: Window A `[0:150]` and Window B `[25:175]`.
6. Uses **Manifold Mixup** on graph embeddings and **Focal Loss** to handle class imbalances.
7. Saves best model weights for fold $N$ to `dv_sttgat_fold{N}.pt`.
8. Saves learning curves and ROC plots as `fold{N}_metrics.png` and a cross-fold average summary plot as `cross_fold_summary.png`.
9. Logs execution metrics and final classification performance into a timestamped file `results_YYYYMMDD_HHMMSS.txt`.

> [!WARNING]
> **No Auto-Generation:** Running `train.py` **will not** automatically generate the BOLD signal cache. If the specified cache directory does not exist or contains no matching subjects, the script will exit with an error. Running Step 1 (`build_cc400_cache.py`) is a **mandatory prerequisite**.

### Run Training with Default Hyperparameters:
```bash
python train.py --phenotype D:/ABIDE/asd_717participants.csv --cache_dir ./cc400_bold_cache
```

### Custom Training Command:
```bash
python train.py --epochs 300 --lr 5e-5 --batch_size 16 --focal_gamma 2.5 --mixup_prob 0.3
```

### Critical Command Options:
* `--phenotype`: Path to your ABIDE participants CSV file.
* `--cache_dir`: Directory where step 1 cached its `.npy` files.
* `--epochs`: Max training epochs per fold (default: `500`). Early stopping patience is set to `100` epochs.
* `--lr`: Initial AdamW learning rate (default: `1e-4`).
* `--batch_size`: Batch size (default: `16`).
* `--n_folds`: Number of CV splits (default: `5`).

---

## 6. Pipeline Step 3: Run Soft-Voting Ensemble Evaluation

Once training finishes, you will have 5 trained checkpoints: `dv_sttgat_fold1.pt` to `dv_sttgat_fold5.pt`. The `ensemble_eval.py` script aggregates their predictions to provide a more robust SOTA score:
1. Loads all 5 fold weights.
2. Performs soft voting (averages predicted probabilities across all 5 models) on each usm subject.
3. Dynamically calculates Youden's J optimal threshold to classify the soft predictions.
4. Outputs the final ensembled **ROC-AUC**, **Accuracy**, **Sensitivity** (true positive rate), and **Specificity** (true negative rate) for the entire dataset.

### Run Ensemble Evaluation:
```bash
python ensemble_eval.py --phenotype D:/ABIDE/asd_717participants.csv --cache_dir ./cc400_bold_cache
```

---

## 7. Optional Step: Hyperparameter Tuning (Optuna)

If you wish to optimize the learning rate, weight decay, loss factors, and mixup parameters for a different dataset or configuration, run `tune.py`. This script:
1. Reuses cached graphs and normalizes the BOLD signals once in memory to maximize performance.
2. Uses Bayesian Optimization (Optuna's Tree-structured Parzen Estimator) to search the parameter space.
3. Implements epoch-level pruning (`MedianPruner`), killing trials mid-training if they show poor validation accuracy relative to prior runs.
4. Persists study history to an SQLite database file (`dv_sttgat_cc400_tune.db`), allowing you to interrupt and resume tuning at any time.
5. Saves the best parameters to `best_hyperparams.json` and generates a copy-pasteable command to `best_hyperparams_<timestamp>.txt`.

### Run Tuning (50 trials max, 200 epochs per fold trial):
```bash
python tune.py --phenotype D:/ABIDE/asd_717participants.csv --cache_dir ./cc400_bold_cache --n_trials 50 --tune_epochs 200
```

### Visualizing Tuning Results (Optuna Dashboard)
Once hyperparameter tuning is running or finished, you can view the results on an interactive web-based dashboard (which displays trial histories, parameter importances, and parallel coordinate charts):

1. Launch the dashboard by pointing it to the SQLite database file:
   ```bash
   optuna-dashboard sqlite:///dv_sttgat_cc400_tune.db
   ```
2. Open your web browser and navigate to the address shown in your terminal (typically `http://127.0.0.1:8080/`).

---

## 8. Codebase Architecture & File Roles

Below is an overview of what each script does in the `dv-sttgat-usm-cc400/` directory:

```
dv-sttgat-usm-cc400/
├── build_cc400_cache.py   # Extracts BOLD signals from raw fMRI using CC400 atlas.
├── data_loader.py         # Loads BOLD matrices, filters for usm, aligns with labels.
├── graph_construction.py  # Builds Pearson and Precision graphs for each subject.
├── cnn.py                 # Multi-scale 1D Inception CNN for BOLD temporal modeling.
├── dataset.py             # Formulates PyG graph Data items + sliding window aug.
├── model.py               # Fuses Pearson, Precision & Learned graphs via GATv2 gates.
├── train.py               # Runs Stratified CV, train/val loops, outputs metric plots.
├── ensemble_eval.py       # Averages probability projections across fold checkpoints.
├── tune.py                # Performs automated hyperparameter optimization.
└── README.md              # Documentation (This file).
```

---

## 9. Troubleshooting & Common Issues

### Issue 1: SSL Certificate Expired / Atlas Download Fails
During step 1, the script attempts to download the Craddock 2012 atlas from `nitrc.org`. If the website's certificate is expired, python might raise an SSL error.
* **Fix:** The script has a built-in fallback that requests the URL while disabling certificate verification (`verify=False`). If this still fails on your network, manually download the file:
  1. Download: [craddock_2011_parcellations.tar.gz](https://cluster_roi.projects.nitrc.org/Parcellations/craddock_2011_parcellations.tar.gz)
  2. Create the directory: `C:\Users\<YourUsername>\nilearn_data\craddock_2012\`
  3. Place the downloaded `.tar.gz` inside that folder.
  4. Run `build_cc400_cache.py` again. It will detect the local archive, extract `scorr05_mean_all.nii.gz` automatically, and complete.

### Issue 2: GraphicalLassoCV taking too long or throwing "ConvergenceWarning"
Calculating the inverse covariance matrix via `GraphicalLassoCV` is mathematically expensive. 
* **Details:** The console might print convergence warnings. These can be safely ignored.
* **Fix:** `graph_construction.py` runs this step in parallel across CPU cores using `joblib`. If a subject's covariance fails to converge within 1000 iterations, the script automatically triggers a robust fallback to **Ledoit-Wolf shrinkage**, ensuring the pipeline never crashes.

### Issue 3: PyTorch Geometric (PyG) installation errors
If `pip install torch-geometric` fails due to compilation or wheel errors, you should install the pre-compiled binaries matching your exact PyTorch and CUDA versions.
1. Run `python -c "import torch; print(torch.__version__)"` to see your PyTorch version (e.g., `2.1.2`).
2. Run `python -c "import torch; print(torch.version.cuda)"` to see your CUDA version (e.g., `12.1`).
3. Visit [PyG Binaries Page](https://data.pyg.org/whl/) or install via:
   ```bash
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   ```
   *(Replace `${TORCH}` and `${CUDA}` with your actual versions, e.g., `torch-2.1.0+cu121`).*

### Issue 4: Out of Memory (OOM) on GPU
If training crashes with an Out of Memory error:
* **Fix:** Decrease the `--batch_size` argument to `8` (e.g., `python train.py --batch_size 8`).
* Alternatively, run on CPU by disabling CUDA visibility:
  ```powershell
  $env:CUDA_VISIBLE_DEVICES=""
  python train.py
  ```
