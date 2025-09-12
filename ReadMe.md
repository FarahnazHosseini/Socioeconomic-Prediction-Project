
# Socioeconomic Prediction with GNN + Classical ML (Census-Style Data)

This repository provides an end-to-end pipeline for predicting socioeconomic indicators from census-style tabular data using three complementary approaches:
1) **Baseline Machine Learning** (Linear/Ridge/Lasso, Random Forest, Gradient Boosting, KNN, SVR, XGBoost\*),
2) **Graph Convolutional Network (GCN, 2-layer)** trained on a **KNN graph** built from features,
3) **Hybrid (GCN Embeddings + ML)** that concatenates learned graph embeddings with standardized tabular features.

> **Typical targets:** `MedianGrossRent`, `FamilyMedianIncome`, `PerCapitaIncome`, `PovertyPercent`  
> **Features:** All numeric covariates after cleaning, optionally standardized and prefixed with `z_`  
> \*XGBoost is optional; used only if installed.

---

## Repository Structure
```
.
├─ PreProcessing.py              # Quick preprocessing to produce final_clean.csv & final_clean_scaled.csv
├─ Baseline.py                   # Baseline regressors on all targets (expects z_* features)
├─ GCN.py                        # Core 2-layer GCN trainer for a single target
├─ GCN on 4 targets.py           # Convenience script to run GCN on 4 targets (and optional K sweep)
├─ Hybrid+ML.py                  # Hybrid training: ML on [z_* features + GCN embeddings]
├─ preprocess_pipeline.py        # (Optional) sklearn Pipeline: median-impute + z-score + leakage-safe splits
├─ README.md                     # This file
└─ data/                         # Place your CSV here (e.g., final.csv)
```
**Note:** Some scripts are Colab-style with hardcoded paths (e.g., `/content/...`). Adjust paths at the top of each script to fit your environment.

---

## Setup

### 1) Environment
Create and activate a fresh environment, then install dependencies:
```bash
# conda (example)
conda create -n socio-gnn python=3.10 -y
conda activate socio-gnn
pip install numpy pandas scipy scikit-learn matplotlib xgboost torch --upgrade
# (Optional) For GPU: install the CUDA-enabled PyTorch per https://pytorch.org/get-started/locally/
```

**Tested minimum versions:**  
`python>=3.9`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `scipy>=1.10`, `matplotlib>=3.7`, `torch>=2.1`

> If `xgboost` is not available, `Baseline.py` will run without it.

### 2) Data
Put your raw CSV (e.g., `final.csv`) under `./data` (or any path you choose). Update scripts accordingly.

---

## Workflow

### A) Preprocessing
Two options are supported:

**Option 1 — Simple script (`PreProcessing.py`):**
- Cleans column names, drops duplicates, coerces numeric-like columns, imputes **median**, standardizes to `z_*`,
- Produces:
  - `final_clean.csv` (imputed, unscaled features + targets),
  - `final_clean_scaled.csv` (z-scored `z_*` + targets).

Run:
```bash
python PreProcessing.py   # Edit paths in the script if needed
```

**Option 2 — Robust sklearn pipeline (`preprocess_pipeline.py`)** *(recommended for leakage-safe experiments)*:
- Fits **imputer & scaler on TRAIN only** when `--test_size` is provided,
- Saves both imputed and scaled **train/test** splits,
- Exports artifacts: `preprocess_imputer.joblib`, `preprocess_scaler.joblib`, `preprocess_pipeline.joblib`, plus a JSON report.

Examples:
```bash
# No split (prepare one full cleaned dataset)
python preprocess_pipeline.py --input ./data/final.csv --output_dir ./data

# With split (avoid data leakage)
python preprocess_pipeline.py --input ./data/final.csv --output_dir ./data --test_size 0.2 --random_state 42
```

**Outputs:**  
- No split → `final_clean.csv`, `final_clean_scaled.csv`  
- With split → `train_clean(.csv)`, `train_clean_scaled(.csv)`, `test_clean(.csv)`, `test_clean_scaled(.csv)`

---

### B) Baseline ML
Trains multiple regressors for each target using **scaled `z_*` features** from `final_clean_scaled.csv`:
```bash
python Baseline.py  # Assumes ./data/final_clean_scaled.csv (edit inside if needed)
```
**Output:** `baseline_all_models.csv` (MAE, RMSE, R² for each model/target)

> Tip: For more robust evaluation, consider cross-validation (e.g., `RepeatedKFold`) and report mean ± std of metrics.

---

### C) GCN (2-layer) on a KNN Graph
Builds a **KNN graph** from features and trains a **2-layer GCN** in a **transductive** setting with early stopping on validation MAE. By default, scripts expect `final_clean_scaled.csv` (or will standardize numeric features if `z_*` are absent).

```bash
# Multiple targets
python "GCN on 4 targets.py"
# or single target
python GCN.py
```

Key hyperparameters (edit at the top of the script):
- `k_neighbors` (e.g., 5/8/12/16), `hidden_dim` (e.g., 64), `epochs` (e.g., 200),
- `learning_rate` (e.g., 5e-3), `weight_decay` (e.g., 1e-4), `dropout_rate` (e.g., 0.2),
- `standardize_y` (True/False), `log1p_y` (True/False), `early_stop_patience` (e.g., 30).

**Per-target outputs:**
- `gcn_results_<target>.csv` (summary: MAE/RMSE/R² + config)
- `gcn_preds_<target>.csv` (`y_true`, `y_pred` on test indices)
- `gcn_history_<target>.csv` (epoch, train_loss, val_mae_std)
- `gcn_emb_<target>.npy` (first-layer embeddings for Hybrid)
- `gcn_pred_vs_true_<target>.png` (scatter plot)
- `gcn_config_<target>.json` (hyperparameters used)

> GPU recommended if available. Move tensors/model to `cuda` for faster training.

---

### D) Hybrid: GCN Embeddings + ML
Use `Hybrid+ML.py` to concatenate learned **GCN embeddings** (`gcn_emb_<target>.npy`) with tabular **`z_*` features**, then train classical ML again. This often improves over Baseline or GCN-only.

Typical steps:
1. Run GCN to generate `gcn_emb_<target>.npy` for your target(s).
2. Run `Hybrid+ML.py` to load `final_clean_scaled.csv`, align rows/indices, concatenate `[z_* + emb]`, and train regressors.
3. Save metrics (e.g., `hybrid_results.csv`) similar to Baseline.

---

## Reproducibility
- Fixed `random_state`/`seed` is used where possible (sklearn splits, PyTorch).
- Preprocessing artifacts (`imputer/scaler`) are saved for consistent transforms.
- For fair comparisons:
  - Use the **same train/test split** across Baseline/GCN/Hybrid.
  - If the target is standardized, **invert** it before reporting metrics.
  - If `log1p` is applied, use `expm1` on predictions before computing MAE/RMSE/R².

---

## Results Template
Fill with your runs:
| Target | Approach | Model                | MAE   | RMSE  | R²   | Notes |
|:------:|:--------:|:---------------------|:-----:|:-----:|:----:|:-----:|
| FamilyMedianIncome | Baseline | RandomForest         |  … |  … |  … | z_* only |
| FamilyMedianIncome | GCN      | 2-layer GCN          |  … |  … |  … | k=8 |
| FamilyMedianIncome | Hybrid   | RF on [z_* + emb]    |  … |  … |  … | emb dim=64 |

---

## Troubleshooting
- **FileNotFoundError**: Paths may be hardcoded (e.g., `/content/...`). Update them at the top of scripts.
- **No feature columns detected**: Ensure your CSV has numeric features (beyond IDs/targets) and not all NaN.
- **xgboost not found**: `pip install xgboost` or ignore (Baseline runs without it).
- **CUDA not available**: GCN runs on CPU but slower. Install a CUDA-enabled PyTorch if you need GPU.
- **Poor metrics after `log1p`**: Make sure to `expm1` predictions **before** computing metrics.

---

## Citation
If you use this code, please cite the repository (replace with your paper/preprint when available):
```bibtex
@misc{gnn_socio_repo,
  title        = {Socioeconomic Prediction with GNN + Classical ML},
  author       = {Hosseini, Farahnaz and collaborators},
  year         = {2025},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/<your-username>/<your-repo>}}
}
```

---

## License
We recommend the **MIT License** for broad research and practical use. Add a `LICENSE` file (MIT text) or choose a license that fits your needs.

---

## Contact
- **Maintainer:** Farahnaz Hosseini  
- **Issues:** Please open a GitHub Issue with a minimal reproducible example and logs.
