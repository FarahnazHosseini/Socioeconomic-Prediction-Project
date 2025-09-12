# ============================================
# stage4_gnn_pipeline.py
# --------------------------------------------
# 2-layer GCN on a KNN graph for census-style tabular data.
# - Robust to feature selection (uses z_* if present, else standardizes numeric cols).
# - Builds KNN graph *after* subsetting and re-normalizes (D^{-1/2} (A+I) D^{-1/2}).
# - Standardizes target y (optional) and inverses for reporting.
# - Early stopping on validation MAE.
# - Saves results, predictions, history, scatter plot, and config.
# - Can loop over multiple targets and perform a small K sweep.
# ============================================

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------
# Helpers
# -----------------------
EXCLUDE_LIKE = [
    "zip","zipcode","zcta","geoid","tract","fips","name","state","county",
    "lat","lon","latitude","longitude","ZIP Code","id","ID"
]

def _renorm_with_self_loops(A_csr: sparse.csr_matrix) -> sparse.csr_matrix:
    A_hat = A_csr + sparse.eye(A_csr.shape[0], format="csr")
    deg = np.asarray(A_hat.sum(1)).ravel()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def _build_knn_norm(X: np.ndarray, k: int) -> sparse.csr_matrix:
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False, n_jobs=-1)
    A = A.maximum(A.T)  # make undirected
    return _renorm_with_self_loops(A)

def _pick_features(df: pd.DataFrame, targets: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if len(z_cols) > 0:
        X = df[z_cols].to_numpy(np.float32)
        return X, z_cols, df

    # else: auto-pick numeric features excluding likely ID or target columns
    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = set()
    for c in numeric.columns:
        cname = c.lower().strip()
        if any(tok in cname for tok in EXCLUDE_LIKE):
            drop_cols.add(c)
        if c in targets:
            drop_cols.add(c)
    feat_cols = [c for c in numeric.columns if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after exclusion.")
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric[feat_cols]).astype(np.float32)
    # attach standardized features as z_* (optional)
    z_df = pd.DataFrame(X, columns=[f"z_{c}" for c in feat_cols], index=df.index)
    merged = df.join(z_df)
    return X, [f"z_{c}" for c in feat_cols], merged

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2   = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2)


# -----------------------
# GCN Model
# -----------------------
class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim,  bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor, return_h1: bool=False):
        # x: (N, F), A: sparse (N, N)
        h1 = torch.sparse.mm(A, self.fc1(x))
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        out = torch.sparse.mm(A, self.fc2(h1)).squeeze(-1)
        if return_h1:
            return out, h1
        return out


# -----------------------
# Training Routine
# -----------------------
def train_gcn_on_target(
    df: pd.DataFrame,
    target: str,
    save_dir: Path,
    k: int = 8,
    subset: Optional[int] = None,
    seed: int = 123,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 5e-3,
    wd: float = 1e-4,
    dropout: float = 0.2,
    standardize_y: bool = True,
    early_stop_patience: int = 30,
    log1p_y: bool = False,
    csv_path: Optional[str] = None # Added csv_path argument
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Keep only rows with non-missing target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")
    dft = df.copy()
    dft = dft.dropna(subset=[target]).reset_index(drop=True)

    # Features
    X, feat_cols, dft = _pick_features(dft, [target])
    y = dft[target].to_numpy(np.float32)

    # Optional log1p transform (reversed for reporting)
    if log1p_y:
        y_trans = np.log1p(y)
    else:
        y_trans = y.copy()

    # Optional standardize y (for stable training)
    if standardize_y:
        y_scaler = StandardScaler()
        y_std = y_scaler.fit_transform(y_trans.reshape(-1,1)).ravel().astype(np.float32)
    else:
        y_scaler = None
        y_std = y_trans.astype(np.float32)

    n_total = len(y_std)

    # Subset for speed (if requested)
    if subset is not None and subset < n_total:
        rng = np.random.default_rng(seed)
        sel_idx = np.sort(rng.choice(n_total, size=subset, replace=False))
    else:
        sel_idx = np.arange(n_total)

    Xs = X[sel_idx]
    ys = y[sel_idx]
    y_stds = y_std[sel_idx]

    # Build KNN graph on the subset, then renormalize
    print(f"[{target}] Building KNN graph on N={Xs.shape[0]} (k={k}) ...")
    A_norm = _build_knn_norm(Xs, k)

    # Torch tensors
    A_coo = A_norm.tocoo()
    indices = np.vstack((A_coo.row, A_coo.col))
    A_sp = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(A_coo.data, dtype=torch.float32),
        (Xs.shape[0], Xs.shape[0])
    ).coalesce()
    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y_stds, dtype=torch.float32)

    # Splits (transductive)
    idx_all = np.arange(len(y_stds))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.4, random_state=seed)
    idx_val, idx_test  = train_test_split(idx_tmp, test_size=0.5, random_state=seed)

    idx_train_t = torch.tensor(idx_train, dtype=torch.long)
    idx_val_t   = torch.tensor(idx_val,   dtype=torch.long)
    idx_test_t  = torch.tensor(idx_test,  dtype=torch.long)

    # Model & Optim
    model = GCN(in_dim=Xs.shape[1], hidden_dim=hidden, out_dim=1, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(delta=1.0)

    # Train loop with early stopping
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    history = []

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        out = model(X_t, A_sp)
        loss = loss_fn(out[idx_train_t], y_t[idx_train_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_t, A_sp)
            val_mae = F.l1_loss(out_val[idx_val_t], y_t[idx_val_t]).item()
        history.append((epoch, float(loss.item()), float(val_mae)))

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # early stopping
        if early_stop_patience is not None and epoch - best_epoch >= early_stop_patience:
            print(f"[{target}] Early stopped at epoch {epoch} (best @ {best_epoch} val_mae={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_std, emb = model(X_t, A_sp, return_h1=True)  # pred on standardized scale; emb = first-layer activations
        pred_std = pred_std.cpu().numpy()

    # invert target transforms
    if standardize_y:
        pred_trans = y_scaler.inverse_transform(pred_std.reshape(-1,1)).ravel()
        y_trans_used = y_trans[sel_idx]  # original-transformed scale before std
    else:
        pred_trans = pred_std
        y_trans_used = y_trans[sel_idx]

    if log1p_y:
        # invert log1p
        pred = np.expm1(pred_trans)
        y_true = ys
    else:
        pred = pred_trans
        y_true = ys

    # Metrics on test indices
    y_test = y_true[idx_test]
    p_test = pred[idx_test]

    mae, rmse, r2 = _metrics(y_test, p_test)
    print(f"[{target}] GCN(2L)  MAE={mae:,.3f}  RMSE={rmse:,.3f}  R2={r2:,.3f}")

    # Save artifacts
    save_dir.mkdir(parents=True, exist_ok=True)
    res_path  = save_dir / f"gcn_results_{target}.csv"
    pred_path = save_dir / f"gcn_preds_{target}.csv"
    hist_path = save_dir / f"gcn_history_{target}.csv"
    emb_path  = save_dir / f"gcn_emb_{target}.npy"
    cfg_path  = save_dir / f"gcn_config_{target}.json"
    plot_path = save_dir / f"gcn_pred_vs_true_{target}.png"

    pd.DataFrame([{
        "Target": target, "Model": "GCN(2L)",
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "N": len(y_true), "k": k, "hidden": hidden, "epochs": epoch,
        "subset": int(len(y_true)), "standardize_y": bool(standardize_y), "log1p_y": bool(log1p_y)
    }]).to_csv(res_path, index=False)

    pd.DataFrame({
        "y_true": y_true[idx_test],
        "y_pred": pred[idx_test]
    }).to_csv(pred_path, index=False)

    pd.DataFrame(history, columns=["epoch","train_loss","val_mae_std"]).to_csv(hist_path, index=False)

    # embeddings (first-layer)
    np.save(emb_path, emb.detach().cpu().numpy())

    # config dump
    with open(cfg_path, "w") as f:
        json.dump({
            "target": target, "k": k, "hidden": hidden, "epochs": epochs, "lr": lr, "wd": wd,
            "dropout": dropout, "subset": None if subset is None else int(subset),
            "standardize_y": bool(standardize_y), "log1p_y": bool(log1p_y), "seed": seed
        }, f, indent=2)

    # Plot Pred vs True
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, p_test, s=10)
    mn = float(min(y_test.min(), p_test.min()))
    mx = float(max(y_test.max(), p_test.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"GCN Pred vs True — {target}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "target": target,
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "paths": {
            "results": str(res_path),
            "preds": str(pred_path),
            "history": str(hist_path),
            "embeddings": str(emb_path),
            "config": str(cfg_path),
            "plot": str(plot_path),
        }
    }


def run_targets(
    df: pd.DataFrame,
    targets: List[str],
    save_dir: Path,
    k: int = 12,
    subset: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    rows = []
    for t in targets:
        try:
            out = train_gcn_on_target(df, t, save_dir=save_dir, k=k, subset=subset, **kwargs)
            rows.append({"Target": t, "Model": "GCN(2L)", "MAE": out["MAE"], "RMSE": out["RMSE"], "R2": out["R2"]})
        except Exception as e:
            print(f"[WARN] Skipped target {t}: {e}")
    res = pd.DataFrame(rows)
    if len(res):
        res.to_csv(save_dir / "gcn_results_all_targets.csv", index=False)
    return res


def k_sweep(
    df: pd.DataFrame,
    target: str,
    save_dir: Path,
    k_values: List[int] = (5,8,12,16),
    subset: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        try:
            out = train_gcn_on_target(df, target, save_dir=save_dir, k=k, subset=subset, **kwargs)
            rows.append({"k": k, "MAE": out["MAE"], "RMSE": out["RMSE"], "R2": out["R2"]})
        except Exception as e:
            print(f"[WARN] k={k} failed: {e}")
    res = pd.DataFrame(rows)
    if len(res):
        res.sort_values("MAE", inplace=True)
        res.to_csv(save_dir / f"gcn_k_sweep_{target}.csv", index=False)
    return res


# -----------------------
# Entry point - Modified for direct execution in Colab
# -----------------------

# Hardcoded arguments based on the original script's defaults
csv_path = "/content/final_clean_scaled.csv" # Assuming this is the correct path after preprocessing
targets_list = ["MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent"]
save_directory = Path("/content/gcn_stage4_outputs")
k_neighbors = 8
subset_size = -1 # Set to use all data
hidden_dim = 64
num_epochs = 200
learning_rate = 5e-3
weight_decay = 1e-4
dropout_rate = 0.2
random_seed = 123
standardize_y_target = True
log1p_y_target = True
early_stop_patience_epochs = 30 # Set to None or 0 to disable
k_sweep_target = "" # Set to a target name (e.g., "MedianGrossRent") to run k-sweep

# Load the dataframe using the specified csv_path
# This part replaces the file selection logic from the original main()
if not os.path.exists(csv_path):
     raise FileNotFoundError(f"CSV file not found at: {csv_path}")

print(f"[INFO] Using CSV: {csv_path}")
df = pd.read_csv(csv_path)


# Prepare outputs
save_directory.mkdir(parents=True, exist_ok=True)

# Hyperparams
subset = None if subset_size == -1 else subset_size
std_y = standardize_y_target
patience = None if early_stop_patience_epochs == 0 else early_stop_patience_epochs

# Run
if k_sweep_target:
    print(f"[INFO] Running k-sweep for target: {k_sweep_target}")
    res = k_sweep(
        df, target=k_sweep_target, save_dir=save_directory,
        k_values=(5,8,12,16), subset=subset, seed=random_seed,
        hidden=hidden_dim, epochs=num_epochs, lr=learning_rate, wd=weight_decay,
        dropout=dropout_rate, standardize_y=std_y, early_stop_patience=patience, log1p_y=log1p_y_target
    )
    print(res)
else:
    print(f"[INFO] Running standard training for targets: {targets_list}")
    res = run_targets(
        df, targets=targets_list, save_dir=save_directory, k=k_neighbors,
        subset=subset, seed=random_seed, hidden=hidden_dim, epochs=num_epochs,
        lr=learning_rate, wd=weight_decay, dropout=dropout_rate,
        standardize_y=std_y, early_stop_patience=patience, log1p_y=log1p_y_target
    )
    print(res)