# ============================================
# stage4_gnn_pipeline.py
# --------------------------------------------
# 2-layer GCN on a KNN graph for census-style tabular data.
# - Robust to feature selection (uses z_* if present, else standardizes numeric cols).
# - Builds KNN graph *after* subsetting and re-normalizes (D^{-1/2} (A+I) D^{-1/2}).
# - Standardizes target y (optional) and inverses for reporting.
# - Early stopping on validation MAE.
# - Saves results, predictions, history, scatter data.
# ============================================

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def pick_feature_columns(df):
    # Prefer z_* standardized features if present
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if len(z_cols) >= 8:  # enough z_* columns
        return z_cols
    # Else, use a curated list if present, otherwise all numeric excluding obvious IDs
    curated = [
        "MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent",
        "HouseMedianValue","AvgAnnualHouseExpenditure","MedianAge",
        "PopulationDensity","NumOfHouseHolds","Population","DiversityIndex",
    ]
    feat_cols = [c for c in curated if c in df.columns]
    if len(feat_cols) >= 4:
        return feat_cols
    # Fallback: all numeric except id-like columns
    drop_like = ["zip","zipcode","zcta","geoid","tract","fips","name","state","county",
                 "lat","lon","latitude","longitude","id","StreetID","ClusterID"]
    blocked = set([c for c in df.columns for k in drop_like if k.lower() in c.lower()])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in blocked]
    return feat_cols


def build_knn_edges(X, k=8):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto")
    nbrs.fit(X)
    _, idx = nbrs.kneighbors(X)
    rows, cols = [], []
    N = X.shape[0]
    for i in range(N):
        for j in idx[i][1:]:  # skip self
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)  # symmetrize
    return np.vstack([rows, cols]).astype(np.int64)


def normalize_adj_sparse(edge_index, num_nodes):
    row = torch.from_numpy(edge_index[0])
    col = torch.from_numpy(edge_index[1])
    vals = torch.ones(row.numel(), dtype=torch.float32)

    # add self-loops
    self_idx = torch.arange(num_nodes, dtype=torch.long)
    row = torch.cat([row, self_idx])
    col = torch.cat([col, self_idx])
    vals = torch.cat([vals, torch.ones(num_nodes, dtype=torch.float32)])

    A = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (num_nodes, num_nodes)).coalesce()
    r, c = A.indices()
    v = A.values()
    deg = torch.zeros(num_nodes, dtype=torch.float32).scatter_add_(0, r, v)
    deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
    v_norm = deg_inv_sqrt[r] * v * deg_inv_sqrt[c]
    A_norm = torch.sparse_coo_tensor(torch.stack([r, c]), v_norm, (num_nodes, num_nodes)).coalesce()
    return A_norm


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x, A_norm):
        return torch.sparse.mm(A_norm, self.lin(x))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.g2 = GCNLayer(hidden_dim, out_dim)
    def forward(self, x, A_norm):
        h = self.g1(x, A_norm)
        h = self.act(h)
        h = self.drop(h)
        return self.g2(h, A_norm)


def run_one_target(df, feat_cols, target, k=8, hidden=64, epochs=120, lr=1e-3, dropout=0.2, seed=42):
    N = len(df)
    print(f"[{target}] Building KNN graph on N={N} (k={k}) ...")

    # X features (standardize if no z_)
    X = df[feat_cols].astype(float).values
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    # Graph
    edge_index = build_knn_edges(Xs, k=k)
    A_norm = normalize_adj_sparse(edge_index, num_nodes=N).coalesce()

    # y (single target) with scaling
    y = df[[target]].astype(float).values
    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y)

    # splits
    idx = np.arange(N)
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=seed)

    # tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(Xs, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(ys, dtype=torch.float32, device=device)
    A_norm = A_norm.to(device)

    # model
    model = GCN(in_dim=Xs.shape[1], hidden_dim=hidden, out_dim=1, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE for robust early stopping

    # training with early stop on val MAE
    best_val = float("inf")
    best_state = None
    patience, bad = 20, 0
    history = []

    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        out = model(X_tensor, A_norm)
        loss = loss_fn(out[idx_train], y_tensor[idx_train])
        loss.backward()
        opt.step()

        # val metrics
        model.eval()
        with torch.no_grad():
            pred_val = model(X_tensor, A_norm)[idx_val].cpu().numpy()
            true_val = y_tensor[idx_val].cpu().numpy()
        pred_val_inv = y_scaler.inverse_transform(pred_val)
        true_val_inv = y_scaler.inverse_transform(true_val)
        val_mae = mean_absolute_error(true_val_inv, pred_val_inv)
        history.append(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                # print(f"[{target}] Early stopping at epoch {ep}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on test
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor, A_norm).cpu().numpy()

    def metrics(split_idx):
        p = pred[split_idx]
        t = ys[split_idx]
        p_inv = y_scaler.inverse_transform(p)
        t_inv = y_scaler.inverse_transform(t)
        mae = mean_absolute_error(t_inv, p_inv)
        # Calculate RMSE by taking the square root of MSE
        rmse = np.sqrt(mean_squared_error(t_inv, p_inv))
        r2 = r2_score(t_inv, p_inv)
        return mae, rmse, r2

    mae, rmse, r2 = metrics(idx_test)
    print(f"[{target}] GCN(2L)  MAE={mae:,.3f}  RMSE={rmse:,.3f}  R2={r2:.3f}")

    return {
        "target": target,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="final.csv")
    ap.add_argument("--targets", nargs="+", default=["MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args, unknown = ap.parse_known_args() # Corrected line

    csv_path = Path(args.csv)
    print(f"[INFO] Using CSV: {csv_path}")
    print(f"[INFO] Running standard training for targets: {args.targets}")

    df = pd.read_csv(csv_path)

    feat_cols = pick_feature_columns(df)
    results = []
    for t in args.targets:
        res = run_one_target(
            df=df, feat_cols=feat_cols, target=t,
            k=args.k, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, dropout=args.dropout, seed=args.seed
        )
        results.append(res)

    # Summary table
    out = pd.DataFrame([
        {"Target": r["target"], "Model": "GCN(2L)", "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]}
        for r in results
    ])
    with pd.option_context("display.float_format", "{:,.6f}".format):
        print(out)

    # Save summary
    out_path = Path("gnn_outputs"); out_path.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path / "stage4_summary.csv", index=False)


if __name__ == "__main__":
    main()