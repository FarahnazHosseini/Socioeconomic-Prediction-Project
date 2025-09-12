# ============================================
# Colab: Hybrid GCN + Classical ML (End-to-End)
# - If you only have final.csv, set MAKE_SYNTHETIC = True to build the clustered dataset.
# - Otherwise set CSV_PATH to your synthetic file (e.g., 'final_synthetic_clustered_by10.csv').
# ============================================

# ---------- 0) Setup ----------
!pip -q install pandas numpy scikit-learn torch xgboost

import os, numpy as np, pandas as pd
from pathlib import Path

# ==== USER SETTINGS ====
MAKE_SYNTHETIC = True                    # اگر فقط final.csv داری → True
INPUT_CSV      = "/content/final.csv"             # فایل خام
SYNTH_CSV      = "Result.csv"  # فایل Synthetic خروجی
GROUP_SIZE     = 10                      # هر چند ردیف = یک خوشه
TARGET         = "MedianGrossRent"       # هدف آموزش/ارزیابی
K              = 8                       # KNN k=
HIDDEN         = 64                      # بُعد لایهٔ مخفی GCN (همان بُعد embedding)
EPOCHS         = 120                     # تعداد Epoch
DROPOUT        = 0.2
SEED           = 42

# اگر فایل‌ها رو از گوگل درایو می‌خوای بخونی/بنویسی، این دو خط رو آنکامنت کن:
# from google.colab import drive
# drive.mount('/content/drive')

# ---------- 1) (Optional) Make synthetic clustered dataset ----------
def make_synthetic_clusters(in_csv, out_csv, group_size=10,
                            cols=("MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent"),
                            seed=123):
    df = pd.read_csv(in_csv)
    df["StreetID"]  = np.arange(len(df))
    df["ClusterID"] = (df["StreetID"] // group_size).astype(int)
    rng = np.random.default_rng(seed)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in CSV")
        df[c] = pd.to_numeric(df[c], errors="coerce")
        means = df.groupby("ClusterID")[c].transform("mean")
        noise = rng.uniform(0.95, 1.05, size=len(df))
        df[c] = means * noise
    df.to_csv(out_csv, index=False)
    return out_csv

csv_path = SYNTH_CSV
if MAKE_SYNTHETIC:
    assert Path(INPUT_CSV).exists(), f"Upload {INPUT_CSV} first (Colab: left panel > Files > Upload)"
    csv_path = make_synthetic_clusters(INPUT_CSV, SYNTH_CSV, group_size=GROUP_SIZE)
print(f"[INFO] Using CSV: {csv_path}")

# ---------- 2) Imports for GCN + ML ----------
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[WARN] xgboost not available; will skip XGBRegressor.")

# ---------- 3) Utilities ----------
def pick_feature_columns(df):
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if len(z_cols) >= 8:
        return z_cols
    curated = [
        "MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent",
        "HouseMedianValue","AvgAnnualHouseExpenditure","MedianAge",
        "PopulationDensity","NumOfHouseHolds","Population","DiversityIndex",
    ]
    feat_cols = [c for c in curated if c in df.columns]
    if len(feat_cols) >= 4:
        return feat_cols
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
        for j in idx[i][1:]:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    return np.vstack([rows, cols]).astype(np.int64)

def normalize_adj_sparse(edge_index, num_nodes):
    row = torch.from_numpy(edge_index[0])
    col = torch.from_numpy(edge_index[1])
    vals = torch.ones(row.numel(), dtype=torch.float32)
    # self-loops
    self_idx = torch.arange(num_nodes, dtype=torch.long)
    row = torch.cat([row, self_idx])
    col = torch.cat([col, self_idx])
    vals = torch.cat([vals, torch.ones(num_nodes, dtype=torch.float32)])

    A = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (num_nodes, num_nodes)).coalesce()
    r, c = A.indices(); v = A.values()
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
    def forward(self, x, A_norm, return_embed=False):
        h = self.g1(x, A_norm)
        h = self.act(h)
        h = self.drop(h)
        out = self.g2(h, A_norm)
        return (out, h) if return_embed else out

def eval_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Corrected RMSE calculation
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ---------- 4) Load data & build graph ----------
df = pd.read_csv(csv_path)
feat_cols = pick_feature_columns(df)
assert TARGET in df.columns, f"Target '{TARGET}' not found. Available: {list(df.columns)}"

X = df[feat_cols].astype(float).values
y = df[[TARGET]].astype(float).values

x_scaler = StandardScaler()
y_scaler = StandardScaler()
Xs = x_scaler.fit_transform(X)
ys = y_scaler.fit_transform(y)

edge_index = build_knn_edges(Xs, k=K)
A_norm = normalize_adj_sparse(edge_index, num_nodes=Xs.shape[0]).coalesce()

idx_all = np.arange(Xs.shape[0])
idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(Xs, dtype=torch.float32, device=device)
y_tensor = torch.tensor(ys, dtype=torch.float32, device=device)
A_norm = A_norm.to(device)

# ---------- 5) Train GCN & get embeddings ----------
model = GCN(in_dim=Xs.shape[1], hidden_dim=HIDDEN, out_dim=1, dropout=DROPOUT).to(device)
opt = Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

best_val = float("inf")
best_state = None
patience, bad = 20, 0

for ep in range(1, EPOCHS+1):
    model.train()
    opt.zero_grad()
    out = model(X_tensor, A_norm)
    loss = loss_fn(out[idx_train], y_tensor[idx_train])
    loss.backward()
    opt.step()

    # val RMSE
    model.eval()
    with torch.no_grad():
        pred_val = model(X_tensor, A_norm)[idx_val].cpu().numpy()
        true_val = ys[idx_val]
    rmse_val = np.sqrt(mean_squared_error(true_val, pred_val)) # Corrected RMSE calculation
    if rmse_val < best_val:
        best_val = rmse_val
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f"[EarlyStop] epoch={ep}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    pred_scaled, H_embed = model(X_tensor, A_norm, return_embed=True)
H = H_embed.detach().cpu().numpy()        # [N, HIDDEN]  ← این همون embeddingهاست
y_scaled = ys

# ---------- 6) Classical ML on (A) EmbeddingsOnly & (B) Embeddings+Features ----------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_ml(X_tr, y_tr, X_va, y_va, X_te, y_te, family_tag):
    rows = []
    preds = {}
    # RF
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr.ravel())
    p = rf.predict(X_te).reshape(-1,1); preds["RF"] = p
    rows.append(("RF",) + eval_regression(y_te, p))
    # GB
    gb = GradientBoostingRegressor(random_state=SEED)
    gb.fit(X_tr, y_tr.ravel())
    p = gb.predict(X_te).reshape(-1,1); preds["GB"] = p
    rows.append(("GB",) + eval_regression(y_te, p))
    # XGB
    if HAS_XGB:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
        xgb.fit(X_tr, y_tr.ravel(), eval_set=[(X_va, y_va.ravel())], verbose=False)
        p = xgb.predict(X_te).reshape(-1,1); preds["XGB"] = p
        rows.append(("XGB",) + eval_regression(y_te, p))
    # build DataFrame (scaled metrics)
    out = pd.DataFrame(rows, columns=["Model","MAE_scaled","RMSE_scaled","R2_scaled"])
    out["Family"] = family_tag
    return out, preds

tr, va, te = idx_train, idx_val, idx_test
# EmbeddingsOnly (scaled space)
df_e_scaled, pred_e_scaled = train_ml(H[tr], y_scaled[tr], H[va], y_scaled[va], H[te], y_scaled[te], "EmbeddingsOnly")
# Embeddings + Raw Features (note: features are *unscaled* X to keep units; model trees insensitive)
Xef_tr, Xef_va, Xef_te = np.hstack([H[tr], X[tr]]), np.hstack([H[va], X[va]]), np.hstack([H[te], X[te]])
df_ef_scaled, pred_ef_scaled = train_ml(Xef_tr, y_scaled[tr], Xef_va, y_scaled[va], Xef_te, y_scaled[te], "Embeddings+Features")

# ---------- 7) Convert scaled predictions to original units & summarize ----------
def inv_scale(y_s): return y_scaler.inverse_transform(y_s)

y_test_inv = inv_scale(y_scaled[te])

def finalize(df_scaled, preds_dict):
    rows = []
    for _, r in df_scaled.iterrows():
        model_key = r["Model"]
        if model_key not in preds_dict:  # e.g., XGB missing
            continue
        pred_inv = inv_scale(preds_dict[model_key])
        mae, rmse, r2 = eval_regression(y_test_inv, pred_inv)
        rows.append({
            "Target": TARGET,
            "Family": r["Family"],
            "Model": model_key,
            "MAE": mae, "RMSE": rmse, "R2": r2
        })
    return pd.DataFrame(rows)

df_e = finalize(df_e_scaled,  pred_e_scaled)
df_ef = finalize(df_ef_scaled, pred_ef_scaled)
summary = pd.concat([df_e, df_ef], axis=0).sort_values(["Family","Model"]).reset_index(drop=True)

print("\n[RESULTS] Hybrid GCN + Classical ML (Test set):")
with pd.option_context("display.float_format", "{:,.6f}".format):
    display(summary)

# ---------- 8) Save outputs ----------
out_dir = Path("gnn_outputs"); out_dir.mkdir(parents=True, exist_ok=True)
sum_path = out_dir / f"hybrid_summary_{TARGET}.csv"
summary.to_csv(sum_path, index=False)
print(f"Saved: {sum_path}")

# Optionally download
# from google.colab import files
# files.download(str(sum_path))