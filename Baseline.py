# === Baseline: multiple regressors on all targets ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# اگر XGBoost نصب هست:
try:
    from xgboost import XGBRegressor
    has_xgb = True
except:
    has_xgb = False

# Load dataset - corrected to load the scaled data
data = pd.read_csv("/content/final_clean_scaled.csv")

targets = [c for c in ["MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent"] if c in data.columns]
feature_cols = [c for c in data.columns if c.startswith("z_")]

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

rows = []
for tgt in targets:
    # Ensure target column exists before proceeding
    if tgt not in data.columns:
        print(f"Warning: Target '{tgt}' not found in the dataset. Skipping.")
        continue

    df_t = data[feature_cols + [tgt]].dropna().reset_index(drop=True)
    X = df_t[feature_cols].values
    y = df_t[tgt].values

    # Check if there are features and samples after dropping NaNs
    if X.shape[0] == 0 or X.shape[1] == 0:
        print(f"Warning: No sufficient data or features for target '{tgt}' after handling NaNs. Skipping.")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.001, random_state=42, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN Regression": KNeighborsRegressor(n_neighbors=7),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1)
    }
    if has_xgb:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae, rmse, r2 = eval_metrics(y_te, y_pred)
        rows.append({"Target":tgt, "Model":name, "MAE":mae, "RMSE":rmse, "R2":r2})

baseline_all = pd.DataFrame(rows).sort_values(["Target","MAE"])
baseline_all_path = "/content/baseline_all_models.csv"
baseline_all.to_csv(baseline_all_path, index=False)

print(baseline_all)
print("\nSaved:", baseline_all_path)