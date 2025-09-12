# === Stage 1: Preprocessing for /mnt/data/final.csv ===
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ---------- Dataset Load----------
p = Path("/content/final.csv")
df = pd.read_csv(p)

# ---------- Clean ----------
# The name of columns are getting 
df.columns = [c.strip() for c in df.columns]

# حذف ردیف‌های تکراری
df = df.drop_duplicates().reset_index(drop=True)

# تبدیل ستون‌هایی که باید عددی باشند
numeric_like = [
    "MedianGrossRent","GQPercent","HouseMedianValue","AvgAnnualHouseExpenditure",
    "FamilyMedianIncome","VeteranPercent","MedianAge","DiversityIndex",
    "AvgTravelTimeToWorkMinutes","PovertyPercent","PerCapitaIncome",
    "NumOfHouseHolds","Population","PopulationDensity"
]
for col in numeric_like:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- Targets / ID ----------
targets = [c for c in ["MedianGrossRent","FamilyMedianIncome","PerCapitaIncome","PovertyPercent"] if c in df.columns]
id_cols = [c for c in ["Unnamed: 0","Zip Code Tabulation Area","FIPS Code","Formatted FIPS","City","State","StateCode"] if c in df.columns]

# ---------- Feature set ----------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in set(targets + id_cols)]

# ایمپیوت میانه برای فیچرها
feat_df = df[feature_cols].copy()
impute_map = feat_df.median(numeric_only=True)
feat_df = feat_df.fillna(impute_map)

# نسخهٔ اسکیل‌شده (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_df.values).astype(np.float32)
feat_scaled_df = pd.DataFrame(X_scaled, columns=[f"z_{c}" for c in feature_cols])

# تجمیع برای ذخیره
target_df = df[targets].copy()
clean_full = pd.concat([df[id_cols], feat_df, target_df], axis=1) if id_cols else pd.concat([feat_df, target_df], axis=1)
clean_scaled = pd.concat([df[id_cols], feat_scaled_df, target_df], axis=1) if id_cols else pd.concat([feat_scaled_df, target_df], axis=1)

# ذخیره
clean_full_path   = "/content/final_clean.csv"
clean_scaled_path = "/content/final_clean_scaled.csv"
clean_full.to_csv(clean_full_path, index=False)
clean_scaled.to_csv(clean_scaled_path, index=False)

print("Saved:")
print(" -", clean_full_path)
print(" -", clean_scaled_path)

# گزارش سریع
print("\nTargets:", targets)
print("ID cols:", id_cols)
print("n_features:", len(feature_cols))
print("Rows x Cols:", df.shape)
print("\nMissing (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))
