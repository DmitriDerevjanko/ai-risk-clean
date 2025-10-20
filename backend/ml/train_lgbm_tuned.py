from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None
import lightgbm as lgb
from data_prep import load_gtd_csv, to_monthly_by_region


# ===============================
# CONFIG
# ===============================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gtd.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

HORIZON = 12
TARGET = "incidents_count"

ENSEMBLE_N = 5
ENSEMBLE_SEEDS = [42, 7, 202, 111, 999]


# ===============================
# SMALL HELPERS (ENSEMBLE WRAPPERS)
# ===============================
class EnsembleRegressor:
    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


class EnsembleClassifier:
    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        avg = sum(probs) / len(probs)
        return avg


# ===============================
# METRICS
# ===============================
def rmse(y_true, y_pred):
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred, squared=False))

def smape_weighted(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    weights = np.maximum(y_true, 1.0)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    diff = np.abs(y_true - y_pred) / denom
    return 100 * float(np.average(diff, weights=weights))


# ===============================
# FEATURE CREATION
# ===============================
def _consecutive_zeros_run(x: pd.Series) -> pd.Series:
    run, c = [], 0
    for v in x:
        if v == 0:
            c += 1
        else:
            c = 0
        run.append(c)
    return pd.Series(run, index=x.index)

def create_features(df: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    df = df.copy().sort_values(["region", "month"])

    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["time_index"] = df.groupby("region").cumcount()

    for col in ["global_mean", "global_sum", "ratio_to_global_mean", "ratio_to_global_sum"]:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    global_mean = df.groupby("month")[TARGET].mean().rename("global_mean")
    global_sum = df.groupby("month")[TARGET].sum().rename("global_sum")

    df = df.merge(global_mean, on="month", how="left")
    df = df.merge(global_sum, on="month", how="left")

    df["ratio_to_global_mean"] = df[TARGET] / (df["global_mean"] + 1e-6)
    df["ratio_to_global_sum"] = df[TARGET] / (df["global_sum"] + 1e-6)

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df.groupby("region")[TARGET].shift(lag)

    for w in [3, 6, 12]:
        s = df.groupby("region")[TARGET].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"] = s.rolling(w).std()

    df["diff_1"] = df.groupby("region")[TARGET].diff(1)
    prev = df.groupby("region")[TARGET].shift(1)
    df["growth_rate"] = df["diff_1"] / (prev + 1.0)
    df["trend_24"] = df.groupby("region")[TARGET].transform(
        lambda x: x.rolling(24, min_periods=6).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 5 else 0
        )
    )

    for w in [6, 12]:
        df[f"zero_share_{w}m"] = (
            df.groupby("region")[TARGET]
              .transform(lambda x: x.shift(1).rolling(w).apply(lambda v: np.mean(v == 0), raw=False))
        )
        df[f"volatility_{w}m"] = (
            df.groupby("region")[TARGET]
              .transform(lambda x: x.shift(1).rolling(w).std())
        )

    df["zero_run"] = df.groupby("region")[TARGET].transform(_consecutive_zeros_run)

    df = df.dropna(thresh=int(df.shape[1] * 0.8)).reset_index(drop=True)

    return df



# ===============================
# REGION PROFILE
# ===============================
def region_profile(sub: pd.DataFrame) -> tuple[str, float]:
    tail = sub.sort_values("month").tail(24)
    zero_share = (tail[TARGET] == 0).mean() if len(tail) > 0 else 0.0
    mean_level = tail[TARGET].mean() if len(tail) > 0 else 0.0
    prof = "sparse" if (zero_share > 0.25 and mean_level < 1.5) else "dense"
    return prof, zero_share


# ===============================
# HELPER: SEASONAL BASELINE
# ===============================
def seasonal_naive(sub: pd.DataFrame, idx):
    s = sub.groupby("region")[TARGET].shift(12).loc[idx]
    return s.ffill().fillna(0.0).values


# ===============================
# MAIN TRAINING LOGIC (WITH ENSEMBLE)
# ===============================
def train_one_region(df: pd.DataFrame, region: str):
    sub = df[df["region"] == region].sort_values("month")
    prof, zero_share = region_profile(sub)
    print(f"\n🔧 Training {region} [{prof}]...")

    test_cutoff = sub["month"].max() - pd.DateOffset(months=HORIZON)
    train, test = sub[sub["month"] < test_cutoff], sub[sub["month"] >= test_cutoff]

    if len(train) < 24:
        print(f"⚠️  {region}: too short, skipping.")
        return None

    inner_cut = max(1, len(train) - 12)
    X_tr = train.drop(columns=["region", "month", TARGET]).iloc[:inner_cut]
    y_tr = train[TARGET].values[:inner_cut]
    X_val = train.drop(columns=["region", "month", TARGET]).iloc[inner_cut:]
    y_val = train[TARGET].values[inner_cut:]
    X_test = test.drop(columns=["region", "month", TARGET])
    y_test = test[TARGET].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    y_val_naive = seasonal_naive(sub, X_val.index)
    y_test_naive = seasonal_naive(sub, X_test.index)

    # ---------- DENSE ----------
    if prof == "dense":
        dense_models = []
        for i, seed in enumerate(ENSEMBLE_SEEDS[:ENSEMBLE_N], start=1):
            model = lgb.LGBMRegressor(
                n_estimators=5000,
                learning_rate=0.02,
                num_leaves=128,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                objective="tweedie",
                tweedie_variance_power=1.3,
                metric="rmse",
                verbosity=-1,
                random_state=seed,
                feature_fraction=0.9,
                bagging_fraction=0.9
            )
            model.fit(
                X_tr_s, y_tr,
                eval_set=[(X_val_s, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
            )
            dense_models.append(model)

        ens = EnsembleRegressor(dense_models)

        y_test_pred = np.maximum(ens.predict(X_test_s), 0.0)
        y_test_pred = 0.8 * y_test_pred + 0.2 * y_test_naive

        metrics = {
            "mae": mean_absolute_error(y_test, y_test_pred),
            "rmse": rmse(y_test, y_test_pred),
            "smape": smape_weighted(y_test, y_test_pred),
        }

        pipeline = {
            "type": "dense",
            "model": ens,                 
            "scaler": scaler,
            "features": X_tr.columns.tolist(),
            "profile": prof,
            "ensemble_size": len(dense_models)
        }
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl"))
        print(f"📍 {region} → SMAPE={metrics['smape']:.2f}%  (ensemble {len(dense_models)})")
        return {"region": region, **metrics}

    # ---------- SPARSE ----------
    clfs = []
    for seed in ENSEMBLE_SEEDS[:ENSEMBLE_N]:
        clf = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            metric="binary_logloss",
            verbosity=-1,
            feature_fraction=0.9,
            bagging_fraction=0.9
        )
        clf.fit(X_tr_s, (y_tr > 0).astype(int))
        clfs.append(clf)

    regs = []
    nonzero = y_tr > 0
    y_tr_log = np.log1p(y_tr[nonzero])
    for seed in ENSEMBLE_SEEDS[:ENSEMBLE_N]:
        reg = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            objective="poisson",
            metric="rmse",
            verbosity=-1,
            random_state=seed,
            feature_fraction=0.9,
            bagging_fraction=0.9
        )
        reg.fit(X_tr_s[nonzero], y_tr_log)
        regs.append(reg)

    ens_clf = EnsembleClassifier(clfs)
    ens_reg = EnsembleRegressor(regs)

    def predict_reg(Xs):
        return np.expm1(np.maximum(ens_reg.predict(Xs), 0.0))

    p_val = ens_clf.predict_proba(X_val_s)[:, 1]
    y_val_reg = predict_reg(X_val_s)
    tau_grid = np.arange(0.25, 0.81, 0.05)
    gamma_grid = [0.5, 0.7, 0.85, 0.95]
    best_s, best_tau, best_gamma = 1e9, 0.5, 0.7
    for tau in tau_grid:
        h = (p_val >= tau).astype(float)
        for gamma in gamma_grid:
            pred_val = gamma * (h * y_val_reg) + (1 - gamma) * y_val_naive
            s = smape_weighted(y_val, pred_val)
            if s < best_s:
                best_s, best_tau, best_gamma = s, tau, gamma

    p_test = ens_clf.predict_proba(X_test_s)[:, 1]
    y_test_reg = predict_reg(X_test_s)
    h_test = (p_test >= best_tau).astype(float)
    y_test_pred = best_gamma * (h_test * y_test_reg) + (1 - best_gamma) * y_test_naive

    metrics = {
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": rmse(y_test, y_test_pred),
        "smape": smape_weighted(y_test, y_test_pred),
    }

    pipeline = {
        "type": "sparse",
        "model": {"clf": ens_clf, "reg": ens_reg},   
        "scaler": scaler,
        "features": X_tr.columns.tolist(),
        "tau": best_tau,
        "gamma": best_gamma,
        "profile": prof,
        "ensemble_size": ENSEMBLE_N
    }
    joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl"))
    print(f"📍 {region} → SMAPE={metrics['smape']:.2f}%  τ={best_tau:.2f} γ={best_gamma:.2f} (ensemble {ENSEMBLE_N})")
    return {"region": region, **metrics}


# ===============================
# ADAPTIVE RETRAINING
# ===============================
def retrain_bad_regions(df, df_results, smape_threshold=50):
    bad = df_results[df_results["smape"] > smape_threshold]["region"].tolist()
    if not bad:
        print("\n✅ No weak regions to retrain.")
        return df_results
    print(f"\n🔁 Re-training {len(bad)} regions with SMAPE>{smape_threshold}...\n")
    improved = []
    for region in tqdm(bad, desc="♻️ Retraining"):
        r = train_one_region(df, region)
        if r: improved.append(r)
    df_new = pd.DataFrame(improved)
    df_final = (
        df_results.set_index("region")
        .combine_first(df_new.set_index("region"))
        .reset_index()
    )
    print("\n✅ Adaptive retraining done.")
    return df_final


# ===============================
# MAIN
# ===============================
def main():
    print(f"📦 Loading data from: {DATA_PATH}")
    raw = load_gtd_csv(DATA_PATH)
    monthly = to_monthly_by_region(raw)
    df = create_features(monthly)

    regions = sorted(df["region"].dropna().unique().tolist())
    print(f"\n🌍 Found {len(regions)} regions. Training hybrid v3.4 (ensemble x{ENSEMBLE_N})...\n")

    results = []
    for region in tqdm(regions, desc="⚙️ First pass"):
        r = train_one_region(df, region)
        if r: results.append(r)

    df_results = pd.DataFrame(results).sort_values("smape")
    out1 = os.path.join(OUTPUTS_DIR, f"metrics_v3p4_first_{TARGET}.csv")
    df_results.to_csv(out1, index=False)
    print(f"\n✅ First pass done. Results saved: {out1}")
    print(df_results.to_string(index=False))
    print(f"\nMean SMAPE (first): {df_results['smape'].mean():.2f}%")

    df_final = retrain_bad_regions(df, df_results, smape_threshold=50)
    out2 = os.path.join(OUTPUTS_DIR, f"metrics_v3p4_final_{TARGET}.csv")
    df_final.to_csv(out2, index=False)

    print(f"\n🏁 FINAL SUMMARY:")
    print(df_final.to_string(index=False))
    print(f"\nMean SMAPE (final): {df_final['smape'].mean():.2f}%")


if __name__ == "__main__":
    main()
