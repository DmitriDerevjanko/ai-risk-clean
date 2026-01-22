# train_lgbm_tuned_fixed_pooled.py
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None
import lightgbm as lgb
from data_prep import load_gtd_csv, to_monthly_by_region
from scipy.stats import pearsonr

from packaging import version
import sklearn
import shutil
from pathlib import Path

from collections import defaultdict
import json
import math

import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gtd.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DIAG_DIR = os.path.join(OUTPUTS_DIR, "diag")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

HORIZON = 12
TARGET = "incidents_count"

# final holdout years (recommended 3). If None, uses HORIZON months for test cutoff.
FINAL_HOLDOUT_YEARS = 3
TEST_MONTHS = FINAL_HOLDOUT_YEARS * 12 if FINAL_HOLDOUT_YEARS is not None else HORIZON

ENSEMBLE_N = 5
ENSEMBLE_SEEDS = [42, 7, 202, 111, 999]


# ===============================
# HELPERS
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
        # ensure we stack numpy arrays and take mean along models axis
        probs = np.stack([m.predict_proba(X) for m in self.models], axis=0)  # shape (M, n, 2)
        avg = probs.mean(axis=0)  # shape (n, 2)
        return avg


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
    """
    Create time-series features per region.
    Uses shift(1) for rolling features to avoid using current/future values.
    After feature creation, we drop rows with too many missing values, then fill remaining NaNs with zeros.
    """
    df = df.copy().sort_values(["region", "month"])
    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["time_index"] = df.groupby("region").cumcount()

    # lag features
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df.groupby("region")[TARGET].shift(lag)

    # rolling features: always shift(1) first so rolls do not include current row
    for w in [3, 6, 12]:
        df[f"roll_mean_{w}"] = (
            df.groupby("region")[TARGET].shift(1).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            df.groupby("region")[TARGET].shift(1).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
        )

    df["diff_1"] = df.groupby("region")[TARGET].diff(1)
    prev = df.groupby("region")[TARGET].shift(1)
    df["growth_rate"] = df["diff_1"] / (prev + 1.0)

    # trend estimated on past window (rolling with min_periods)
    def _trend(series):
        def _fit(v):
            if len(v) > 5:
                return float(np.polyfit(np.arange(len(v)), v, 1)[0])
            return 0.0
        return series.shift(0).rolling(24, min_periods=6).apply(_fit, raw=False)

    df["trend_24"] = df.groupby("region")[TARGET].transform(_trend)

    # zero-share and volatility over past windows: shift(1) then rolling
    for w in [6, 12]:
        df[f"zero_share_{w}m"] = (
            df.groupby("region")[TARGET].shift(1).rolling(w, min_periods=1).apply(lambda v: np.mean(v == 0), raw=False).reset_index(level=0, drop=True)
        )
        df[f"volatility_{w}m"] = (
            df.groupby("region")[TARGET].shift(1).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
        )

    df["zero_run"] = df.groupby("region")[TARGET].transform(_consecutive_zeros_run)

    # drop rows that are mostly missing but be less aggressive than before, then fill remaining NaNs.
    thresh = int(df.shape[1] * 0.5)  # require at least 50% non-null
    df = df.dropna(thresh=thresh).reset_index(drop=True)
    # fill remaining NaNs with 0 (safe for counts/lag features) and small numeric imputation
    df = df.fillna(0.0)
    return df


# ===============================
# REGION PROFILE / BASELINES
# ===============================
def region_profile(sub: pd.DataFrame) -> tuple[str, float]:
    tail = sub.sort_values("month").tail(24)
    zero_share = (tail[TARGET] == 0).mean() if len(tail) > 0 else 0.0
    mean_level = tail[TARGET].mean() if len(tail) > 0 else 0.0
    prof = "sparse" if (zero_share > 0.25 and mean_level < 1.5) else "dense"
    return prof, zero_share


def seasonal_naive(sub: pd.DataFrame, idx):
    """
    Seasonal naive: take value from same month 12 months ago for the rows specified by idx.
    sub: dataframe for a single region (sorted by month).
    idx: index (can be a list/Index of rows inside sub or global indices) - we use reindex to be robust.
    Returns array aligned with idx order.
    """
    if sub.empty:
        return np.zeros(len(idx))
    shifted = sub.set_index("month")[TARGET].shift(12)  # index=month
    # get months for requested rows robustly
    months = sub.reindex(idx)["month"]
    # reindex shifted by months values; if missing -> NaN -> ffill then fill 0
    preds = shifted.reindex(months.values).ffill().fillna(0.0).values
    return preds


def heuristic_forecast_for_sparse(train_sub: pd.DataFrame, horizon=HORIZON):
    """
    Heuristic forecast for sparse series. IMPORTANT: expects only training rows (no test rows).
    """
    sub = train_sub
    if len(sub) >= 12:
        last = sub[TARGET].iloc[-12:].values
        preds = np.tile(last, int(np.ceil(horizon / 12)))[:horizon]
        return preds
    if len(sub) >= 3:
        return np.repeat(sub[TARGET].iloc[-min(6, len(sub)):].mean(), horizon)
    return np.zeros(horizon)


# ===============================
# UTILITY: DETECT HOLDOUT / OVERLAP
# ===============================
def detect_train_test_overlap(df: pd.DataFrame, test_months=TEST_MONTHS):
    """
    Returns list of problems: (region, problem_type, details...)
    problem_type can be: 'too_short', 'wrong_test_size', 'overlap'
    """
    problems = []
    for region in sorted(df["region"].dropna().unique().tolist()):
        sub = df[df["region"] == region].sort_values("month").reset_index(drop=True)
        if len(sub) < test_months + 1:
            problems.append((region, "too_short", len(sub)))
            continue
        test = sub.tail(test_months)
        train = sub.iloc[:-test_months]
        if len(test) != test_months:
            problems.append((region, "wrong_test_size", len(test)))
        # ensure strict ordering: last train month < first test month
        if not train.empty and train["month"].max() >= test["month"].min():
            problems.append((region, "overlap", str(train["month"].max()), str(test["month"].min())))
    return problems


# ===============================
# TRAIN ONE REGION
# ===============================
def train_one_region(df: pd.DataFrame, region: str, save_pipeline=True):
    sub = df[df["region"] == region].sort_values("month").reset_index(drop=True)
    prof, zero_share = region_profile(sub)
    print(f"\n🔧 Training {region} [{prof}]...")

    # --- FIX: use .tail to guarantee exact TEST_MONTHS in holdout ---
    if len(sub) < TEST_MONTHS + 24:
        # need at least some train length (>=24) + holdout
        test = sub.tail(TEST_MONTHS)
        train = sub.iloc[:-TEST_MONTHS]
    else:
        test = sub.tail(TEST_MONTHS)
        train = sub.iloc[:-TEST_MONTHS]

    if len(train) < 24:
        print(f"⚠️  {region}: too short, skipping.")
        return None

    # inner split: last 12 months of train used as validation
    inner_cut = max(1, len(train) - 12)
    X_tr = train.drop(columns=["region", "month", TARGET]).iloc[:inner_cut]
    y_tr = train[TARGET].values[:inner_cut]
    X_val = train.drop(columns=["region", "month", TARGET]).iloc[inner_cut:]
    y_val = train[TARGET].values[inner_cut:]
    X_test = test.drop(columns=["region", "month", TARGET])
    y_test = test[TARGET].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    # handle case X_tr has only one column etc
    X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_tr_s.shape[1]))
    X_test_s = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, X_tr_s.shape[1]))

    # compute naive preds using seasonal_naive on region-only dataframe
    y_val_naive = seasonal_naive(sub, X_val.index)
    y_test_naive = seasonal_naive(sub, X_test.index)

    if prof == "dense":
        dense_models = []
        for i, seed in enumerate(ENSEMBLE_SEEDS[:ENSEMBLE_N], start=1):
            model = lgb.LGBMRegressor(
                n_estimators=2000, learning_rate=0.02, num_leaves=128,
                subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                objective="tweedie", tweedie_variance_power=1.3,
                metric="rmse", verbosity=-1, random_state=seed,
                feature_fraction=0.9, bagging_fraction=0.9, n_jobs=-1
            )
            # early stopping uses validation; if no val rows exist, skip eval_set
            if len(X_val_s) > 0:
                model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], eval_metric="rmse",
                          callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
            else:
                model.fit(X_tr_s, y_tr)
            dense_models.append(model)

        ens = EnsembleRegressor(dense_models)
        y_test_pred = np.maximum(ens.predict(X_test_s), 0.0)
        # blend with seasonal naive
        y_test_pred = 0.8 * y_test_pred + 0.2 * y_test_naive

        metrics = {"mae": mean_absolute_error(y_test, y_test_pred),
                   "rmse": rmse(y_test, y_test_pred),
                   "smape": smape_weighted(y_test, y_test_pred)}
        pipeline = {"type": "dense", "model": ens, "scaler": scaler, "features": X_tr.columns.tolist(),
                    "profile": prof, "ensemble_size": len(dense_models),
                    "training_end": train["month"].max()}
        if save_pipeline:
            joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl"))
        print(f"📍 {region} → SMAPE={metrics['smape']:.2f}%  (ensemble {len(dense_models)})")
        return {"region": region, **metrics}

    # sparse branch
    clfs = []
    for seed in ENSEMBLE_SEEDS[:ENSEMBLE_N]:
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=64,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            random_state=seed, metric="binary_logloss", verbosity=-1,
            feature_fraction=0.9, bagging_fraction=0.9, n_jobs=-1
        )
        clf.fit(X_tr_s, (y_tr > 0).astype(int))
        clfs.append(clf)

    nonzero_mask = y_tr > 0
    if nonzero_mask.sum() < 3:
        print(f"ℹ️  {region}: too few nonzero events ({nonzero_mask.sum()}), using heuristic fallback.")
        # IMPORTANT: use only TRAIN for heuristic to avoid leakage
        y_test_pred = heuristic_forecast_for_sparse(train)[:len(test)]
        metrics = {"mae": mean_absolute_error(test[TARGET].values, y_test_pred),
                   "rmse": rmse(test[TARGET].values, y_test_pred),
                   "smape": smape_weighted(test[TARGET].values, y_test_pred)}
        pipeline = {"type": "sparse_heuristic", "scaler": scaler, "features": X_tr.columns.tolist(), "profile": "sparse",
                    "training_end": train["month"].max()}
        if save_pipeline:
            joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl"))
        print(f"📍 {region} → SMAPE={metrics['smape']:.2f}% (heuristic)")
        return {"region": region, **metrics}

    regs = []
    for seed in ENSEMBLE_SEEDS[:ENSEMBLE_N]:
        reg = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.03, num_leaves=64,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            objective="poisson", metric="rmse", verbosity=-1, random_state=seed,
            feature_fraction=0.9, bagging_fraction=0.9, n_jobs=-1
        )
        reg.fit(X_tr_s[nonzero_mask], y_tr[nonzero_mask])
        regs.append(reg)

    ens_clf = EnsembleClassifier(clfs)
    ens_reg = EnsembleRegressor(regs)

    def predict_reg(Xs):
        return np.maximum(ens_reg.predict(Xs), 0.0)

    p_val = ens_clf.predict_proba(X_val_s)[:, 1] if len(X_val_s) > 0 else np.array([])
    y_val_reg = predict_reg(X_val_s) if len(X_val_s) > 0 else np.array([])

    # grid-search tau/gamma on validation if available, otherwise fallback to defaults
    tau_grid = np.arange(0.25, 0.81, 0.05)
    gamma_grid = [0.5, 0.7, 0.85, 0.95]
    best_s, best_tau, best_gamma = 1e9, 0.5, 0.7
    if len(X_val_s) > 0:
        for tau in tau_grid:
            h = (p_val >= tau).astype(float)
            for gamma in gamma_grid:
                pred_val = gamma * (h * y_val_reg) + (1 - gamma) * y_val_naive
                s = smape_weighted(y_val, pred_val)
                if s < best_s:
                    best_s, best_tau, best_gamma = s, tau, gamma

    # test-time
    p_test = ens_clf.predict_proba(X_test_s)[:, 1] if len(X_test_s) > 0 else np.array([])
    y_test_reg = predict_reg(X_test_s) if len(X_test_s) > 0 else np.array([])
    h_test = (p_test >= best_tau).astype(float) if p_test.size > 0 else np.zeros_like(y_test_reg)
    y_test_pred = best_gamma * (h_test * y_test_reg) + (1 - best_gamma) * y_test_naive

    metrics = {"mae": mean_absolute_error(y_test, y_test_pred),
               "rmse": rmse(y_test, y_test_pred),
               "smape": smape_weighted(y_test, y_test_pred)}
    pipeline = {"type": "sparse", "model": {"clf": ens_clf, "reg": ens_reg}, "scaler": scaler,
                "features": X_tr.columns.tolist(), "tau": best_tau, "gamma": best_gamma,
                "profile": "sparse", "ensemble_size": ENSEMBLE_N,
                "training_end": train["month"].max()}
    if save_pipeline:
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl"))
    print(f"📍 {region} → SMAPE={metrics['smape']:.2f}%  τ={best_tau:.2f} γ={best_gamma:.2f} (ensemble {ENSEMBLE_N})")
    return {"region": region, **metrics}


# ===============================
# POOLED GLOBAL + LOCAL ADAPTERS — FIXED (train only on per-region train rows)
# ===============================
def train_pooled_global_then_local(df, bad_regions, features, target=TARGET, models_dir=MODELS_DIR):
    df_proc = df.copy().sort_values(["region", "month"]).reset_index(drop=True)

    # --- FIX: create explicit per-region is_test mask using tail(TEST_MONTHS) ---
    def mark_is_test(group):
        is_test = pd.Series(False, index=group.index)
        if len(group) >= TEST_MONTHS:
            is_test.loc[group.tail(TEST_MONTHS).index] = True
        else:
            # if too short, mark all as non-test (will be handled upstream)
            is_test.loc[:] = False
        return is_test

    is_test_mask = pd.Series(False, index=df_proc.index)
    for _, group in df_proc.groupby("region"):
        if len(group) >= TEST_MONTHS:
            is_test_mask.loc[group.tail(TEST_MONTHS).index] = True
    train_mask = ~is_test_mask

    # training dataframe for pooled global model (EXCLUDES all holdouts)
    pooled_train = df_proc[train_mask].copy()
    if pooled_train.empty:
        raise ValueError("No pooled training rows (check TEST_MONTHS).")

    X_train = pooled_train[features + ["region"]]
    y_train = pooled_train[target].values

    numeric_feats = [c for c in features if c != "region"]
    cat_feats = ["region"]

    # OneHotEncoder compat
    ohe_kwargs = {"handle_unknown": "ignore"}
    skl_ver = version.parse(sklearn.__version__)
    if skl_ver >= version.parse("1.2"):
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    preproc = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(**ohe_kwargs), cat_feats)
    ], remainder="drop")

    X_train_proc = preproc.fit_transform(X_train)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    # train global on pooled_train only (no holdout rows)
    global_model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.3,
        n_estimators=2000, learning_rate=0.03, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    global_model.fit(X_train_proc, y_train)

    # compute global_pred for all rows (including test) — but store preds only for train rows
    X_all = df_proc[features + ["region"]]
    X_all_proc = preproc.transform(X_all)
    if hasattr(X_all_proc, "toarray"):
        X_all_proc = X_all_proc.toarray()
    preds_all = global_model.predict(X_all_proc)

    # SAFE: fill global_pred only for training rows to avoid accidental leakage if code later reuses resid
    df_proc["global_pred"] = np.nan
    df_proc.loc[train_mask, "global_pred"] = preds_all[train_mask.values]
    # resid only for train rows (safe)
    df_proc["resid"] = np.nan
    df_proc.loc[train_mask, "resid"] = df_proc.loc[train_mask, target] - df_proc.loc[train_mask, "global_pred"]

    # local adapters: train each on that region's training rows only
    local_adapters = {}
    for region in bad_regions:
        sub_region = df_proc[df_proc["region"] == region]
        # sub_train defined by explicit train_mask
        sub_train = sub_region[~is_test_mask.loc[sub_region.index]]
        # ensure resid exists and dropna
        sub_train = sub_train.dropna(subset=["resid"])
        if len(sub_train) < 12:
            print(f"Skip local adapter for {region}: too few train rows ({len(sub_train)})")
            continue
        X_sub = sub_train[features + ["region"]]
        X_sub_proc = preproc.transform(X_sub)
        if hasattr(X_sub_proc, "toarray"):
            X_sub_proc = X_sub_proc.toarray()
        y_resid = sub_train["resid"].values

        adapter = lgb.LGBMRegressor(
            objective="regression", n_estimators=500, learning_rate=0.05, num_leaves=16,
            random_state=7, n_jobs=-1
        )
        adapter.fit(X_sub_proc, y_resid)
        local_adapters[region] = adapter
        print(f"Trained local adapter for {region} ({len(sub_train)} train rows)")

    joblib.dump(
        {"preproc": preproc, "global_model": global_model, "local_adapters": local_adapters},
        os.path.join(models_dir, "pooled_global_with_local_adapters.pkl"),
        compress=3
    )
    return preproc, global_model, local_adapters

def cleanup_models(models_dir: str | Path, keep: int = 12, archive: bool = True):
    """
    Оставляет последние `keep` pipeline-файлов в models_dir, остальные перемещает в models_dir/archive.
    Для сортировки сначала пробует читать `training_end` внутри пкл; если не получилось — использует mtime.
    """
    models_dir = Path(models_dir)
    MODE_PATTERNS = ["pipeline__*__*.pkl", "*pooled*.pkl", "pooled_global_with_local_adapters.pkl"]
    # collect unique files
    files = []
    for pat in MODE_PATTERNS:
        for p in models_dir.glob(pat):
            if p.is_file() and p not in files:
                files.append(p)

    if not files:
        print("cleanup_models: нет pipeline-файлов для обработки.")
        return

    rows = []
    for f in files:
        sort_key = None
        try:
            # try to load and read training_end if present
            pip = joblib.load(f)
            te = pip.get("training_end", None)
            if te is not None:
                sort_key = pd.to_datetime(te)
        except Exception:
            sort_key = None
        # fallback: file mtime
        if sort_key is None or pd.isna(sort_key):
            try:
                sort_key = pd.to_datetime(f.stat().st_mtime, unit="s")
            except Exception:
                sort_key = pd.NaT
        rows.append((f, sort_key))

    # sort newest first: rows with valid datetimes sorted descending, then NaT at end
    rows_sorted = sorted(rows, key=lambda x: (pd.isna(x[1]), x[1]), reverse=True)
    keep_files = [r[0] for r in rows_sorted[:keep]]
    delete_files = [r[0] for r in rows_sorted[keep:]]

    if not delete_files:
        print(f"cleanup_models: всего {len(files)} файлов — нечего удалять (keep={keep}).")
        return

    archive_dir = models_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    print(f"cleanup_models: оставляем {len(keep_files)} файлов, перемещаем {len(delete_files)} в {archive_dir}")
    for p in keep_files:
        print("  KEEP ", p.name)
    for p in delete_files:
        print("  MOVE ", p.name)

    # perform move
    for p in delete_files:
        try:
            target = archive_dir / p.name
            if archive:
                shutil.move(str(p), str(target))
            else:
                p.unlink()
        except Exception as e:
            print("cleanup_models: ошибка при обработке", p, e)

    print("cleanup_models: завершено.")

def pooled_predict_region(preproc, global_model, local_adapters, df_region, features):
    X = df_region[features + ["region"]]
    X_proc = preproc.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    g = global_model.predict(X_proc)
    region = df_region["region"].iloc[0]
    if region in local_adapters:
        adj = local_adapters[region].predict(X_proc)
    else:
        adj = np.zeros_like(g)
    pred = np.maximum(g + adj, 0.0)
    return pred


# ===============================
# VALIDATION / HOLDOUT ANALYSIS
# ===============================
def _metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "smape": float(smape_weighted(y_true, y_pred))
    }


def validate_final_holdout_for_region(df_region, pipeline_path=None, pooled_preproc=None, pooled_global=None, pooled_local=None, features=None, test_months=TEST_MONTHS):
    """
    Возвращает dict с метриками:
      - overall (metrics on entire holdout)
      - per_year: list of (year, metrics)
      - train_val_info: approx train metrics if computable
      - flags: 'ok' / 'warning' / 'overfit'
    """
    sub = df_region.sort_values("month").reset_index(drop=True)

    # --- FIX: use tail to ensure exact TEST_MONTHS rows in test_sub
    if len(sub) < test_months + 1:
        test_sub = sub.tail(test_months)
        train_sub = sub.iloc[:-test_months]
    else:
        test_sub = sub.tail(test_months)
        train_sub = sub.iloc[:-test_months]

    res = {"region": sub["region"].iloc[0], "n_test_rows": len(test_sub)}
    if len(test_sub) == 0:
        res["note"] = "no_holdout_rows"
        return res

    # baseline seasonal naive
    base_pred = seasonal_naive(sub, test_sub.index)

    # try to load per-region pipeline if available
    model_pred = None
    train_val_info = {"train_smape": None, "val_smape": None, "training_end": None}
    if pipeline_path and os.path.exists(pipeline_path):
        try:
            pip = joblib.load(pipeline_path)
            train_val_info["training_end"] = pip.get("training_end", None)
            if pip["type"] == "dense":
                X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                base_model_pred = np.maximum(pip["model"].predict(X_test_s), 0.0)
                model_pred = 0.8 * base_model_pred + 0.2 * seasonal_naive(sub, test_sub.index)
            elif pip["type"] == "sparse":
                X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                p_test = pip["model"]["clf"].predict_proba(X_test_s)[:, 1]
                y_test_reg = np.maximum(pip["model"]["reg"].predict(X_test_s), 0.0)
                h_test = (p_test >= pip["tau"]).astype(float)
                model_pred = pip["gamma"] * (h_test * y_test_reg) + (1 - pip["gamma"]) * seasonal_naive(sub, test_sub.index)
            elif pip["type"].startswith("pooled"):
                if "preproc" in pip and "global_model" in pip:
                    Xp = test_sub[pip["features"] + ["region"]]
                    Xp_proc = pip["preproc"].transform(Xp)
                    if hasattr(Xp_proc, "toarray"): Xp_proc = Xp_proc.toarray()
                    g = pip["global_model"].predict(Xp_proc)
                    adj = pip.get("local_adapter", None).predict(Xp_proc) if pip.get("local_adapter", None) else np.zeros_like(g)
                    model_pred = np.maximum(g + adj, 0.0)
                else:
                    model_pred = seasonal_naive(sub, test_sub.index)
            else:
                model_pred = seasonal_naive(sub, test_sub.index)
        except Exception:
            model_pred = seasonal_naive(sub, test_sub.index)
    else:
        # try pooled_predict if pooled objects passed
        if pooled_preproc is not None and pooled_global is not None and features is not None:
            try:
                model_pred = pooled_predict_region(pooled_preproc, pooled_global, pooled_local or {}, test_sub, features)
            except Exception:
                model_pred = seasonal_naive(sub, test_sub.index)
        else:
            model_pred = seasonal_naive(sub, test_sub.index)

    # overall metrics
    res["baseline"] = _metrics(test_sub[TARGET].values, base_pred)
    res["model"] = _metrics(test_sub[TARGET].values, model_pred)

    # per-year breakdown in holdout
    per_year = defaultdict(list)
    test_idx_list = list(test_sub.index)
    for i_local, idx in enumerate(test_idx_list):
        row = sub.loc[idx]
        y = row[TARGET]
        per_year[row["month"].year].append((y, float(model_pred[i_local]), float(base_pred[i_local])))

    per_year_metrics = []
    for year in sorted(per_year.keys()):
        arr = np.array([[x[0], x[1], x[2]] for x in per_year[year]])
        y_true = arr[:, 0]
        y_model = arr[:, 1]
        y_base = arr[:, 2]
        per_year_metrics.append({
            "year": int(year),
            **_metrics(y_true, y_model),
            "baseline_smape": float(smape_weighted(y_true, y_base)),
            "n": int(len(y_true))
        })
    res["per_year"] = per_year_metrics

    # compare with train/val if available: approximate by evaluating on last 12 months of train
    if len(train_sub) >= 12:
        train_hold = train_sub.tail(12)
        try:
            if pipeline_path and os.path.exists(pipeline_path):
                pip = joblib.load(pipeline_path)
                if pip["type"] == "dense":
                    X_tr_hold_s = pip["scaler"].transform(train_hold[pip["features"]])
                    pred_tr_hold = np.maximum(pip["model"].predict(X_tr_hold_s), 0.0)
                    pred_tr_hold = 0.8 * pred_tr_hold + 0.2 * seasonal_naive(sub, train_hold.index)
                elif pip["type"] == "sparse":
                    X_tr_hold_s = pip["scaler"].transform(train_hold[pip["features"]])
                    p_tr = pip["model"]["clf"].predict_proba(X_tr_hold_s)[:, 1]
                    y_tr_reg = np.maximum(pip["model"]["reg"].predict(X_tr_hold_s), 0.0)
                    h_tr = (p_tr >= pip["tau"]).astype(float)
                    pred_tr_hold = pip["gamma"] * (h_tr * y_tr_reg) + (1 - pip["gamma"]) * seasonal_naive(sub, train_hold.index)
                else:
                    pred_tr_hold = seasonal_naive(sub, train_hold.index)
            elif pooled_preproc is not None and pooled_global is not None and features is not None:
                pred_tr_hold = pooled_predict_region(pooled_preproc, pooled_global, pooled_local or {}, train_hold, features)
            else:
                pred_tr_hold = seasonal_naive(sub, train_hold.index)
            train_metrics = _metrics(train_hold[TARGET].values, pred_tr_hold)
            train_val_info["train_smape"] = train_metrics["smape"]
        except Exception:
            train_val_info["train_smape"] = None
    res["train_val_info"] = train_val_info

    # flagging heuristics: if test_smape much worse than train_smape -> overfit
    test_smape = res["model"]["smape"]
    train_smape = train_val_info.get("train_smape", None)
    flag = "ok"
    if train_smape is not None:
        # heuristics:
        if test_smape > train_smape * 1.5 or (test_smape - train_smape) > 20:
            flag = "overfit"
        elif test_smape > train_smape * 1.25 or (test_smape - train_smape) > 10:
            flag = "warning"
    else:
        # no train comparison; use baseline comparison
        if res["model"]["smape"] > res["baseline"]["smape"] * 1.5 and res["model"]["smape"] > 30:
            flag = "warning"
    res["flag"] = flag
    return res
def save_metrics_summary(df: pd.DataFrame, models_dir: str, features: list, outputs_dir: str, test_months: int = TEST_MONTHS, out_fname: str = None):
    """
    Для каждого региона:
      - Берёт последние `test_months` строк как holdout
      - Формирует прогноз с существующего pipeline (если есть) или seasonal_naive
      - Считает mae, rmse, smape и pearson correlation (если возможно)
      - Сохраняет строку в CSV outputs_dir/out_fname (если out_fname None -> metrics_summary_incidents_count.csv)
    """
    if out_fname is None:
        out_fname = f"metrics_summary_{TARGET}.csv"
    out_path = os.path.join(outputs_dir, out_fname)

    regions = sorted(df["region"].dropna().unique().tolist())
    rows = []
    for region in regions:
        sub = df[df["region"] == region].sort_values("month").reset_index(drop=True)
        if sub.empty:
            continue
        # Определяем holdout последними test_months
        test_sub = sub.tail(test_months)
        train_sub = sub.iloc[:-test_months] if len(sub) >= test_months else sub.iloc[:0]
        if len(test_sub) == 0:
            continue

        # попробуем загрузить per-region pipeline
        pipeline_path = os.path.join(models_dir, f"pipeline__{region.replace(' ', '_')}__.pkl")
        model_pred = None
        try:
            if os.path.exists(pipeline_path):
                pip = joblib.load(pipeline_path)
                if pip.get("type") == "dense":
                    X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                    base_model_pred = np.maximum(pip["model"].predict(X_test_s), 0.0)
                    model_pred = 0.8 * base_model_pred + 0.2 * seasonal_naive(sub, test_sub.index)
                elif pip.get("type") == "sparse":
                    X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                    p_test = pip["model"]["clf"].predict_proba(X_test_s)[:, 1]
                    y_test_reg = np.maximum(pip["model"]["reg"].predict(X_test_s), 0.0)
                    h_test = (p_test >= pip.get("tau", 0.5)).astype(float)
                    model_pred = pip.get("gamma", 0.8) * (h_test * y_test_reg) + (1 - pip.get("gamma", 0.8)) * seasonal_naive(sub, test_sub.index)
                elif pip.get("type", "").startswith("pooled"):
                    if "preproc" in pip and "global_model" in pip:
                        Xp = test_sub[pip["features"] + ["region"]]
                        Xp_proc = pip["preproc"].transform(Xp)
                        if hasattr(Xp_proc, "toarray"):
                            Xp_proc = Xp_proc.toarray()
                        g = pip["global_model"].predict(Xp_proc)
                        adj = pip.get("local_adapter", None).predict(Xp_proc) if pip.get("local_adapter", None) else np.zeros_like(g)
                        model_pred = np.maximum(g + adj, 0.0)
                    else:
                        model_pred = seasonal_naive(sub, test_sub.index)
                else:
                    model_pred = seasonal_naive(sub, test_sub.index)
            else:
                # нет локальной модели -> seasonal naive
                model_pred = seasonal_naive(sub, test_sub.index)
        except Exception:
            model_pred = seasonal_naive(sub, test_sub.index)

        # гарантируем numpy array
        y_true = test_sub[TARGET].values.astype(float)
        y_pred = np.asarray(model_pred, dtype=float)

        # compute metrics
        try:
            mae_v = float(np.mean(np.abs(y_true - y_pred)))
        except Exception:
            mae_v = None
        try:
            rmse_v = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        except Exception:
            rmse_v = None
        try:
            smape_v = float(smape_weighted(y_true, y_pred))
        except Exception:
            smape_v = None
        # pearson
        corr_v = None
        try:
            if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
                corr_v = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            corr_v = None

        rows.append({
            "region": region,
            "n_test_rows": int(len(y_true)),
            "mae": mae_v,
            "rmse": rmse_v,
            "smape": smape_v,
            "corr": corr_v
        })

    df_out = pd.DataFrame(rows).sort_values("smape")
    df_out.to_csv(out_path, index=False)
    print(f"✅ Metrics summary saved: {out_path}")
    return out_path


def validate_all_regions_holdout(df, models_dir=MODELS_DIR, years=FINAL_HOLDOUT_YEARS, features=None, pooled_objs=None, mode="local", out_csv=None):
    """
    mode: "local" - validate using saved per-region pipelines (pipeline__region__.pkl)
          "pooled" - validate using pooled objects passed in pooled_objs (preproc, global, local)
    """
    regions = sorted(df["region"].dropna().unique().tolist())
    rows = []
    pooled_preproc, pooled_global, pooled_local = (None, None, None)
    if pooled_objs:
        pooled_preproc, pooled_global, pooled_local = pooled_objs
    for region in tqdm(regions, desc=f"Validating holdouts ({mode})"):
        sub = df[df["region"] == region].sort_values("month").reset_index(drop=True)
        pipeline_path = os.path.join(models_dir, f"pipeline__{region.replace(' ', '_')}__.pkl")
        if mode == "local":
            r = validate_final_holdout_for_region(sub, pipeline_path=pipeline_path, pooled_preproc=None, pooled_global=None, pooled_local=None, features=features, test_months=years*12)
        elif mode == "pooled":
            r = validate_final_holdout_for_region(sub, pipeline_path=None, pooled_preproc=pooled_preproc, pooled_global=pooled_global, pooled_local=pooled_local, features=features, test_months=years*12)
        else:
            raise ValueError("mode must be 'local' or 'pooled'")

        # flatten for CSV: overall model/baseline metrics + flag + per_year details as JSON-string
        per_year_str = ""
        try:
            per_year_str = json.dumps(r.get("per_year", []))
        except Exception:
            per_year_str = str(r.get("per_year", []))
        rows.append({
            "region": r.get("region"),
            "n_test_rows": r.get("n_test_rows"),
            "model_smape": r.get("model", {}).get("smape"),
            "baseline_smape": r.get("baseline", {}).get("smape"),
            "model_mae": r.get("model", {}).get("mae"),
            "flag": r.get("flag"),
            "per_year": per_year_str
        })
    df_val = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.join(OUTPUTS_DIR, f"holdout_validation_{mode}_{years}y.csv")
    df_val.to_csv(out_csv, index=False)
    print(f"Holdout validation ({mode}) saved to: {out_csv}")
    return df_val


# ===============================
# ROLLING BACKTEST / DIAGNOSTICS
# ===============================
def rolling_backtest_region(df_region, features, train_window_months=36, horizon=12, step_months=6, model_trainer_callable=None):
    """
    Rolling-origin / expanding-window backtest.
    Returns pandas DataFrame of metrics per cutoff.
    """
    rows = []
    sub = df_region.sort_values("month").reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(rows)
    min_month = sub["month"].min()
    max_month = sub["month"].max()

    cur_cutoff = min_month + pd.DateOffset(months=train_window_months)
    while cur_cutoff + pd.DateOffset(months=horizon) <= max_month:
        train_mask = sub["month"] < cur_cutoff
        test_mask = (sub["month"] >= cur_cutoff) & (sub["month"] < cur_cutoff + pd.DateOffset(months=horizon))
        train = sub[train_mask]
        test = sub[test_mask]
        if len(train) < 12 or len(test) == 0:
            cur_cutoff = cur_cutoff + pd.DateOffset(months=step_months)
            continue

        # baseline
        baseline = seasonal_naive(sub.loc[train.index.union(test.index)], test.index)

        # model: if trainer provided, fit on train & predict on test
        if model_trainer_callable is not None:
            try:
                predict_fn = model_trainer_callable(train, features)
                y_pred = predict_fn(test[features + ["region"]])
            except Exception:
                y_pred = baseline
        else:
            y_pred = baseline

        metrics = {
            "cutoff": str(cur_cutoff.date()),
            "train_end": str(train["month"].max().date()),
            "n_train": len(train),
            "n_test": len(test),
            "mae": float(mean_absolute_error(test[TARGET].values, y_pred)),
            "rmse": float(rmse(test[TARGET].values, y_pred)),
            "smape": float(smape_weighted(test[TARGET].values, y_pred))
        }
        rows.append(metrics)
        cur_cutoff = cur_cutoff + pd.DateOffset(months=step_months)
    return pd.DataFrame(rows)


def pipeline_trainer_from_saved(pipeline_path):
    """
    Returns trainer(train_df, features) -> predict_fn(X_df)
    predict_fn expects X_df with features + ['region'] and returns numpy array preds aligned to X_df rows.
    """
    def trainer(train_df, features):
        pip = joblib.load(pipeline_path)
        def predict_fn(X_df):
            # X_df is rows to predict (with 'region' col)
            if pip["type"] == "dense":
                Xs = pip["scaler"].transform(X_df[pip["features"]])
                pred = np.maximum(pip["model"].predict(Xs), 0.0)
                pred = 0.8 * pred + 0.2 * seasonal_naive(pd.concat([train_df, X_df], sort=False), X_df.index)
                return pred
            elif pip["type"] == "sparse":
                Xs = pip["scaler"].transform(X_df[pip["features"]])
                p = pip["model"]["clf"].predict_proba(Xs)[:, 1]
                yreg = np.maximum(pip["model"]["reg"].predict(Xs), 0.0)
                h = (p >= pip["tau"]).astype(float)
                pred = pip["gamma"] * (h * yreg) + (1 - pip["gamma"]) * seasonal_naive(pd.concat([train_df, X_df], sort=False), X_df.index)
                return pred
            else:
                return seasonal_naive(pd.concat([train_df, X_df], sort=False), X_df.index)
        return predict_fn
    return trainer


def plot_region_diagnostics(df_region, region_name, pipeline_path=None, pooled_objs=None, features=None, test_months=TEST_MONTHS, out_dir=DIAG_DIR):
    sub = df_region.sort_values("month").reset_index(drop=True)
    # Use tail to define holdout exactly
    test_sub = sub.tail(test_months)
    train_sub = sub.iloc[:-test_months] if len(sub) >= test_months else sub.iloc[:0]
    if len(test_sub) == 0:
        print(f"No holdout for {region_name}, skipping diagnostic plot.")
        return

    # Get model predictions same as validate_final_holdout_for_region
    model_pred = None
    if pipeline_path and os.path.exists(pipeline_path):
        try:
            pip = joblib.load(pipeline_path)
            if pip["type"] == "dense":
                X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                base_model_pred = np.maximum(pip["model"].predict(X_test_s), 0.0)
                model_pred = 0.8 * base_model_pred + 0.2 * seasonal_naive(sub, test_sub.index)
            elif pip["type"] == "sparse":
                X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                p_test = pip["model"]["clf"].predict_proba(X_test_s)[:, 1]
                y_test_reg = np.maximum(pip["model"]["reg"].predict(X_test_s), 0.0)
                h_test = (p_test >= pip["tau"]).astype(float)
                model_pred = pip["gamma"] * (h_test * y_test_reg) + (1 - pip["gamma"]) * seasonal_naive(sub, test_sub.index)
            else:
                model_pred = seasonal_naive(sub, test_sub.index)
        except Exception:
            model_pred = seasonal_naive(sub, test_sub.index)
    elif pooled_objs is not None and features is not None:
        try:
            model_pred = pooled_predict_region(pooled_objs[0], pooled_objs[1], pooled_objs[2] or {}, test_sub, features)
        except Exception:
            model_pred = seasonal_naive(sub, test_sub.index)
    else:
        model_pred = seasonal_naive(sub, test_sub.index)

    base_pred = seasonal_naive(sub, test_sub.index)
    y_true = test_sub[TARGET].values
    # per-year smape
    per_year = defaultdict(list)
    test_idx_list = list(test_sub.index)
    for i_local, idx in enumerate(test_idx_list):
        row = sub.loc[idx]
        per_year[row["month"].year].append((row[TARGET], float(model_pred[i_local]), float(base_pred[i_local])))
    years = sorted(per_year.keys())
    smapes = [smape_weighted(np.array([x[0] for x in per_year[y]]), np.array([x[1] for x in per_year[y]])) for y in years]
    baselines = [smape_weighted(np.array([x[0] for x in per_year[y]]), np.array([x[2] for x in per_year[y]])) for y in years]

    # plot SMAPE by year
    plt.figure(figsize=(8, 4))
    plt.plot(years, smapes, marker='o', label='model SMAPE')
    plt.plot(years, baselines, marker='o', linestyle='--', label='baseline SMAPE')
    plt.title(f"{region_name} — Holdout SMAPE by year")
    plt.xlabel("year")
    plt.ylabel("SMAPE (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fn1 = os.path.join(out_dir, f"{region_name.replace(' ','_')}_smape_by_year.png")
    plt.savefig(fn1)
    plt.close()

    # residuals histogram
    resid = y_true - model_pred
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=30)
    plt.title(f"{region_name} — Residuals (test)")
    plt.xlabel("residual (true - pred)")
    plt.ylabel("count")
    plt.tight_layout()
    fn2 = os.path.join(out_dir, f"{region_name.replace(' ','_')}_residuals_hist.png")
    plt.savefig(fn2)
    plt.close()

    print(f"Saved diagnostics plots for {region_name}: {fn1}, {fn2}")


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
        if r:
            improved.append(r)
    df_new = pd.DataFrame(improved)
    df_final = (df_new.set_index("region").combine_first(df_results.set_index("region")).reset_index())
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

    # quick detect for overlaps / wrong holdout sizes
    problems = detect_train_test_overlap(df, test_months=TEST_MONTHS)
    if problems:
        print("⚠️ Detected potential train/test issues (region, problem, details):")
        for p in problems:
            print("   ", p)
        print("Пожалуйста, проверьте длины рядов и HOLDOUT. Я продолжаю, но исправления holdout уже встроены в код (.tail).")

    regions = sorted(df["region"].dropna().unique().tolist())
    print(f"\n🌍 Found {len(regions)} regions. Training hybrid v3.4 (ensemble x{ENSEMBLE_N})...\n")
    print(f"ℹ️ Final holdout set per-region: last {FINAL_HOLDOUT_YEARS} years ({TEST_MONTHS} months)\n")

    # prepare feature columns for pooled/adaptation/validation later
    sample_region = df[df["region"] == regions[0]]
    feature_cols = [c for c in sample_region.columns if c not in ("region", "month", TARGET)]

    results = []
    for region in tqdm(regions, desc="⚙️ First pass"):
        r = train_one_region(df, region)
        if r:
            results.append(r)

    df_results = pd.DataFrame(results).sort_values("smape")
    out1 = os.path.join(OUTPUTS_DIR, f"metrics_v3p4_first_{TARGET}.csv")
    df_results.to_csv(out1, index=False)
    print(f"\n✅ First pass done. Results saved: {out1}")
    print(df_results.to_string(index=False))
    print(f"\nMean SMAPE (first): {df_results['smape'].mean():.2f}%")

    df_final = retrain_bad_regions(df, df_results, smape_threshold=50)

    # pooled adaptation: try for SMAPE>30 (uses fixed training mask, no holdout leakage)
    bad_regions = df_final[df_final["smape"] > 30]["region"].tolist()
    preproc = global_model = local_adapters = None
    if bad_regions:
        print(f"\n🔎 Found {len(bad_regions)} bad regions for pooled adaptation: {bad_regions}")
        try:
            preproc, global_model, local_adapters = train_pooled_global_then_local(df, bad_regions, features=feature_cols)
        except Exception as e:
            print(f"⚠️ Pooled adaptation failed: {e}")
            preproc, global_model, local_adapters = None, None, None

        improved_rows = []
        for region in bad_regions:
            sub = df[df["region"] == region].sort_values("month")
            test_sub = sub.tail(TEST_MONTHS)
            train_sub = sub.iloc[:-TEST_MONTHS]
            if len(train_sub) < 12 or len(test_sub) == 0:
                continue
            pred_pooled = pooled_predict_region(preproc, global_model, local_adapters, test_sub, feature_cols)

            # baseline: existing pipeline
            pipeline_path = os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl")
            baseline_smape = None
            if os.path.exists(pipeline_path):
                try:
                    pip = joblib.load(pipeline_path)
                    if pip["type"] == "dense":
                        X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                        base_pred = np.maximum(pip["model"].predict(X_test_s), 0.0)
                        base_pred = 0.8 * base_pred + 0.2 * seasonal_naive(sub, test_sub.index)
                    elif pip["type"] == "sparse":
                        X_test_s = pip["scaler"].transform(test_sub[pip["features"]])
                        p_test = pip["model"]["clf"].predict_proba(X_test_s)[:, 1]
                        y_test_reg = np.maximum(pip["model"]["reg"].predict(X_test_s), 0.0)
                        h_test = (p_test >= pip["tau"]).astype(float)
                        base_pred = pip["gamma"] * (h_test * y_test_reg) + (1 - pip["gamma"]) * seasonal_naive(sub, test_sub.index)
                    else:
                        base_pred = seasonal_naive(sub, test_sub.index)
                    baseline_smape = smape_weighted(test_sub[TARGET].values, base_pred)
                except Exception:
                    baseline_smape = None

            pooled_smape = smape_weighted(test_sub[TARGET].values, pred_pooled)
            print(f"Region {region}: baseline SMAPE={baseline_smape} pooled SMAPE={pooled_smape:.2f}")
            if baseline_smape is None or pooled_smape < baseline_smape:
                per_region_pipeline = {"type": "pooled_adapter", "preproc": preproc, "global_model": global_model,
                                       "local_adapter": local_adapters.get(region, None), "features": feature_cols,
                                       "training_end": train_sub["month"].max()}
                joblib.dump(per_region_pipeline, os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__pooled.pkl"))
                improved_rows.append({"region": region,
                                      "mae": mean_absolute_error(test_sub[TARGET].values, pred_pooled),
                                      "rmse": rmse(test_sub[TARGET].values, pred_pooled),
                                      "smape": pooled_smape})
                print(f"✅ Replaced/added pooled pipeline for {region} (SMAPE {pooled_smape:.2f})")
            else:
                print(f"✖️  Pooled did not improve for {region} (keep baseline).")

        if improved_rows:
            df_new = pd.DataFrame(improved_rows)
            df_final = (df_new.set_index("region").combine_first(df_final.set_index("region")).reset_index())

    out2 = os.path.join(OUTPUTS_DIR, f"metrics_v3p4_final_{TARGET}.csv")
    df_final.to_csv(out2, index=False)
    print(f"\n🏁 FINAL SUMMARY:")
    print(df_final.to_string(index=False))
    print(f"\nMean SMAPE (final): {df_final['smape'].mean():.2f}%")
    try:
        metrics_summary_path = save_metrics_summary(df, MODELS_DIR, feature_cols, OUTPUTS_DIR, test_months=TEST_MONTHS)
    except Exception as e:
        print("⚠️ Failed to save metrics summary:", e)
    # ----------------------------
    # Holdout validation: run both local and pooled validations (separate CSVs)
    # ----------------------------
    pooled_objs = None
    if preproc is not None and global_model is not None:
        pooled_objs = (preproc, global_model, local_adapters)

    # 1) local validation (per-region saved pipelines)
    try:
        df_val_local = validate_all_regions_holdout(df, models_dir=MODELS_DIR, years=FINAL_HOLDOUT_YEARS,
                                                    features=feature_cols, pooled_objs=None, mode="local",
                                                    out_csv=os.path.join(OUTPUTS_DIR, f"holdout_validation_local_{FINAL_HOLDOUT_YEARS}y.csv"))
        print("\nHoldout validation (local) summary:")
        print(df_val_local.sort_values("model_smape").to_string(index=False))
    except Exception as e:
        print(f"⚠️ Local holdout validation failed: {e}")

    # 2) pooled validation (if pooled trained)
    df_val_pooled = None
    if pooled_objs is not None:
        try:
            df_val_pooled = validate_all_regions_holdout(df, models_dir=MODELS_DIR, years=FINAL_HOLDOUT_YEARS,
                                                         features=feature_cols, pooled_objs=pooled_objs, mode="pooled",
                                                         out_csv=os.path.join(OUTPUTS_DIR, f"holdout_validation_pooled_{FINAL_HOLDOUT_YEARS}y.csv"))
            print("\nHoldout validation (pooled) summary:")
            print(df_val_pooled.sort_values("model_smape").to_string(index=False))
        except Exception as e:
            print(f"⚠️ Pooled holdout validation failed: {e}")

    # ----------------------------
    # Diagnostics: plots + rolling backtests for flagged regions
    # ----------------------------
    # choose flagged regions from local validation first (fallback to pooled flags if local not present)
    flagged = []
    if 'df_val_local' in locals() and df_val_local is not None:
        flagged = df_val_local[df_val_local["flag"].isin(["overfit", "warning"])]["region"].tolist()
    elif df_val_pooled is not None:
        flagged = df_val_pooled[df_val_pooled["flag"].isin(["overfit", "warning"])]["region"].tolist()

    if flagged:
        print(f"\n🩺 Running diagnostics for flagged regions: {flagged}")
    for region in flagged:
        region_df = df[df["region"] == region].sort_values("month")
        # prefer local pipeline for plots
        pipeline_path = os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl")
        pooled_to_use = pooled_objs if pooled_objs is not None else None

        try:
            plot_region_diagnostics(region_df, region, pipeline_path=pipeline_path if os.path.exists(pipeline_path) else None,
                                    pooled_objs=pooled_to_use, features=feature_cols, test_months=TEST_MONTHS, out_dir=DIAG_DIR)
        except Exception as e:
            print(f"⚠️ Plotting diagnostics failed for {region}: {e}")

        # rolling backtest using saved pipeline if available, else pooled
        rb = None
        try:
            if os.path.exists(pipeline_path):
                trainer = pipeline_trainer_from_saved(pipeline_path)
                rb = rolling_backtest_region(region_df, feature_cols, train_window_months=36, horizon=12, step_months=6, model_trainer_callable=trainer)
            elif pooled_objs is not None:
                # simple trainer that uses pooled objs (fit nothing, just use pooled preproc+global+local)
                def pooled_trainer(train_df, features):
                    def predict_fn(X_df):
                        return pooled_predict_region(pooled_objs[0], pooled_objs[1], pooled_objs[2] or {}, X_df, features)
                    return predict_fn
                rb = rolling_backtest_region(region_df, feature_cols, train_window_months=36, horizon=12, step_months=6, model_trainer_callable=pooled_trainer)
            else:
                rb = rolling_backtest_region(region_df, feature_cols, train_window_months=36, horizon=12, step_months=6, model_trainer_callable=None)
            if rb is not None and not rb.empty:
                rb_path = os.path.join(DIAG_DIR, f"rolling_backtest_{region.replace(' ','_')}.csv")
                rb.to_csv(rb_path, index=False)
                print(f"Saved rolling backtest for {region}: {rb_path}")
        except Exception as e:
            print(f"⚠️ Rolling backtest failed for {region}: {e}")
            # ----------------------------
        # Cleanup old saved pipelines: keep only latest 12
        # ----------------------------
        try:
            cleanup_models(MODELS_DIR, keep=12, archive=True)
        except Exception as e:
            print("⚠️ cleanup_models failed:", e)

if __name__ == "__main__":
    main()
