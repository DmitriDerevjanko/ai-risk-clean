from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import sys
import time
import uuid
import math
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple, List
import re
from scipy.stats import pearsonr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PATH = os.path.join(CURRENT_DIR, "..", "ml")
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

import train_lgbm_tuned
from train_lgbm_tuned import EnsembleRegressor, EnsembleClassifier, create_features
# ensure main namespace compatibility for deserialized ensembles
sys.modules["__main__"].EnsembleRegressor = train_lgbm_tuned.EnsembleRegressor
sys.modules["__main__"].EnsembleClassifier = train_lgbm_tuned.EnsembleClassifier

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("ai-risk-api")

def tnow() -> float:
    return time.perf_counter()

def dt_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

# -----------------------------
# FASTAPI
# -----------------------------
app = FastAPI(title="AI Global Risk Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "ml", "models")
DATA_PATH = os.path.join(BASE_DIR, "..", "ml", "data", "raw", "gtd.csv")

# New: allow external directory (e.g. /mnt/data) via env var EXTERNAL_DATA_DIR or default /mnt/data
EXTERNAL_DATA_DIR = os.getenv("EXTERNAL_DATA_DIR", "/mnt/data")
CONTAINER_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "ml", "outputs")

def choose_existing_path(basename: str) -> str:
    """
    Return first existing path among:
      1) EXTERNAL_DATA_DIR/<basename>
      2) CONTAINER_OUTPUTS_DIR/<basename>
      3) fallback to EXTERNAL_DATA_DIR/<basename> (so logs show attempted external path)
    """
    candidate1 = os.path.join(EXTERNAL_DATA_DIR, basename)
    candidate2 = os.path.join(CONTAINER_OUTPUTS_DIR, basename)
    if os.path.exists(candidate1):
        return candidate1
    if os.path.exists(candidate2):
        return candidate2
    return candidate1

# -----------------------------
# IMPORTANT: load ONLY metrics_summary_incidents_count.csv
# -----------------------------
METRICS_SUMMARY_CSV = choose_existing_path("metrics_summary_incidents_count.csv")
metrics_summary_df = None

def _try_load_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            log.info(f"✅ Loaded CSV: {path} | rows={len(df)}")
            return df
        except Exception as e:
            log.warning(f"⚠️ Failed to load CSV {path}: {e}")
            return None
    else:
        log.warning(f"ℹ️ CSV not found: {path}")
        return None

metrics_summary_df = _try_load_csv(METRICS_SUMMARY_CSV)

# -----------------------------
# HELPERS
# -----------------------------
def next_month(dt: pd.Timestamp) -> pd.Timestamp:
    return dt + pd.offsets.MonthEnd(1)

def safe_num(x: Any):
    """Convert to float or return None (never 'N/A')."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def log_head_tail(name: str, arr: np.ndarray, k: int = 5):
    """Log first and last elements of a numpy array for debugging."""
    try:
        l = len(arr)
        head = arr[:k].tolist() if l else []
        tail = arr[-k:].tolist() if l else []
        log.info(f"🔎 {name}: len={l}, head={head}, tail={tail}")
    except Exception as e:
        log.warning(f"⚠️ log_head_tail failed for {name}: {e}")

def _normalize_name(s: str) -> str:
    """Lowercase and keep only alnum for robust matching."""
    return re.sub(r'[^0-9a-z]', '', s.lower())

def list_pipeline_files(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.startswith("pipeline__") and f.endswith(".pkl")]

def find_model_for_region(models_dir: str, region: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find best pipeline file for provided region name.
    Returns tuple (full_path, reason) or (None, reason).
    """
    safe_region = region.strip().replace(" ", "_")
    candidate_exact = os.path.join(models_dir, f"pipeline__{safe_region}__.pkl")
    if os.path.exists(candidate_exact):
        return candidate_exact, "exact_match"

    pooled_candidate = os.path.join(models_dir, f"pipeline__{safe_region}__pooled.pkl")
    if os.path.exists(pooled_candidate):
        return pooled_candidate, "pooled_exact"

    files = list_pipeline_files(models_dir)
    norm_region = _normalize_name(region)
    for fn in files:
        m = re.match(r'pipeline__(.+?)__(.*)\.pkl$', fn)
        core = m.group(1) if m else fn.replace("pipeline__", "").replace(".pkl", "")
        if _normalize_name(core) == norm_region:
            return os.path.join(models_dir, fn), "normalized_match"

    for fn in files:
        m = re.match(r'pipeline__(.+?)__(.*)\.pkl$', fn)
        core = m.group(1) if m else fn.replace("pipeline__", "").replace(".pkl", "")
        ncore = _normalize_name(core)
        if norm_region in ncore or ncore in norm_region:
            return os.path.join(models_dir, fn), "partial_normalized"

    return None, "not_found"

# ---------- Backtest metrics helper (kept for reference but NOT used by /api/predict) ----------
def smape_numpy(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.abs(a) + np.abs(b) + 1e-8)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)

def compute_backtest_metrics(model_obj: Any, scaler: Any, profile: str, features: List[str],
                             hist_df: pd.DataFrame, region: str, tau: float = 0.5, gamma: float = 0.8,
                             n_back: int = 96) -> Dict[str, Optional[float]]:
    """
    NOTE: This function is intentionally left here for reference but is NOT called by the API.
    """
    try:
        dates = hist_df.index.to_list()
        if len(dates) < 12:
            return {}
        # ensure n_back not larger than available - reserve at least 12 months
        n_back = min(n_back, max(12, len(dates) - 12))
        y_true_list = []
        y_pred_list = []
        # iterate for last n_back months where we predict t using data up to t-1
        for i in range(len(dates) - n_back, len(dates)):
            # build tail up to i-1 (so we predict for dates[i])
            tail_index = dates[:i]
            tail_vals = hist_df.loc[tail_index].reset_index().rename(columns={"index": "month"})
            tail_vals["region"] = region
            # create features based on tail; X_all will have rows up to last known month (i-1)
            X_all = create_features(tail_vals)
            if X_all.empty:
                # fallback: use naive last observed
                y_pred = float(tail_vals["incidents_count"].iloc[-1]) if len(tail_vals) > 0 else 0.0
            else:
                X_last = X_all.drop(columns=["region", "month", "incidents_count"], errors="ignore").iloc[-1:]
                # reindex to saved features order
                if features:
                    X_last = X_last.reindex(columns=features, fill_value=0.0)
                # scale
                if scaler is not None:
                    try:
                        X_scaled = scaler.transform(X_last)
                        X_scaled_arr = X_scaled if isinstance(X_scaled, np.ndarray) else np.asarray(X_scaled)
                    except Exception:
                        X_scaled_arr = X_last.values
                else:
                    X_scaled_arr = X_last.values
                # predict using same logic as main loop
                try:
                    if str(profile).startswith("pooled"):
                        # best-effort: try using pooled_predict_region if available
                        try:
                            X_pooled = X_last.copy()
                            X_pooled["region"] = region
                            y_arr = train_lgbm_tuned.pooled_predict_region(model_obj.get("preproc", None),
                                                                           model_obj.get("global_model", None),
                                                                           model_obj.get("local_adapter", None) or {},
                                                                           X_pooled,
                                                                           list(X_last.columns))
                            y_pred = float(y_arr.ravel()[0])
                        except Exception:
                            # fallback: last observed
                            y_pred = float(tail_vals["incidents_count"].iloc[-1])
                    elif profile == "dense":
                        # prefer passing DataFrame to preserve feature names if model accepts it
                        try:
                            y_pred = float(np.maximum(model_obj.predict(X_last)[0], 0.0))
                        except Exception:
                            y_pred = float(np.maximum(model_obj.predict(X_scaled_arr)[0], 0.0))
                    elif profile == "sparse":
                        clf = model_obj.get("clf") if isinstance(model_obj, dict) else None
                        reg = model_obj.get("reg") if isinstance(model_obj, dict) else model_obj
                        if clf is not None:
                            p = clf.predict_proba(X_last)[:, 1] if hasattr(clf, "predict_proba") else clf.predict_proba(X_scaled_arr)[:, 1]
                            y_reg = float(np.expm1(np.maximum(reg.predict(X_last)[0], 0.0))) if hasattr(reg, "predict") else float(np.expm1(np.maximum(reg.predict(X_scaled_arr)[0], 0.0)))
                            hbin = 1.0 if p.ravel()[0] >= float(tau) else 0.0
                            y_pred = float(gamma) * (hbin * y_reg)
                        else:
                            # fallback single regressor
                            try:
                                y_pred = float(np.maximum(model_obj.predict(X_last)[0], 0.0))
                            except Exception:
                                y_pred = float(np.maximum(model_obj.predict(X_scaled_arr)[0], 0.0))
                    else:
                        y_pred = float(tail_vals["incidents_count"].iloc[-1])
                except Exception:
                    y_pred = float(tail_vals["incidents_count"].iloc[-1])

            y_true = float(hist_df.iloc[i]["incidents_count"])
            y_true_list.append(y_true)
            y_pred_list.append(max(float(y_pred), 0.0))

        a = np.array(y_true_list, dtype=float)
        b = np.array(y_pred_list, dtype=float)
        if len(a) == 0:
            return {}
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        smape = smape_numpy(a, b)
        corr = None
        try:
            if np.std(a) > 0 and np.std(b) > 0:
                corr = float(pearsonr(a, b)[0])
        except Exception:
            corr = None
        return {
            "mae_validation": safe_num(mae),
            "rmse_validation": safe_num(rmse),
            "smape_validation": safe_num(smape),
            "corr_validation": safe_num(corr) if corr is not None else None
        }
    except Exception as e:
        log.warning(f"⚠️ Backtest metrics calculation failed: {e}")
        return {}

# -----------------------------
# Helper to extract metrics from that single DF
# -----------------------------
def _extract_metrics_from_row(row: pd.Series) -> Dict[str, Optional[float]]:
    """
    Normalize different column names into standard keys:
    mae_validation, rmse_validation, smape_validation, corr_validation
    """
    if row is None or row.empty:
        return {}

    col = row

    # Common column names (we expect metrics_summary to contain: region,n_test_rows,mae,rmse,smape,corr)
    def first_val(keys):
        for k in keys:
            if k in col and pd.notna(col.get(k)):
                try:
                    return float(col.get(k))
                except Exception:
                    v = str(col.get(k))
                    try:
                        v2 = v.replace(",", "").replace("%", "").strip()
                        return float(v2)
                    except Exception:
                        pass
        lower_map = {cname.lower(): cname for cname in col.index.astype(str)}
        for k in keys:
            lk = k.lower()
            if lk in lower_map:
                raw = col.get(lower_map[lk])
                if pd.notna(raw):
                    try:
                        return float(raw)
                    except Exception:
                        try:
                            v2 = str(raw).replace(",", "").replace("%", "").strip()
                            return float(v2)
                        except Exception:
                            pass
        return None

    possible_mae = ["mae", "model_mae", "mae_validation", "mean_absolute_error"]
    possible_rmse = ["rmse", "model_rmse", "rmse_validation", "root_mean_squared_error", "rmse_test"]
    possible_smape = ["smape", "model_smape", "smape_validation", "smape(%)", "smape_%"]
    possible_corr = ["corr", "corr_validation", "correlation", "pearson", "pearsonr", "pearson_corr"]

    return {
        "mae_validation": safe_num(first_val(possible_mae)),
        "rmse_validation": safe_num(first_val(possible_rmse)),
        "smape_validation": safe_num(first_val(possible_smape)),
        "corr_validation": safe_num(first_val(possible_corr))
    }

# -----------------------------
# IMPORTANT: _find_precomputed_metrics now RETURNS only from metrics_summary_df
# -----------------------------
def _find_precomputed_metrics(region: str) -> Dict[str, Optional[float]]:
    """
    Return metrics only from metrics_summary_df (metrics_summary_incidents_count.csv).
    Matching is robust (exact or normalized).
    """
    if metrics_summary_df is None:
        log.info("ℹ️ metrics_summary_df not loaded; no precomputed metrics will be returned.")
        return {}

    df = metrics_summary_df

    # try exact match in 'region' column first
    if "region" in df.columns:
        exact = df[df["region"] == region]
        if not exact.empty:
            return _extract_metrics_from_row(exact.iloc[0])

    # try normalized string match across string columns (prefer 'region' but accept others)
    norm_target = _normalize_name(region)
    for c in df.columns:
        if df[c].dtype == object:
            try:
                norm_vals = df[c].astype(str).apply(_normalize_name)
                matches = norm_vals == norm_target
                if matches.any():
                    return _extract_metrics_from_row(df.loc[matches.idxmax()])
            except Exception:
                continue

    # no match found
    return {}

# ================================
# MAIN API ENDPOINT
# ================================
@app.get("/api/predict")
def predict(
    request: Request,
    region: str = Query(..., description="Region name as in GTD 'region_txt'"),
    horizon: Optional[int] = Query(None, description="Forecast horizon (months)"),
    debug: Optional[int] = Query(0, description="Include debug info if 1")
):
    req_id = str(uuid.uuid4())[:8]
    api_t0 = tnow()
    log.info(f"=== 🧠 NEW REQUEST [{req_id}] ===")
    log.info(f"➡️ Query: region='{region}', horizon_query={horizon}, debug={debug}")

    requested_h = horizon
    effective_h = 12 * 13 if requested_h is None else int(requested_h)
    log.info(f"[{req_id}] 📏 Horizon — requested={requested_h}, effective={effective_h}")

    try:
        # ---------- MODEL LOAD ----------
        m_t0 = tnow()
        model_path, reason = find_model_for_region(MODELS_DIR, region)
        if model_path is None:
            log.warning(f"[{req_id}] Model lookup for region '{region}' failed: {reason}")
            return {"error": f"Model not found for region '{region}'"}
        log.info(f"[{req_id}] 📂 Using model file: {model_path} (reason={reason})")

        model_data = joblib.load(model_path)
        scaler = model_data.get("scaler", None)
        model_obj = model_data.get("model", None)
        profile = model_data.get("type", "dense")
        features = model_data.get("features", []) or []
        tau = model_data.get("tau", 0.5)
        gamma = model_data.get("gamma", 0.8)

        # keep model_data around for pooled helper
        log.info(f"[{req_id}] ✅ Model metadata loaded in {dt_ms(m_t0):.1f} ms | type={profile} | features={len(features)}")

        # ---------- DATA LOAD ----------
        if not os.path.exists(DATA_PATH):
            return {"error": f"Data file not found at {DATA_PATH}"}
        df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
        log.info(f"[{req_id}] 📑 Data loaded | rows={len(df)}")

        # ---------- FILTER REGION ----------
        df_region = df[df["region_txt"] == region].copy()
        df_region = df_region[(df_region["imonth"] >= 1) & (df_region["imonth"] <= 12)]
        df_region["date"] = pd.to_datetime(
            df_region["iyear"].astype(str) + "-" + df_region["imonth"].astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )
        df_region = df_region.dropna(subset=["date"])
        if df_region.empty:
            return {"error": f"No data for region '{region}'"}
        log.info(f"[{req_id}] 🗓️ Date range: {df_region['date'].min()} .. {df_region['date'].max()}")

        # ---------- MONTHLY AGGREGATION ----------
        month_hist = (
            df_region.groupby(pd.Grouper(key="date", freq="ME"))
            .size()
            .rename("incidents_count")
            .to_frame()
        )
        if month_hist.empty:
            return {"error": "No monthly data available"}

        last_date = month_hist.index.max()
        start = month_hist.index.min()
        full_index = pd.period_range(start, last_date, freq="M").to_timestamp("M")
        month_hist = month_hist.reindex(full_index).fillna(0).astype(int)

        # ---------- SERIES PREP ----------
        hist = month_hist.copy().reset_index().rename(columns={"index": "month"})
        hist["region"] = region
        series = hist["incidents_count"].astype(float)
        log_head_tail(f"[{req_id}] series", series.values)

        # ---------- FORECAST LOOP ----------
        preds, forecast_dates = [], []
        cur_date = next_month(last_date)
        noise_amp = float(np.nanstd(series.tail(24))) * 0.2 if len(series) >= 2 else 0.1

        def predict_from_model(m, X_df_or_arr):
            """Return 1d numpy preds. Accepts DataFrame or array; keeps robustness."""
            if m is None:
                return np.zeros((X_df_or_arr.shape[0],))
            # If m expects DataFrame with columns (like sklearn pipelines), try passing DataFrame.
            try:
                if isinstance(X_df_or_arr, pd.DataFrame):
                    return np.asarray(m.predict(X_df_or_arr)).ravel()
            except Exception:
                pass
            # fallback array
            try:
                return np.asarray(m.predict(np.asarray(X_df_or_arr))).ravel()
            except Exception as e:
                raise

        for step in range(effective_h):
            tail = hist.tail(36).copy()
            X_all = create_features(tail)
            if X_all.empty:
                log.warning(f"[{req_id}] empty features at step {step}, stopping forecast")
                break

            X_last_df = X_all.drop(columns=["region", "month", "incidents_count"], errors="ignore").iloc[-1:].copy()
            if features:
                X_last_df = X_last_df.reindex(columns=features, fill_value=0.0)

            X_for_scale = X_last_df
            if scaler is not None:
                try:
                    X_scaled = scaler.transform(X_for_scale)
                    if isinstance(X_scaled, np.ndarray):
                        X_scaled_arr = X_scaled
                        X_scaled_df = pd.DataFrame(X_scaled_arr, columns=features, index=X_last_df.index) if features else pd.DataFrame(X_scaled_arr)
                    else:
                        X_scaled_df = X_scaled
                        X_scaled_arr = np.asarray(X_scaled_df)
                except Exception as e:
                    log.warning(f"[{req_id}] scaler.transform failed: {e}; using raw features")
                    X_scaled_df = X_last_df
                    X_scaled_arr = X_last_df.values
            else:
                X_scaled_df = X_last_df
                X_scaled_arr = X_last_df.values

            y_hat = 0.0
            try:
                if str(profile).startswith("pooled"):
                    # best-effort attempt to use a pooled prediction helper if provided in train_lgbm_tuned
                    try:
                        preproc = model_data.get("preproc", None)
                        global_model = model_data.get("global_model", None)
                        local_adapter = model_data.get("local_adapter", None)
                        X_pooled = X_last_df.copy()
                        X_pooled["region"] = region
                        y_hat_arr = train_lgbm_tuned.pooled_predict_region(preproc, global_model, local_adapter or {}, X_pooled, list(X_last_df.columns))
                        y_hat = float(np.maximum(y_hat_arr.ravel()[0], 0.0))
                    except Exception as e:
                        log.warning(f"[{req_id}] pooled_predict_region failed: {e}")
                        y_hat = float(series.iloc[-1])  # fallback naive
                elif profile == "dense":
                    try:
                        preds_model = predict_from_model(model_obj, X_scaled_df)
                    except Exception:
                        preds_model = predict_from_model(model_obj, X_scaled_arr)
                    y_hat = float(np.maximum(preds_model.ravel()[0], 0.0))
                elif profile == "sparse":
                    clf = None
                    reg = None
                    if isinstance(model_obj, dict):
                        clf = model_obj.get("clf")
                        reg = model_obj.get("reg")
                    else:
                        try:
                            preds_model = predict_from_model(model_obj, X_scaled_df)
                        except Exception:
                            preds_model = predict_from_model(model_obj, X_scaled_arr)
                        y_hat = float(np.maximum(preds_model.ravel()[0], 0.0))
                    if clf is not None and reg is not None:
                        if hasattr(clf, "predict_proba"):
                            p = clf.predict_proba(X_scaled_df)[:, 1] if isinstance(X_scaled_df, (pd.DataFrame, np.ndarray)) else clf.predict_proba(X_scaled_arr)[:, 1]
                        elif isinstance(clf, (list, tuple)):
                            probs = np.stack([c.predict_proba(X_scaled_df)[:, 1] for c in clf], axis=0)
                            p = probs.mean(axis=0)
                        else:
                            raise ValueError("Unsupported clf in sparse pipeline")

                        try:
                            y_reg_arr = predict_from_model(reg, X_scaled_df)
                        except Exception:
                            y_reg_arr = predict_from_model(reg, X_scaled_arr)
                        y_reg = float(np.maximum(y_reg_arr.ravel()[0], 0.0))
                        hbin = 1.0 if float(np.atleast_1d(p).ravel()[0]) >= float(tau) else 0.0
                        y_hat = float(gamma) * (hbin * y_reg) + (1.0 - float(gamma)) * float(train_lgbm_tuned.seasonal_naive(tail, [X_all.index[-1]])[0])
                else:
                    y_hat = float(train_lgbm_tuned.seasonal_naive(tail, [X_all.index[-1]])[0])
            except Exception as e:
                log.warning(f"[{req_id}] prediction failed at step {step}: {e}; using seasonal naive fallback")
                try:
                    y_hat = float(train_lgbm_tuned.seasonal_naive(tail, [X_all.index[-1]])[0])
                except Exception:
                    y_hat = 0.0

            month_idx = cur_date.month - 1
            seasonal_wave = 1.0 + 0.08 * math.sin(2 * math.pi * month_idx / 12)
            y_hat = y_hat * seasonal_wave
            last_obs = float(series.iloc[-1]) if len(series) > 0 else 0.0
            y_hat = 0.7 * y_hat + 0.3 * last_obs
            y_hat += float(np.random.normal(0, noise_amp))
            y_hat = max(y_hat, 0.0)

            preds.append(round(y_hat, 2))
            forecast_dates.append(cur_date.strftime("%Y-%m-%d"))

            new_row = pd.DataFrame({"region": [region], "month": [cur_date], "incidents_count": [y_hat]})
            hist = pd.concat([hist, new_row], ignore_index=True)
            series = hist["incidents_count"]
            last_date = cur_date
            cur_date = next_month(cur_date)

        # ---------- METRICS (on recent months of real history vs forecast head) ----------
        mae = rmse = smape = None
        try:
            hist_values = month_hist["incidents_count"].astype(float).fillna(0).values
            preds_arr = np.array(preds, dtype=float)
            n = min(12, len(hist_values), len(preds_arr))
            if n > 0:
                y_true = hist_values[-n:]
                y_pred = preds_arr[:n]
                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                smape = float(np.mean(2.0 * np.abs(y_pred - y_true) /
                                      (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
        except Exception as e:
            log.warning(f"[{req_id}] ⚠️ Metrics unavailable: {e}")

        # ---------- VALIDATION METRICS ---------- (take ONLY precomputed values from metrics_summary_df)
        val_metrics = _find_precomputed_metrics(region)
        if val_metrics:
            log.info(f"[{req_id}] ✅ Returning precomputed validation metrics for {region} (from metrics_summary_incidents_count.csv)")
        else:
            log.info(f"[{req_id}] ℹ️ No precomputed metrics found in metrics_summary_incidents_count.csv for {region}")

        # ---------- RESPONSE ----------
        payload: Dict[str, Any] = {
            "region": region,
            "forecast_dates": forecast_dates,
            "forecast_values": preds,
            "forecast": preds,
            "mean_forecast": safe_num(np.mean(preds) if preds else None),
            "last_observed": safe_num(series.iloc[-effective_h - 1] if len(series) > effective_h else series.iloc[-1]),
            "metrics": {"mae": mae, "rmse": rmse, "smape": smape},
            "validation_metrics": val_metrics,
            "history": [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in zip(month_hist.index, month_hist["incidents_count"].values)]
        }

        if debug == 1:
            payload["debug"] = {
                "request_id": req_id,
                "model_path": model_path,
                "model_lookup_reason": reason,
                "profile": profile,
                "features_count": len(features),
                "features": features[:40],
                "tau": tau,
                "gamma": gamma,
                "history_len": len(month_hist),
                "preds_len": len(preds),
                "effective_horizon": effective_h,
            }

        return payload

    except Exception as e:
        log.exception(f"[{req_id}] 💥 ERROR during prediction")
        return {"error": str(e)}
