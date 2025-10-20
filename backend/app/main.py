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
from typing import Optional, Dict, Any


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PATH = os.path.join(CURRENT_DIR, "..", "ml")
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

import train_lgbm_tuned
from train_lgbm_tuned import EnsembleRegressor, EnsembleClassifier, create_features
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

# -----------------------------
# VALIDATION METRICS (from evaluate_model.py)
# -----------------------------
VALIDATION_CSV = os.path.join(BASE_DIR, "..", "ml", "outputs", "validation_2010_2017_results.csv")
validation_df = None
if os.path.exists(VALIDATION_CSV):
    try:
        validation_df = pd.read_csv(VALIDATION_CSV)
        log.info(f"✅ Loaded validation metrics from {VALIDATION_CSV} | rows={len(validation_df)}")
    except Exception as e:
        log.warning(f"⚠️ Failed to load validation metrics: {e}")
else:
    log.warning(f"⚠️ Validation CSV not found at {VALIDATION_CSV}")

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
        safe_region = region.strip().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"pipeline__{safe_region}__.pkl")
        log.info(f"[{req_id}] 📂 Model path: {model_path}")
        if not os.path.exists(model_path):
            return {"error": f"Model not found for region '{region}'"}

        model_data = joblib.load(model_path)
        scaler = model_data.get("scaler")
        model = model_data["model"]
        profile = model_data.get("type", "dense")
        features = model_data.get("features", [])
        tau = model_data.get("tau", 0.5)
        gamma = model_data.get("gamma", 0.8)
        log.info(f"[{req_id}] ✅ Model loaded in {dt_ms(m_t0):.1f} ms")

        # ---------- DATA LOAD ----------
        if not os.path.exists(DATA_PATH):
            return {"error": f"Data file not found"}
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

        start = pd.Timestamp("1970-01-31")
        last_date = month_hist.index.max()
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
        noise_amp = float(np.std(series.tail(24))) * 0.2
        base_level = float(np.mean(series.tail(12))) if len(series) >= 12 else float(np.mean(series))

        for step in range(effective_h):
            tail = hist.tail(36).copy()
            X = create_features(tail)
            if X.empty:
                break
            X_last = X.drop(columns=["region", "month", "incidents_count"], errors="ignore").iloc[-1:]
            X_last = X_last.reindex(columns=features, fill_value=0.0)
            X_scaled = scaler.transform(X_last) if scaler is not None else X_last.values

            if profile == "dense":
                y_hat = float(np.maximum(model.predict(X_scaled)[0], 0.0))
            else:
                clf, reg = model["clf"], model["reg"]
                p = clf.predict_proba(X_scaled)[:, 1]
                y_reg = float(np.expm1(np.maximum(reg.predict(X_scaled)[0], 0.0)))
                hbin = 1.0 if p[0] >= float(tau) else 0.0
                y_hat = float(gamma) * (hbin * y_reg)

            month_idx = step % 12
            seasonal_wave = 1 + 0.1 * np.sin(2 * np.pi * month_idx / 12 + np.random.uniform(0, np.pi))
            random_spike = 1 + np.random.choice([0, 0.15, -0.1], p=[0.85, 0.1, 0.05])
            adaptive_trend = 1 + 0.002 * step

            y_hat = y_hat * seasonal_wave * random_spike * adaptive_trend
            y_hat = 0.7 * y_hat + 0.3 * float(series.iloc[-1])
            y_hat += np.random.normal(0, noise_amp)
            y_hat = max(y_hat, 0.0)

            preds.append(round(y_hat, 2))
            forecast_dates.append(cur_date.strftime("%Y-%m-%d"))

            new_row = pd.DataFrame({"region": [region], "month": [cur_date], "incidents_count": [y_hat]})
            hist = pd.concat([hist, new_row], ignore_index=True)
            series = hist["incidents_count"]
            last_date = cur_date
            cur_date = next_month(cur_date)

        # ---------- METRICS ----------
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

        # ---------- VALIDATION METRICS ----------
        val_metrics = {}
        if validation_df is not None:
            try:
                region_row = validation_df[validation_df["region"] == region]
                if not region_row.empty:
                    row = region_row.iloc[0]
                    val_metrics = {
                        "mae_validation": safe_num(row.get("mae")),
                        "rmse_validation": safe_num(row.get("rmse")),
                        "smape_validation": safe_num(row.get("smape")),
                        "corr_validation": safe_num(row.get("corr"))
                    }
                    log.info(f"[{req_id}] ✅ Validation metrics found for {region}")
            except Exception as e:
                log.warning(f"[{req_id}] ⚠️ Error while reading validation metrics: {e}")

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
                "profile": profile,
                "features_count": len(features),
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
