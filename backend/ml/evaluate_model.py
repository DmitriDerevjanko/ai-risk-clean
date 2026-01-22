from __future__ import annotations
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import train_lgbm_tuned
from train_lgbm_tuned import (
    EnsembleClassifier,
    EnsembleRegressor,
    create_features,
    seasonal_naive,
    smape_weighted,
)
from data_prep import load_gtd_csv, to_monthly_by_region

# ensure pickled ensemble classes are available under __main__ when loading
sys.modules["__main__"].EnsembleClassifier = train_lgbm_tuned.EnsembleClassifier
sys.modules["__main__"].EnsembleRegressor = train_lgbm_tuned.EnsembleRegressor

# ================================
# CONFIG
# ================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gtd.csv")
os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)

# How many years of recent data to use for validation (before the per-region holdout)
VALID_YEARS = 3

# Try to reuse TEST_MONTHS from training config so validation aligns with train/test split
TEST_MONTHS = getattr(train_lgbm_tuned, "TEST_MONTHS", getattr(train_lgbm_tuned, "FINAL_HOLDOUT_YEARS", 3) * 12)

# ================================
# HELPERS
# ================================
def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred, squared=False))


# safe scalar extractor for seasonal_naive (it returns np.ndarray)
def seasonal_naive_scalar(sub, months_or_index):
    arr = seasonal_naive(sub, months_or_index)
    if isinstance(arr, (list, tuple)):
        arr = np.asarray(arr)
    if isinstance(arr, np.ndarray):
        if arr.size >= 1:
            return float(arr.ravel()[0])
        return 0.0
    try:
        return float(arr)
    except Exception:
        return 0.0


# ================================
# EVALUATION + LONG FORECAST (реалистичный)
# ================================
def evaluate_region_extended(region, df):
    print(f"\n🔍 Evaluating region: {region}")

    path = os.path.join(MODELS_DIR, f"pipeline__{region.replace(' ', '_')}__.pkl")
    if not os.path.exists(path):
        print(f"⚠️ No model for {region}")
        return None, None

    obj = joblib.load(path, mmap_mode=None)
    profile = obj.get("profile", "dense")
    scaler = obj["scaler"]
    model = obj["model"]
    train_features = obj.get("features", [])
    tau = obj.get("tau", 0.5)
    gamma = obj.get("gamma", 0.8)
    calibrator = obj.get("calibrator", None)  # may be None

    sub = df[df["region"] == region].sort_values("month").reset_index(drop=True)
    if sub.empty:
        print(f"⚠️ No rows for region {region}")
        return None, None

    # ======== VALIDATION on recent data (before holdout) ========
    # val_end is the last month before the per-region test (holdout) window
    last_month = sub["month"].max()
    val_end = last_month - pd.DateOffset(months=TEST_MONTHS)
    val_start = val_end - pd.DateOffset(years=VALID_YEARS) + pd.DateOffset(days=1)
    # ensure inclusive range
    val_data = sub[(sub["month"] > val_start) & (sub["month"] <= val_end)].copy()

    if len(val_data) < 12:
        print(f"⚠️ Skipping {region}: not enough data for recent validation (need >=12 rows).")
        return None, None

    # prepare X_val according to saved features (fill missing with 0)
    X_val = val_data.drop(columns=["region", "month", "incidents_count"], errors="ignore")
    X_val = X_val.reindex(columns=train_features, fill_value=0.0)
    y_val = val_data["incidents_count"].values

    # scale (guard)
    try:
        Xs = scaler.transform(X_val)
    except Exception as e:
        print(f"⚠️ scaler.transform failed for {region}: {e}")
        return None, None

    # make predictions consistent with how models were trained
    if profile == "dense":
        # dense models were trained on log1p in pipeline; try inverse
        try:
            pred_raw = model.predict(Xs)
            y_pred_model = np.expm1(np.clip(pred_raw, -20, 50))
        except Exception:
            y_pred_model = np.maximum(model.predict(Xs), 0.0)

        # seasonal naive for the validation months (vector)
        y_naive_arr = seasonal_naive(sub, val_data["month"])
        # ensure numeric vector
        y_naive = np.asarray(y_naive_arr, dtype=float)
        # blend with seasonal naive — keep same blend used at training time (0.9 / 0.1)
        y_pred = 0.9 * y_pred_model + 0.1 * y_naive
        y_pred = np.maximum(y_pred, 0.0)

    else:  # sparse
        clf = model["clf"]
        reg = model["reg"]
        try:
            p_raw = clf.predict_proba(Xs)[:, 1]
        except Exception:
            p_raw = np.zeros(len(Xs))
        # apply saved calibrator if present
        if calibrator is not None:
            try:
                p = calibrator.transform(p_raw)
            except Exception:
                p = p_raw
        else:
            p = p_raw

        try:
            y_reg = np.maximum(reg.predict(Xs), 0.0)
        except Exception:
            y_reg = np.zeros(len(Xs))

        h = (p >= tau).astype(float)
        y_naive_arr = seasonal_naive(sub, val_data["month"])
        y_naive = np.asarray(y_naive_arr, dtype=float)
        y_pred = gamma * (h * y_reg) + (1 - gamma) * y_naive
        y_pred = np.maximum(y_pred, 0.0)

    # metrics (guard corr)
    mae_v = mean_absolute_error(y_val, y_pred)
    rmse_v = rmse(y_val, y_pred)
    smape_v = smape_weighted(y_val, y_pred)
    try:
        corr = np.corrcoef(y_val, y_pred)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    metrics_val = {
        "region": region,
        "mae": mae_v,
        "rmse": rmse_v,
        "smape": smape_v,
        "corr": corr,
    }

    # ======== FORECAST (from last data point forward) ========
    hist = sub.copy()
    last_date = hist["month"].max()
    preds, forecast_dates = [], []
    # next month start
    cur_date = (last_date + pd.offsets.MonthBegin(1)).normalize()
    horizon = 12 * 13  # 13 years for 2018-2030 like horizon (adjust as needed)

    noise_amp = float(np.std(hist["incidents_count"].tail(24))) * 0.2 if len(hist) >= 24 else 0.0

    for step in range(horizon):
        tail = hist.tail(36).copy()
        X = create_features(tail)
        if X.empty:
            break

        X_last = X.drop(columns=["region", "month", "incidents_count"], errors="ignore").iloc[-1:]
        X_last = X_last.reindex(columns=train_features, fill_value=0.0)

        try:
            X_scaled = scaler.transform(X_last)
        except Exception:
            break

        # compute scalar y_naive_pt safely
        y_naive_pt = seasonal_naive_scalar(hist, pd.Index([cur_date]))

        if profile == "dense":
            try:
                pred_log = model.predict(X_scaled)[0]
                y_hat_model = float(np.expm1(np.clip(pred_log, -20, 50)))
            except Exception:
                y_hat_model = float(np.maximum(model.predict(X_scaled)[0], 0.0))
            y_hat = 0.9 * y_hat_model + 0.1 * y_naive_pt

        else:
            try:
                p_raw = train_lgbm_tuned.EnsembleClassifier([0]).predict_proba  # dummy guard
            except Exception:
                pass
            try:
                p_raw = model["clf"].predict_proba(X_scaled)[:, 1]
            except Exception:
                p_raw = np.array([0.0])
            if calibrator is not None:
                try:
                    p_val_single = float(calibrator.transform(p_raw)[0])
                except Exception:
                    p_val_single = float(p_raw[0])
            else:
                p_val_single = float(p_raw[0])
            try:
                y_reg_single = float(np.maximum(model["reg"].predict(X_scaled)[0], 0.0))
            except Exception:
                y_reg_single = 0.0
            h = 1.0 if p_val_single >= tau else 0.0
            y_hat = gamma * (h * y_reg_single) + (1 - gamma) * y_naive_pt

        # small seasonal / noise / anchoring to last obs
        month_idx = (cur_date.month - 1) % 12
        seasonal_wave = 1 + 0.05 * np.sin(2 * np.pi * month_idx / 12)
        random_spike = 1.0 + np.random.normal(0.0, 0.05)
        y_hat = float(y_hat) * seasonal_wave * random_spike
        # anchor to last observed value to avoid explosion
        y_hat = 0.7 * y_hat + 0.3 * float(hist["incidents_count"].iloc[-1])
        # add small noise
        y_hat += np.random.normal(0, noise_amp)
        y_hat = max(float(y_hat), 0.0)

        # append scalars only
        preds.append(float(y_hat))
        forecast_dates.append(pd.Timestamp(cur_date))

        # add to history for auto-regressive feature creation
        hist = pd.concat([hist, pd.DataFrame({
            "region": [region], "month": [pd.Timestamp(cur_date)], "incidents_count": [y_hat]
        })], ignore_index=True)

        cur_date = cur_date + pd.offsets.MonthBegin(1)

    # === sanitize preds & dates before plotting ===
    # ensure preds is flat list of floats
    sanitized_preds = []
    bad_idx = []
    for i, v in enumerate(preds):
        try:
            sanitized_preds.append(float(np.array(v).ravel()[0]))
        except Exception:
            sanitized_preds.append(np.nan)
            bad_idx.append(i)
    if bad_idx:
        print(f"⚠️ Warning: coerced non-scalar preds at indices: {bad_idx}")

    # ensure forecast_dates are datetimes
    try:
        forecast_dates = pd.to_datetime(forecast_dates)
    except Exception:
        print("⚠️ Warning: forecast_dates conversion failed; attempting per-item parse.")
        fd2 = []
        for d in forecast_dates:
            try:
                fd2.append(pd.to_datetime(d))
            except Exception:
                fd2.append(pd.NaT)
        forecast_dates = pd.to_datetime(fd2)

    # PLOT (guard length match)
    plt.figure(figsize=(10, 4))
    plt.plot(val_data["month"], y_val, label="True (recent validation)", marker="o")
    plt.plot(val_data["month"], y_pred, label="Predicted (recent validation)", marker="x")
    if len(sanitized_preds) > 0 and len(sanitized_preds) == len(forecast_dates):
        plt.plot(forecast_dates, sanitized_preds, label="Forecast (future)", marker="^")
    elif len(sanitized_preds) > 0:
        print(f"⚠️ Not plotting forecast: length mismatch preds({len(sanitized_preds)}) vs dates({len(forecast_dates)})")
    plt.title(f"{region} — Validation (recent) + Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return metrics_val, {
        "region": region,
        "forecast_dates": [d.strftime("%Y-%m") if not pd.isna(d) else None for d in forecast_dates],
        "forecast_values": sanitized_preds
    }


# ================================
# MAIN
# ================================
def main():
    raw = load_gtd_csv(DATA_PATH)
    monthly = to_monthly_by_region(raw)
    df = create_features(monthly)
    regions = sorted(df["region"].dropna().unique().tolist())

    results_val = []
    forecasts_future = []

    for r in regions:
        try:
            mval, mfut = evaluate_region_extended(r, df)
        except Exception as e:
            print(f"⚠️ Unexpected error evaluating {r}: {e}")
            mval, mfut = None, None
        if mval:
            results_val.append(mval)
        if mfut:
            forecasts_future.append(mfut)

    df_val = pd.DataFrame(results_val)
    if not df_val.empty:
        print("\n📊 Validation (recent) Summary:")
        print(df_val.to_string(index=False))
        print(f"\nMean SMAPE: {df_val['smape'].mean():.2f}% | Mean Corr: {df_val['corr'].mean():.2f}")
        out_path = os.path.join(os.path.dirname(__file__), "outputs", "validation_recent_results.csv")
        df_val.to_csv(out_path, index=False)
        print(f"✅ Saved validation results → {out_path}")
    else:
        print("⚠️ No validation rows collected.")

    if forecasts_future:
        df_forecast = pd.DataFrame(forecasts_future)
        out_path2 = os.path.join(os.path.dirname(__file__), "outputs", "forecast_future_results.csv")
        df_forecast.to_csv(out_path2, index=False)
        print(f"✅ Saved long-term forecast → {out_path2}")
    else:
        print("⚠️ No forecasts generated.")

if __name__ == "__main__":
    main()
