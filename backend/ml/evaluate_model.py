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

# Регистрируем классы для совместимости с joblib
sys.modules["__main__"].EnsembleClassifier = train_lgbm_tuned.EnsembleClassifier
sys.modules["__main__"].EnsembleRegressor = train_lgbm_tuned.EnsembleRegressor


# ================================
# CONFIG
# ================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gtd.csv")


# ================================
# HELPERS
# ================================
def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred, squared=False))


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
    tau, gamma = obj.get("tau", 0.5), obj.get("gamma", 0.8)

    sub = df[df["region"] == region].sort_values("month")

    # ======== VALIDATION (2010–2017) ========
    val_start, val_end = pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31")
    val_data = sub[(sub["month"] >= val_start) & (sub["month"] <= val_end)]

    if len(val_data) < 12:
        print(f"⚠️ Skipping {region}: not enough data for 2010–2017 validation.")
        return None, None

    X_val = val_data.drop(columns=["region", "month", "incidents_count"], errors="ignore")
    X_val = X_val.reindex(columns=train_features, fill_value=0.0)
    y_val = val_data["incidents_count"].values
    Xs = scaler.transform(X_val)

    if profile == "dense":
        y_pred = np.maximum(model.predict(Xs), 0.0)
        y_pred = 0.8 * y_pred + 0.2 * seasonal_naive(sub, X_val.index)
    else:
        clf, reg = model["clf"], model["reg"]
        p = clf.predict_proba(Xs)[:, 1]
        y_reg = np.expm1(np.maximum(reg.predict(Xs), 0.0))
        h = (p >= tau).astype(float)
        y_pred = gamma * (h * y_reg) + (1 - gamma) * seasonal_naive(sub, X_val.index)

    metrics_val = {
        "region": region,
        "mae": mean_absolute_error(y_val, y_pred),
        "rmse": rmse(y_val, y_pred),
        "smape": smape_weighted(y_val, y_pred),
        "corr": np.corrcoef(y_val, y_pred)[0, 1],
    }

    # ======== FORECAST (2018–2030) ========
    hist = sub.copy()
    last_date = hist["month"].max()
    preds, forecast_dates = [], []
    cur_date = last_date + pd.offsets.MonthEnd(1)
    horizon = 12 * 13 

    noise_amp = np.std(hist["incidents_count"].tail(24)) * 0.2
    base_level = np.mean(hist["incidents_count"].tail(12))

    for step in range(horizon):
        tail = hist.tail(36).copy()
        X = create_features(tail)
        if X.empty:
            break

        X_last = X.drop(columns=["region", "month", "incidents_count"], errors="ignore").iloc[-1:]
        X_last = X_last.reindex(columns=train_features, fill_value=0.0)
        X_scaled = scaler.transform(X_last)

        if profile == "dense":
            y_hat = float(np.maximum(model.predict(X_scaled)[0], 0.0))
        else:
            clf, reg = model["clf"], model["reg"]
            p = clf.predict_proba(X_scaled)[:, 1]
            y_reg = float(np.expm1(np.maximum(reg.predict(X_scaled)[0], 0.0)))
            h = 1.0 if p[0] >= tau else 0.0
            y_hat = gamma * (h * y_reg)

        month_idx = step % 12
        seasonal_wave = 1 + 0.1 * np.sin(2 * np.pi * month_idx / 12 + np.random.uniform(0, np.pi))
        random_spike = 1 + np.random.choice([0, 0.15, -0.1], p=[0.85, 0.1, 0.05])
        adaptive_trend = 1 + 0.002 * step  

        y_hat = y_hat * seasonal_wave * random_spike * adaptive_trend

        y_hat = 0.7 * y_hat + 0.3 * hist["incidents_count"].iloc[-1]

        y_hat += np.random.normal(0, noise_amp)

        y_hat = max(y_hat, 0.0)
        preds.append(y_hat)
        forecast_dates.append(cur_date)

        hist = pd.concat([hist, pd.DataFrame({
            "region": [region],
            "month": [cur_date],
            "incidents_count": [y_hat],
        })], ignore_index=True)
        cur_date += pd.offsets.MonthEnd(1)

    # ======== PLOT ========
    plt.figure(figsize=(10, 4))
    plt.plot(val_data["month"], y_val, label="True (2010–2017)", marker="o", color="#1f9d8f")
    plt.plot(val_data["month"], y_pred, label="Predicted (2010–2017)", marker="x", color="#f59e0b")
    if preds:
        plt.plot(forecast_dates, preds, label="Forecast 2018–2030", marker="^", color="#e11d48")
    plt.title(f"{region} — Validation 2010–2017 + Forecast 2018–2030 (realistic)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return metrics_val, {
        "region": region,
        "forecast_dates": [d.strftime("%Y-%m") for d in forecast_dates],
        "forecast_values": preds
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
        mval, mfut = evaluate_region_extended(r, df)
        if mval:
            results_val.append(mval)
        if mfut:
            forecasts_future.append(mfut)

    df_val = pd.DataFrame(results_val)
    print("\n📊 Validation 2010–2017 Summary:")
    print(df_val.to_string(index=False))
    print(f"\nMean SMAPE: {df_val['smape'].mean():.2f}% | Mean Corr: {df_val['corr'].mean():.2f}")
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "validation_2010_2017_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_val.to_csv(out_path, index=False)
    print(f"✅ Saved validation results → {out_path}")

    df_forecast = pd.DataFrame(forecasts_future)
    out_path2 = os.path.join(os.path.dirname(__file__), "outputs", "forecast_2018_2030_results.csv")
    df_forecast.to_csv(out_path2, index=False)
    print(f"✅ Saved long-term forecast → {out_path2}")


if __name__ == "__main__":
    main()
