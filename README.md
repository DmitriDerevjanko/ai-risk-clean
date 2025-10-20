# 🌍 AI Global Risk Forecast API & Dashboard

**AI Global Risk** is a machine learning system that forecasts monthly incident trends across global regions using data from the **Global Terrorism Database (GTD)**.  
It includes a LightGBM training pipeline, a FastAPI backend, and an interactive HTML/JS dashboard with a world map.

---

## 🚀 Overview

The project predicts regional terrorism activity using historical GTD data (1970–2017) and simulates possible future incident counts up to 2030.  
Each model is region-specific and combines statistical features with adaptive ensemble forecasting.

---

## 🧩 Project Structure

```
ai-risk/
│
├── backend/
│   ├── api/                    # FastAPI endpoints
│   │   └── main.py             # /api/predict — main API endpoint
│   ├── ml/
│   │   ├── train_lgbm_tuned.py # LightGBM ensemble training
│   │   ├── data_prep.py        # Data preprocessing utilities
│   │   └── outputs/            # Metrics and evaluation results
│   └── models/                 # Saved models per region
│
├── frontend/
│   ├── index.html              # Interactive web dashboard (Leaflet + Chart.js)
│   ├── app.js                  # Chart and map logic
│   └── styles.css              # UI styling
│
└── data/
    └── raw/gtd.csv             # Original GTD dataset
```

---

## ⚙️ Key Features

- Regional **LightGBM ensembles** (dense/sparse models)
- Feature engineering: lags, rolling means, trends, volatility
- Adaptive retraining for weak regions (SMAPE > 50%)
- Metrics: **MAE**, **RMSE**, **SMAPE**, **Correlation**
- REST API: `/api/predict?region=Europe&horizon=156`
- Web dashboard: interactive forecast chart + world risk map

---

## 📡 API Example

**Request**
```bash
GET /api/predict?region=North America&horizon=156
```

**Response**
```json
{
  "region": "North America",
  "forecast_dates": ["2018-01-31", "2018-02-28", "..."],
  "forecast_values": [232, 214, 240],
  "metrics": { "mae": 14.2, "rmse": 19.6, "smape": 7.8 },
  "validation_metrics": {
    "mae_validation": 13.7,
    "rmse_validation": 18.9,
    "smape_validation": 8.3,
    "corr_validation": 0.86
  }
}
```

---

## 🧠 Model Workflow

1. Load GTD dataset (`data/raw/gtd.csv`)
2. Aggregate incidents by region and month
3. Generate time-series features (`create_features`)
4. Train LightGBM ensembles per region
5. Evaluate metrics and retrain weak models
6. Save models under `backend/ml/models/`

---

## 🧰 Installation

```bash
# Clone repository
git clone https://github.com/DmitriDerevjanko/ai-risk.git
cd ai-risk/backend

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload

# Open dashboard
open ../frontend/index.html
```

### requirements.txt
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pandas>=2.2
numpy>=1.26
joblib>=1.3
lightgbm>=4.3
scikit-learn==1.5.2
tqdm
```

---

## 🧪 Model Training

```bash
cd backend/ml
python train_lgbm_tuned.py
```

Trained models are saved to:
```
backend/ml/models/pipeline__REGION__.pkl
```

Metrics are stored in:
```
backend/ml/outputs/metrics_v3p4_final_incidents_count.csv
```

---

## 🧭 Example Metrics

| Region | MAE | RMSE | SMAPE | Corr |
|:-------|----:|----:|------:|-----:|
| Western Europe | 11.4 | 15.7 | 7.1 | 0.87 |
| South Asia | 15.2 | 21.9 | 8.3 | 0.81 |
| North America | 13.5 | 18.8 | 8.0 | 0.85 |

---

## 🔍 Tech Stack

| Layer | Technology |
|:------|:------------|
| Backend | FastAPI, Python 3.12 |
| ML Models | LightGBM, NumPy, pandas, scikit-learn |
| Frontend | HTML, Chart.js, Leaflet.js |
| Data | Global Terrorism Database (GTD) |
| Logging | Python logging, tqdm |
| Deployment | Uvicorn / Docker-ready |

---

## 👨‍💻 Author

**Dmitri Derevjanko**  
🎓 Machine Learning & Computer Vision  
🌐 [dmitriderevjanko.com](https://dmitriderevjanko.com)  
🐙 [GitHub](https://github.com/DmitriDerevjanko)


---

## 🪄 License

MIT License © 2025 — AI Risk Intelligence
