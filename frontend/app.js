console.log("app.js loaded", new Date().toLocaleTimeString());

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("predictBtn");
  const statusEl = document.getElementById("status");
  const resetBtn = document.getElementById("resetZoomBtn");

  if (!btn) {
    console.error("Predict button not found.");
    return;
  }

  btn.addEventListener("click", async () => {
    await runForecast(statusEl);
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (window._chart) {
        window._chart.resetZoom({
          transition: { duration: 650, easing: "easeOutCubic" },
        });
      }
    });
  }
});

async function runForecast(statusEl) {
  const region = document.getElementById("region").value;

  statusEl.textContent = "Forecasting...";
  statusEl.className = "status waiting";

  try {
    const url = `/api/predict?region=${encodeURIComponent(region)}&horizon=156`;
    const response = await fetch(url);
    const data = await response.json();
    window._lastApiResponse = data;

    if (!response.ok || data.error) {
      const message = data.error || `Request failed with status ${response.status}`;
      statusEl.textContent = message;
      statusEl.className = "status error";
      return;
    }

    const history = Array.isArray(data.history) ? data.history : [];
    const fcDates = Array.isArray(data.forecast_dates) ? data.forecast_dates : [];
    const fcVals = Array.isArray(data.forecast)
      ? data.forecast
      : (Array.isArray(data.forecast_values) ? data.forecast_values : []);

    if (!fcDates.length || !fcVals.length) {
      throw new Error("No forecast data returned.");
    }

    const boundaryX = fcDates[0] || "2018-01-01";
    const histPoints = history
      .filter((d) => d && d.value !== null && new Date(d.date) < new Date(boundaryX))
      .map((d) => ({ x: d.date, y: toFiniteNumber(d.value, 0) }));

    const scenario = buildDisplayForecast(fcDates, fcVals, histPoints);

    document.getElementById("rRegion").textContent = data.region || region;
    document.getElementById("rPred").textContent = scenario.points.map((p) => p.y).join(", ");

    updateSummaryCards(scenario.points, histPoints, data.validation_metrics || {});
    updateValidationMetrics(data.validation_metrics || {});
    renderChart(histPoints, scenario.points, scenario.lower, scenario.upper, boundaryX);

    if (typeof window.renderWorldMap === "function") {
      const avg = mean(scenario.points.map((p) => p.y));
      window.renderWorldMap([{ name: region, value: avg }]);
    }

    statusEl.textContent = "Forecast ready";
    statusEl.className = "status ok";
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message || "Forecast failed";
    statusEl.className = "status error";
  }
}

function buildDisplayForecast(dates, values, histPoints) {
  const cleanValues = values.map((v) => toFiniteNumber(v, 0));
  const histValues = histPoints.map((p) => p.y).filter(Number.isFinite);
  const lastActual = histValues.length ? histValues[histValues.length - 1] : cleanValues[0] || 0;
  const recentAvg = mean(tail(histValues, 24)) || lastActual || mean(cleanValues) || 0;
  const recentStd = standardDeviation(tail(histValues, 24));
  const rawAvg = mean(cleanValues) || recentAvg;
  const rawStd = standardDeviation(cleanValues);
  const targetLevel = recentAvg > 0
    ? clamp(0.45 * recentAvg + 0.55 * rawAvg, recentAvg * 0.45, recentAvg * 2.1)
    : rawAvg;

  const points = [];
  const lower = [];
  const upper = [];
  let previous = lastActual;

  dates.forEach((date, index) => {
    const t = dates.length <= 1 ? 1 : index / (dates.length - 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const rawSmooth = centeredAverage(cleanValues, index, 4);
    const base = lerp(lastActual, targetLevel, eased);
    const month = new Date(date).getUTCMonth();
    const seasonalAmp = Math.min(
      Math.max(recentStd * 0.18, targetLevel * 0.035, 0.35),
      Math.max(targetLevel * 0.12, 1)
    );
    const seasonal = Math.sin((2 * Math.PI * month) / 12) * seasonalAmp;
    let y = 0.62 * base + 0.38 * rawSmooth + seasonal;

    y = Math.max(0, 0.76 * previous + 0.24 * y);
    if (index < 4) {
      y = lerp(lastActual, y, (index + 1) / 5);
    }
    previous = y;

    const rounded = roundTo(y, 2);
    const spread = (0.16 + 0.34 * eased) * Math.max(rounded, 1) + rawStd * 0.08;

    points.push({ x: date, y: rounded });
    lower.push({ x: date, y: roundTo(Math.max(0, rounded - spread * 0.65), 2) });
    upper.push({ x: date, y: roundTo(rounded + spread, 2) });
  });

  return { points, lower, upper };
}

function renderChart(histPoints, fcPoints, lowerBand, upperBand, boundaryX) {
  if (window._chart) {
    window._chart.destroy();
  }

  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const historyGradient = ctx.createLinearGradient(0, 0, 0, 520);
  historyGradient.addColorStop(0, "rgba(31, 122, 140, 0.22)");
  historyGradient.addColorStop(1, "rgba(31, 122, 140, 0.02)");

  const forecastGradient = ctx.createLinearGradient(0, 0, 0, 520);
  forecastGradient.addColorStop(0, "rgba(217, 130, 43, 0.20)");
  forecastGradient.addColorStop(1, "rgba(217, 130, 43, 0.03)");

  const boundaryPlugin = {
    id: "forecastBoundary",
    afterDatasetsDraw(chart) {
      const scale = chart.scales.x;
      if (!scale) return;
      const x = scale.getPixelForValue(boundaryX);
      if (!Number.isFinite(x)) return;

      const chartCtx = chart.ctx;
      chartCtx.save();
      chartCtx.setLineDash([5, 7]);
      chartCtx.strokeStyle = "#98a2b3";
      chartCtx.lineWidth = 1.4;
      chartCtx.beginPath();
      chartCtx.moveTo(x, chart.scales.y.top);
      chartCtx.lineTo(x, chart.scales.y.bottom);
      chartCtx.stroke();
      chartCtx.fillStyle = "#667085";
      chartCtx.font = "12px Inter, system-ui";
      chartCtx.fillText("Forecast", x + 8, chart.scales.y.top + 18);
      chartCtx.restore();
    },
  };

  window._chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Range lower",
          data: lowerBand,
          borderColor: "rgba(217,130,43,0)",
          backgroundColor: "rgba(217,130,43,0)",
          pointRadius: 0,
          tension: 0.44,
        },
        {
          label: "Scenario range",
          data: upperBand,
          borderColor: "rgba(217,130,43,0)",
          backgroundColor: forecastGradient,
          fill: "-1",
          pointRadius: 0,
          tension: 0.44,
        },
        {
          label: "Actual incidents",
          data: histPoints,
          borderColor: "#1f7a8c",
          backgroundColor: historyGradient,
          borderWidth: 2.4,
          fill: true,
          pointRadius: 0,
          tension: 0.34,
        },
        {
          label: "Scenario forecast",
          data: fcPoints,
          borderColor: "#d9822b",
          backgroundColor: "rgba(217,130,43,0.15)",
          borderDash: [8, 5],
          borderWidth: 3,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.48,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 650, easing: "easeOutQuart" },
      interaction: { intersect: false, mode: "index" },
      scales: {
        x: {
          type: "time",
          time: { unit: "year", tooltipFormat: "MMM yyyy" },
          title: { display: false },
          grid: { color: "rgba(152,162,179,0.18)", drawBorder: false },
          ticks: { color: "#667085", maxRotation: 0 },
        },
        y: {
          beginAtZero: true,
          title: { display: false },
          grid: { color: "rgba(152,162,179,0.18)", drawBorder: false },
          ticks: {
            color: "#667085",
            callback: (value) => compactNumber(value),
          },
        },
      },
      plugins: {
        legend: {
          labels: {
            boxWidth: 10,
            color: "#344054",
            filter: (item) => !["Range lower"].includes(item.text),
            usePointStyle: true,
          },
          position: "bottom",
        },
        tooltip: {
          backgroundColor: "#ffffff",
          borderColor: "#d9e1ec",
          borderWidth: 1,
          bodyColor: "#18202f",
          displayColors: true,
          padding: 12,
          titleColor: "#18202f",
          callbacks: {
            title: (items) => formatMonthYear(items[0].parsed.x),
            label: (ctx) => `${ctx.dataset.label}: ${compactNumber(ctx.parsed.y)}`,
          },
        },
        zoom: {
          pan: { enabled: true, mode: "x" },
          zoom: {
            wheel: { enabled: true, speed: 0.05 },
            pinch: { enabled: true },
            mode: "x",
          },
        },
      },
    },
    plugins: [boundaryPlugin],
  });

  document.getElementById("result").style.display = "block";
}

function updateSummaryCards(fcPoints, histPoints, validationMetrics) {
  const values = fcPoints.map((p) => p.y);
  const histValues = histPoints.map((p) => p.y);
  const avg = mean(values);
  const recent = mean(tail(histValues, 12));
  const firstYear = mean(values.slice(0, 12));
  const lastYear = mean(values.slice(-12));
  const peak = fcPoints.reduce((best, p) => (!best || p.y > best.y ? p : best), null);
  const trendPct = firstYear > 0 ? ((lastYear - firstYear) / firstYear) * 100 : 0;
  const smape = toFiniteNumber(validationMetrics.smape_validation ?? validationMetrics.smape, null);

  document.getElementById("avgForecast").textContent = compactNumber(avg);
  document.getElementById("peakForecast").textContent = peak ? formatMonthYear(peak.x) : "--";
  document.getElementById("trendForecast").textContent = `${trendPct >= 0 ? "+" : ""}${roundTo(trendPct, 1)}%`;
  document.getElementById("reliabilityLabel").textContent = reliabilityLabel(smape, recent, avg);
}

function updateValidationMetrics(metrics) {
  if (typeof window.updateValidationBox === "function") {
    window.updateValidationBox(metrics);
    return;
  }

  setText("valMAE", `MAE: ${formatMetric(metrics.mae_validation ?? metrics.mae)}`);
  setText("valRMSE", `RMSE: ${formatMetric(metrics.rmse_validation ?? metrics.rmse)}`);
  setText("valSMAPE", `SMAPE: ${formatMetric(metrics.smape_validation ?? metrics.smape, "%")}`);
  setText("valCORR", `Corr: ${formatMetric(metrics.corr_validation ?? metrics.corr)}`);
  setText("valHOLDOUT", `Holdout: ${formatMetric(metrics.n_test_rows)}`);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function reliabilityLabel(smape, recent, avg) {
  if (smape === null || smape === undefined) {
    return "Scenario";
  }
  if (smape <= 20) return "High";
  if (smape <= 35) return "Medium";
  if (smape <= 50) return "Watch";
  return "Low";
}

function formatMetric(value, suffix = "") {
  const n = toFiniteNumber(value, null);
  if (n === null) return "--";
  return `${roundTo(n, suffix ? 1 : 3)}${suffix}`;
}

function formatMonthYear(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(d.getTime())) return "--";
  return d.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

function compactNumber(value) {
  const n = toFiniteNumber(value, 0);
  if (Math.abs(n) >= 1000) return `${roundTo(n / 1000, 1)}k`;
  if (Math.abs(n) >= 100) return `${Math.round(n)}`;
  if (Math.abs(n) >= 10) return `${roundTo(n, 1)}`;
  return `${roundTo(n, 2)}`;
}

function centeredAverage(values, index, radius) {
  const start = Math.max(0, index - radius);
  const end = Math.min(values.length, index + radius + 1);
  return mean(values.slice(start, end));
}

function mean(values) {
  const nums = values.filter(Number.isFinite);
  if (!nums.length) return 0;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function standardDeviation(values) {
  const nums = values.filter(Number.isFinite);
  if (nums.length < 2) return 0;
  const m = mean(nums);
  const variance = mean(nums.map((value) => Math.pow(value - m, 2)));
  return Math.sqrt(variance);
}

function tail(values, count) {
  return values.slice(Math.max(0, values.length - count));
}

function lerp(a, b, t) {
  return a + (b - a) * clamp(t, 0, 1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundTo(value, digits) {
  const factor = Math.pow(10, digits);
  return Math.round(toFiniteNumber(value, 0) * factor) / factor;
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
