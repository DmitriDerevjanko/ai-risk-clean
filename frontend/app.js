console.log("✅ app.js loaded at", new Date().toLocaleTimeString());

document.addEventListener("DOMContentLoaded", () => {
  console.group("🚀 DOM Ready Sequence");
  console.time("⏱️ DOM Initialization Time");

  const btn = document.getElementById("predictBtn");
  const statusEl = document.getElementById("status");

  if (!btn) {
    console.error("❌ Button not found in DOM!");
    console.groupEnd();
    return;
  }
  console.log("✅ Button found, adding event listeners");

  btn.addEventListener("click", async () => {
    console.group("🟢 Predict button clicked");
    console.time("⏱️ Forecast Duration");
    await runForecast(statusEl);
    console.timeEnd("⏱️ Forecast Duration");
    console.groupEnd();
  });

  const resetBtn = document.getElementById("resetZoomBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      console.log("🔄 Reset zoom clicked");
      if (window._chart) {
        console.time("⏱️ Zoom reset time");
        window._chart.resetZoom({
          transition: { duration: 800, easing: "easeOutCubic" },
        });
        console.timeEnd("⏱️ Zoom reset time");
      } else {
        console.warn("⚠️ No chart found to reset.");
      }
    });
  }

  console.timeEnd("⏱️ DOM Initialization Time");
  console.groupEnd();
});

async function runForecast(statusEl) {
  const region = document.getElementById("region").value;
  console.group(`▶️ runForecast() | region="${region}"`);
  console.time("⏱️ API Fetch Time");

  statusEl.textContent = "⏳ Forecasting...";
  statusEl.className = "status waiting";

  try {
    // 🔁 13 years × 12 months = 156 months (forecast until 2030)
    const url = `/api/predict?region=${encodeURIComponent(region)}&horizon=156`;
    console.log("🌐 Fetching:", url);

    const response = await fetch(url);
    console.log("📩 Response status:", response.status);
    const data = await response.json();
    console.timeEnd("⏱️ API Fetch Time");

    if (data.error) {
      console.error("💥 API Error:", data.error);
      statusEl.textContent = "❌ Error: " + data.error;
      statusEl.className = "status error";
      console.groupEnd();
      return;
    }

    const hist = data.history || [];
    const fcDates = data.forecast_dates || [];
    const fcVals = data.forecast || [];

    console.log(`📊 History length: ${hist.length}, Forecast length: ${fcDates.length}`);
    if (!hist.length || !fcVals.length)
      console.warn("⚠️ One of datasets is empty!");

    const histPoints = hist
      .filter((d) => d.value !== null && new Date(d.date) <= new Date("2017-12-31"))
      .map((d) => ({ x: d.date, y: d.value }));

    const fcPoints = fcDates.map((d, i) => ({
      x: d,
      y: fcVals[i] ?? 0,
    }));

    statusEl.textContent = "✅ Forecast ready";
    statusEl.className = "status ok";

    const val = data.validation_metrics || {};

    document.getElementById("rRegion").textContent = data.region;
    document.getElementById("rPred").textContent = fcVals.join(", ");

    // ✅ Show validation metrics
    const valBox = document.getElementById("validationBox");
    if (valBox) {
      valBox.innerHTML = `
        <h4>Validation Metrics (1970–2017)</h4>
        <p>
          MAE: ${val.mae_validation?.toFixed(3) ?? "—"} |
          RMSE: ${val.rmse_validation?.toFixed(3) ?? "—"} |
          SMAPE: ${val.smape_validation?.toFixed(2) ?? "—"}% |
          Corr: ${val.corr_validation?.toFixed(3) ?? "—"}
        </p>
      `;
    }

    renderChart(histPoints, fcPoints);
  } catch (err) {
    console.error("💥 Exception caught in runForecast:", err);
    statusEl.textContent = "❌ " + err.message;
    statusEl.className = "status error";
  }

  console.groupEnd();
}

function renderChart(histPoints, fcPoints) {
  console.group("📈 Chart Rendering");
  console.time("⏱️ Chart Render Time");

  try {
    if (window._chart) {
      window._chart.destroy();
    }

    const ctx = document.getElementById("chart").getContext("2d");
    const boundaryX = "2018-01-01";

    const boundaryPlugin = {
      id: "forecastBoundary",
      afterDatasetsDraw(chart) {
        const x = chart.scales.x.getPixelForValue(boundaryX);
        const ctx = chart.ctx;
        ctx.save();
        ctx.setLineDash([6, 6]);
        ctx.strokeStyle = "#9aa4b2";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, chart.scales.y.top);
        ctx.lineTo(x, chart.scales.y.bottom);
        ctx.stroke();
        ctx.fillStyle = "#667085";
        ctx.font = "12px Inter, system-ui";
        ctx.fillText("Forecast starts", x + 6, chart.scales.y.top + 14);
        ctx.restore();
      },
    };

    window._chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Actual incidents (1970–2017)",
            data: histPoints,
            borderColor: "#1f9d8f",
            backgroundColor: "rgba(31,157,143,0.12)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 0,
            fill: true,
          },
          {
            label: "Forecast (2018–2030)",
            data: fcPoints,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245,158,11,0.1)",
            borderDash: [6, 4],
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 2.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            type: "time",
            time: { unit: "year", tooltipFormat: "MMM yyyy" },
            title: { display: true, text: "Year" },
            grid: { color: "#eef2f7" },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Incidents" },
            grid: { color: "#eef2f7" },
          },
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            backgroundColor: "#fff",
            borderColor: "#ddd",
            borderWidth: 1,
            titleColor: "#000",
            bodyColor: "#000",
            callbacks: {
              title: (items) =>
                new Date(items[0].parsed.x).toLocaleString(undefined, {
                  month: "short",
                  year: "numeric",
                }),
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}`,
            },
          },
          zoom: {
            pan: { enabled: true, mode: "xy" },
            zoom: {
              wheel: { enabled: true, speed: 0.05 },
              mode: "x",
            },
          },
        },
      },
      plugins: [boundaryPlugin],
    });

    document.getElementById("result").style.display = "block";
    console.log("✅ Chart rendered successfully!");
  } catch (err) {
    console.error("💥 Chart rendering error:", err);
  }

  console.timeEnd("⏱️ Chart Render Time");
  console.groupEnd();
}
