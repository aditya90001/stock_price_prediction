
const backendURL = "http://127.0.0.1:8000";  // Change if your FastAPI runs elsewhere

document.getElementById("stockForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const stock = document.getElementById("stockInput").value.trim();
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = `<div class='alert alert-info'>⏳ Analyzing ${stock}...</div>`;

  try {
    const formData = new FormData();
    formData.append("stock", stock);

    // Call API
    const res = await fetch(`${backendURL}/analyze_stock`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (res.ok) {
      const preview = data.data_preview;
      const charts = data.charts;

      let tableRows = preview.map(item => `
        <tr>
          <td>${item.Date}</td>
          <td>${item.Predicted_Close.toFixed(2)}</td>
        </tr>
      `).join("");

      resultDiv.innerHTML = `
        <div class="card p-4">
          <h3>✅ Analysis Complete for ${stock}</h3>
          <h5 class="mt-3">📊 Predicted Next 10 Days</h5>
          <div class="table-container">
            <table class="table table-striped">
              <thead><tr><th>Date</th><th>Predicted Close</th></tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>

          <h5 class="mt-4">📈 Charts</h5>
          <div>
            <img src="${backendURL}${charts.prediction}" alt="Prediction Chart">
            <img src="${backendURL}${charts.ema_20_50}" alt="EMA 20/50">
            <img src="${backendURL}${charts.ema_100_200}" alt="EMA 100/200">
            <img src="${backendURL}${charts.crossover_signals}" alt="Crossover Signals">
          </div>
        </div>
      `;
    } else {
      resultDiv.innerHTML = `<div class='alert alert-danger'>❌ Error: ${data.error}</div>`;
    }

  } catch (error) {
    console.error(error);
    resultDiv.innerHTML = `<div class='alert alert-danger'>❌ Something went wrong. Check backend console.</div>`;
  }
});

