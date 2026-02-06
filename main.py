from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# IMPORTANT for server environments (Render / Railway)
import matplotlib
matplotlib.use("Agg")

# === FastAPI App ===
app = FastAPI(title="📈 Stock Price Prediction API", version="4.0")

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global Variables ===
MODEL_PATH = "stock_dl_model.h5"
model = None
os.makedirs("charts", exist_ok=True)

# === Startup Check ===
@app.on_event("startup")
def verify_files():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model file not found:", MODEL_PATH)

# === Root ===
@app.get("/")
def home():
    return {
        "message": "🚀 Welcome to the Stock Price Prediction API",
        "usage": {
            "POST /analyze_stock": "Analyze stock & generate predictions",
            "GET /get_predictions/{stock}": "Get next 60 predicted prices",
            "GET /download_chart/{stock}/{chart_type}": "Download charts"
        }
    }

# === Analyze Stock ===
@app.post("/analyze_stock")
def analyze_stock(stock: str = Form(...)):
    global model

    try:
        # Load model once
        if model is None:
            model = load_model(MODEL_PATH)

        # Fetch data
        data = yf.download(stock, period="2y", auto_adjust=True, threads=False)
        if data.empty:
            return JSONResponse(status_code=400, content={"error": "Invalid stock symbol"})

        data = data[['Close']]

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Prepare input
        last_60_days = scaled_data[-60:]
        current_input = np.array([last_60_days])

        # Predict next 60 days
        future_predictions = []
        for _ in range(60):
            next_pred = model.predict(current_input, verbose=0)[0][0]
            future_predictions.append(next_pred)
            current_input = np.append(
                current_input[:, 1:, :],
                [[[next_pred]]],
                axis=1
            )

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        # 🔥 FIXED PART: use CURRENT DATE + BUSINESS DAYS
        today = pd.Timestamp.today().normalize()
        future_dates = pd.bdate_range(
            start=today + pd.Timedelta(days=1),
            periods=60
        )

        # Prediction DataFrame
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_predictions.flatten()
        })

        # Save CSV
        csv_path = f"charts/{stock}_predictions.csv"
        pred_df.to_csv(csv_path, index=False)

        # === Prediction Chart ===
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label="Historical Close")
        plt.plot(pred_df['Date'], pred_df['Predicted_Close'], label="Predicted")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"charts/{stock}_prediction.png")
        plt.close()

        return {
            "message": f"✅ Analysis complete for {stock}",
            "data_preview": pred_df.head(10).to_dict(orient="records"),
            "charts": {
                "prediction": f"/download_chart/{stock}/prediction"
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Get Predictions ===
@app.get("/get_predictions/{stock}")
def get_predictions(stock: str):
    csv_path = f"charts/{stock}_predictions.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "Run /analyze_stock first"})
    df = pd.read_csv(csv_path)
    return {"stock": stock, "next_60_days": df.to_dict(orient="records")}

# === Download Charts ===
@app.get("/download_chart/{stock}/{chart_type}")
def download_chart(stock: str, chart_type: str):
    path = f"charts/{stock}_{chart_type}.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Chart not found"})

# === Entry Point ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
