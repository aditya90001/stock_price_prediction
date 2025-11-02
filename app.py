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
import uvicorn

app = FastAPI(title="Stock Price Prediction API", version="3.1")

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model setup ===
MODEL_PATH = "stock_dl_model.h5"
model = None  # Load lazily

# === Folder for saving charts ===
os.makedirs("charts", exist_ok=True)


@app.on_event("startup")
def verify_files():
    """Check model existence before serving."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model file not found: stock_dl_model.h5")


@app.get("/")
def home():
    return {
        "message": "📈 Welcome to the Stock Price Prediction API 🚀",
        "usage": {
            "POST /analyze_stock": "Analyze any stock and generate 60-day predictions + charts",
            "GET /download_chart/{stock}/{chart_type}": "Download charts: prediction, ema_20_50, ema_100_200",
            "GET /get_predictions/{stock}": "Get next 60 predicted prices as JSON"
        },
    }


@app.post("/analyze_stock")
def analyze_stock(stock: str = Form(...)):
    global model

    try:
        # === Lazy model loading ===
        if model is None:
            print("🔄 Loading model into memory...")
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")

        # === Fetch stock data ===
        data = yf.download(stock, period="2y")
        if data.empty:
            return JSONResponse(status_code=400, content={"error": "Invalid stock symbol or no data found."})

        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # === Prepare input for model ===
        last_60_days = scaled_data[-60:]
        current_input = np.array([last_60_days])

        # === Predict next 60 days ===
        future_predictions = []
        for _ in range(60):
            next_pred = model.predict(current_input, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            current_input = np.append(current_input[:, 1:, :],
                                      np.array([[next_pred]]).reshape(1, 1, 1),
                                      axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # === Prepare output ===
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions.flatten()})
        pred_df.set_index('Date', inplace=True)
        csv_path = f"charts/{stock}_predictions.csv"
        pred_df.to_csv(csv_path)

        # === Generate charts ===
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Historical Close', color='blue')
        plt.plot(pred_df['Predicted_Close'], label='Predicted Next 60 Days', color='orange')
        plt.title(f'{stock} Stock Price Prediction (Next 60 Days)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pred_chart = f"charts/{stock}_prediction.png"
        plt.savefig(pred_chart)
        plt.close()

        return {
            "message": f"✅ Analysis complete for {stock}",
            "charts": {
                "prediction": f"/download_chart/{stock}/prediction",
            },
            "data_preview": pred_df.head(10).to_dict(orient="records"),
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/get_predictions/{stock}")
def get_predictions(stock: str):
    csv_path = f"charts/{stock}_predictions.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "No prediction found. Run /analyze_stock first."})

    pred_df = pd.read_csv(csv_path)
    return {"stock": stock, "next_60_days": pred_df.to_dict(orient="records")}


@app.get("/download_chart/{stock}/{chart_type}")
def download_chart(stock: str, chart_type: str):
    chart_path = f"charts/{stock}_{chart_type}.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png", filename=os.path.basename(chart_path))
    return JSONResponse(status_code=404, content={"error": "Chart not found"})


# === Render entry point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

