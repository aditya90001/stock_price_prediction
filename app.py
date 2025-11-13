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

# === FastAPI App ===
app = FastAPI(title="📈 Stock Price Prediction API", version="4.0")

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for frontend)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# === Global Variables ===
MODEL_PATH = "stock_dl_model.h5"
model = None  # Lazy loaded model
os.makedirs("charts", exist_ok=True)  # Folder for charts

# === Check model availability ===
@app.on_event("startup")
def verify_files():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model file not found: stock_dl_model.h5")


# === Root Endpoint ===
@app.get("/")
def home():
    return {
        "message": "🚀 Welcome to the Stock Price Prediction API",
        "usage": {
            "POST /analyze_stock": "Analyze a stock and generate 60-day predictions + EMA charts",
            "GET /download_chart/{stock}/{chart_type}": "Download charts: prediction, ema_20_50, ema_100_200, crossover_signals",
            "GET /get_predictions/{stock}": "Get next 60 predicted prices as JSON",
        },
    }


# === Analyze Stock Endpoint ===
@app.post("/analyze_stock")
def analyze_stock(stock: str = Form(...)):
    global model

    try:
        # --- Load Model Lazily ---
        if model is None:
            print("🔄 Loading model into memory...")
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")

        # --- Fetch Stock Data ---
        data = yf.download(stock, period="2y")
        if data.empty:
            return JSONResponse(status_code=400, content={"error": "Invalid stock symbol or no data found."})

        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # --- Predict Next 60 Days ---
        last_60_days = scaled_data[-60:]
        current_input = np.array([last_60_days])

        future_predictions = []
        for _ in range(60):
            next_pred = model.predict(current_input, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            current_input = np.append(
                current_input[:, 1:, :],
                np.array([[next_pred]]).reshape(1, 1, 1),
                axis=1
            )

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # ✅ --- Prepare DataFrame (Fixed Indentation Issue) ---
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60)

        # ✅ Create prediction DataFrame (Date as normal column)
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions.flatten()
        })

        # ✅ Save CSV with Date as normal column (frontend fix)
        csv_path = f"charts/{stock}_predictions.csv"
        pred_df.to_csv(csv_path, index=False)

        # === Chart 1: Prediction Chart ===
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Historical Close', color='blue')
        plt.plot(pred_df['Date'], pred_df['Predicted_Close'], label='Predicted Next 60 Days', color='orange')
        plt.title(f'{stock} Stock Price Prediction (Next 60 Days)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pred_chart = f"charts/{stock}_prediction.png"
        plt.savefig(pred_chart)
        plt.close()

        # === Chart 2: EMA 20 & 50 ===
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['EMA_20'], label='EMA 20', color='green')
        plt.plot(data['EMA_50'], label='EMA 50', color='red')
        plt.title(f'{stock} - EMA 20 & EMA 50 Chart')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        ema_20_50_chart = f"charts/{stock}_ema_20_50.png"
        plt.savefig(ema_20_50_chart)
        plt.close()

        # === Chart 3: EMA 100 & 200 ===
        data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['EMA_100'], label='EMA 100', color='purple')
        plt.plot(data['EMA_200'], label='EMA 200', color='orange')
        plt.title(f'{stock} - EMA 100 & EMA 200 Chart')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        ema_100_200_chart = f"charts/{stock}_ema_100_200.png"
        plt.savefig(ema_100_200_chart)
        plt.close()

        # === Chart 4: EMA 20/50 Crossover (Buy/Sell) ===
        data['Signal'] = 0
        data.loc[data['EMA_20'] > data['EMA_50'], 'Signal'] = 1
        data['Crossover'] = data['Signal'].diff()

        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.7)
        plt.plot(data['EMA_20'], label='EMA 20', color='green', alpha=0.8)
        plt.plot(data['EMA_50'], label='EMA 50', color='red', alpha=0.8)
        plt.scatter(data.index[data['Crossover'] == 1],
                    data['Close'][data['Crossover'] == 1],
                    label='Buy Signal', marker='^', color='green', s=100)
        plt.scatter(data.index[data['Crossover'] == -1],
                    data['Close'][data['Crossover'] == -1],
                    label='Sell Signal', marker='v', color='red', s=100)
        plt.title(f"{stock} - EMA 20/50 Crossover Buy & Sell Signals")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        crossover_chart = f"charts/{stock}_crossover_signals.png"
        plt.savefig(crossover_chart)
        plt.close()

        # === Return JSON Response ===
        return {
            "message": f"✅ Analysis complete for {stock}",
            "charts": {
                "prediction": f"/download_chart/{stock}/prediction",
                "ema_20_50": f"/download_chart/{stock}/ema_20_50",
                "ema_100_200": f"/download_chart/{stock}/ema_100_200",
                "crossover_signals": f"/download_chart/{stock}/crossover_signals"
            },
            "data_preview": pred_df.head(10).to_dict(orient="records"),
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# === Get Predictions ===
@app.get("/get_predictions/{stock}")
def get_predictions(stock: str):
    csv_path = f"charts/{stock}_predictions.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "No prediction found. Run /analyze_stock first."})
    pred_df = pd.read_csv(csv_path)
    return {"stock": stock, "next_60_days": pred_df.to_dict(orient="records")}


# === Download Chart Endpoint ===
@app.get("/download_chart/{stock}/{chart_type}")
def download_chart(stock: str, chart_type: str):
    chart_path = f"charts/{stock}_{chart_type}.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png", filename=os.path.basename(chart_path))
    return JSONResponse(status_code=404, content={"error": "Chart not found"})


# === Render Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
