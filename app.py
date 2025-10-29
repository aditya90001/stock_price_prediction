import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (Render doesn't have one)

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Price Prediction API", version="1.0")

MODEL_DIR = "models"
CHART_DIR = "charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Welcome to the Stock Prediction API 🚀",
        "usage": {
            "POST /analyze_stock": "Predict next 60 days for given stock symbol (e.g. TCS.NS)",
            "GET /download_chart/{stock}/{chart_type}": "Download chart - chart_type = 'prediction' or 'ema'",
        },
    }


@app.post("/analyze_stock")
def analyze_stock(stock: str = Form(...)):
    try:
        # Fetch data for last 3 years
        end = dt.datetime.now()
        start = end - dt.timedelta(days=3 * 365)
        data = yf.download(stock, start=start, end=end)

        if data.empty or "Close" not in data.columns:
            return JSONResponse(status_code=400, content={"error": "Invalid stock symbol or no data available"})

        # Save raw CSV
        csv_path = f"{CHART_DIR}/{stock}.csv"
        data.to_csv(csv_path)

        # Create EMA charts
        data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(data["Close"], label="Close", color="blue")
        plt.plot(data["EMA20"], label="EMA 20", color="red")
        plt.plot(data["EMA50"], label="EMA 50", color="green")
        plt.title(f"{stock} - EMA Chart")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        ema_path = f"{CHART_DIR}/{stock}_ema.png"
        plt.savefig(ema_path)
        plt.close()

        # Prepare data for LSTM
        close_data = data[["Close"]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        seq_len = 60
        x_train, y_train = [], []
        for i in range(seq_len, len(scaled_data)):
            x_train.append(scaled_data[i - seq_len:i])
            y_train.append(scaled_data[i])
        x_train, y_train = np.array(x_train), np.array(y_train)

        model_path = os.path.join(MODEL_DIR, f"{stock}.h5")

        # Load model if exists, else train
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
            model.save(model_path)

        # Predict next 60 days
        last_60_days = scaled_data[-seq_len:]
        future_input = last_60_days.reshape(1, seq_len, 1)
        predictions = []

        for _ in range(60):
            pred = model.predict(future_input, verbose=0)
            predictions.append(pred[0, 0])
            future_input = np.append(future_input[:, 1:, :], [[pred]], axis=1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        future_dates = pd.date_range(end + dt.timedelta(days=1), periods=60)
        future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": predictions.flatten()})

        # Plot prediction chart
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data["Close"], label="Historical Close")
        plt.plot(future_df["Date"], future_df["Predicted_Close"], label="Predicted Next 60 Days", color="red")
        plt.title(f"{stock} - 60-Day Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        pred_path = f"{CHART_DIR}/{stock}_prediction.png"
        plt.savefig(pred_path)
        plt.close()

        return {
            "message": f"Analysis complete for {stock}",
            "available_charts": ["prediction", "ema"],
            "download_endpoints": {
                "prediction_chart": f"/download_chart/{stock}/prediction",
                "ema_chart": f"/download_chart/{stock}/ema",
            },
            "next_60_days": future_df.to_dict(orient="records")[:10],  # show sample
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/download_chart/{stock}/{chart_type}")
def download_chart(stock: str, chart_type: str):
    chart_file = f"{CHART_DIR}/{stock}_{chart_type}.png"
    if not os.path.exists(chart_file):
        return JSONResponse(status_code=404, content={"error": "Chart not found"})
    return FileResponse(chart_file, media_type="image/png", filename=f"{stock}_{chart_type}.png")

