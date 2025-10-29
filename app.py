from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

app = FastAPI(title="Stock 60-Day Prediction API", version="2.0")

# Create folders for saving models and charts
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)


# -------- Helper function: train or load LSTM --------
def get_or_train_model(stock, data):
    model_path = f"models/{stock}_model.h5"
    scaler = MinMaxScaler(feature_range=(0, 1))

    close_data = data["Close"].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_data)

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        model.save(model_path)

    return model, scaler


# -------- API endpoint: analyze and predict --------
@app.post("/analyze_stock")
async def analyze_stock(stock: str = Form(...)):
    """
    Analyze stock data, generate 60-day forecast and EMA charts.
    Available chart types for download:
    - prediction : shows next 60-day forecast
    - ema        : shows last 200 days with EMA indicators
    """
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=5 * 365)  # last 5 years

        data = yf.download(stock, start=start, end=end)
        if data.empty or "Close" not in data.columns:
            return JSONResponse({"error": f"No valid data found for {stock}"}, status_code=400)

        model, scaler = get_or_train_model(stock, data)

        # --- 1️⃣ Predict Next 60 Days ---
        last_60_days = data["Close"].values[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60_days)
        next_predictions = []

        seq = last_60_scaled
        for _ in range(60):
            pred = model.predict(np.reshape(seq, (1, 60, 1)), verbose=0)
            next_predictions.append(pred[0][0])
            seq = np.append(seq[1:], pred[0][0])
            seq = np.reshape(seq, (60, 1))

        next_predictions = scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1))
        future_dates = pd.date_range(end + dt.timedelta(days=1), periods=60)

        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": next_predictions.flatten()
        })

        # --- Make Prediction Chart ---
        plt.figure(figsize=(10, 5))
        plt.plot(data.index[-100:], data["Close"].iloc[-100:], label="Last 100 Days Actual")
        plt.plot(prediction_df["Date"], prediction_df["Predicted_Close"], "r--", label="Next 60 Days Forecast")
        plt.title(f"{stock} - 60-Day Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        prediction_chart_path = f"charts/{stock}_prediction.png"
        plt.savefig(prediction_chart_path)
        plt.close()

        # --- 2️⃣ EMA Chart (Technical View) ---
        ema20 = data["Close"].ewm(span=20, adjust=False).mean()
        ema50 = data["Close"].ewm(span=50, adjust=False).mean()
        ema100 = data["Close"].ewm(span=100, adjust=False).mean()
        ema200 = data["Close"].ewm(span=200, adjust=False).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(data["Close"].iloc[-200:], label="Closing Price", color="black")
        plt.plot(ema20.iloc[-200:], label="EMA 20", color="green")
        plt.plot(ema50.iloc[-200:], label="EMA 50", color="orange")
        plt.plot(ema100.iloc[-200:], label="EMA 100", color="red")
        plt.plot(ema200.iloc[-200:], label="EMA 200", color="purple")
        plt.title(f"{stock} - Last 200 Days with EMAs")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        ema_chart_path = f"charts/{stock}_ema.png"
        plt.savefig(ema_chart_path)
        plt.close()

        # --- Return Summary ---
        return {
            "message": f"Analysis complete for {stock}",
            "available_charts": {
                "prediction_chart": f"/download_chart/{stock}/prediction",
                "ema_chart": f"/download_chart/{stock}/ema"
            },
            "next_60_day_forecast": prediction_df.to_dict(orient="records"),
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# -------- Download chart endpoint --------
@app.get("/download_chart/{stock}/{chart_type}")
async def download_chart(stock: str, chart_type: str):
    """
    Download available chart types:
    - prediction
    - ema
    """
    valid_types = ["prediction", "ema"]
    if chart_type not in valid_types:
        return JSONResponse({"error": f"Invalid chart_type. Choose from {valid_types}"}, status_code=400)

    path = f"charts/{stock}_{chart_type}.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=f"{stock}_{chart_type}.png")
    return JSONResponse({"error": "Chart not found"}, status_code=404)


@app.get("/")
async def home():
    return {
        "message": "Welcome to the Stock Forecast API!",
        "usage": {
            "POST /analyze_stock": "Form input: stock=RELIANCE.NS (or any NSE/BSE ticker)",
            "GET /download_chart/{stock}/{chart_type}": "chart_type can be 'prediction' or 'ema'"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
