from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

app = FastAPI(title="Stock Price Prediction API", version="2.0")

# Load the trained LSTM model
MODEL_PATH = "stock_dl_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file 'stock_dl_model.h5' not found in project directory.")
model = load_model(MODEL_PATH)

# Folder for charts
os.makedirs("charts", exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to Stock Price Prediction API! Use /analyze_stock endpoint to predict."}


@app.post("/analyze_stock")
def analyze_stock(stock: str = Form(...)):
    try:
        # Fetch last 2 years of stock data
        data = yf.download(stock, period="2y")
        if data.empty:
            return JSONResponse(status_code=400, content={"error": "Invalid stock symbol or no data found."})

        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare last 60 days for input
        last_60_days = scaled_data[-60:]
        current_input = last_60_days.reshape(1, 60, 1)

        # Predict next 60 days
        future_predictions = []
        for i in range(60):
            next_pred = model.predict(current_input, verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)

        # Inverse transform predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60)

        # Combine predictions into DataFrame
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions.flatten()})
        pred_df.set_index('Date', inplace=True)

        # === CHARTS ===
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Historical Close Price', color='blue')
        plt.plot(pred_df['Predicted Close'], label='Predicted Next 60 Days', color='orange')
        plt.title(f'{stock} Stock Price Prediction for Next 60 Days')
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        prediction_chart = f"charts/{stock}_prediction.png"
        plt.savefig(prediction_chart)
        plt.close()

        # === EMA 20 & 50 CHART ===
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close', color='blue')
        plt.plot(data['EMA20'], label='EMA 20', color='orange')
        plt.plot(data['EMA50'], label='EMA 50', color='green')
        plt.title(f'{stock} - EMA 20 & EMA 50')
        plt.legend()
        plt.tight_layout()

        ema_20_50_chart = f"charts/{stock}_ema_20_50.png"
        plt.savefig(ema_20_50_chart)
        plt.close()

        # === EMA 100 & 200 CHART ===
        data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close', color='blue')
        plt.plot(data['EMA100'], label='EMA 100', color='purple')
        plt.plot(data['EMA200'], label='EMA 200', color='red')
        plt.title(f'{stock} - EMA 100 & EMA 200')
        plt.legend()
        plt.tight_layout()

        ema_100_200_chart = f"charts/{stock}_ema_100_200.png"
        plt.savefig(ema_100_200_chart)
        plt.close()

        return {
            "message": f"Successfully generated 60-day prediction for {stock}",
            "charts": {
                "prediction": f"/download_chart/{stock}/prediction",
                "ema_20_50": f"/download_chart/{stock}/ema_20_50",
                "ema_100_200": f"/download_chart/{stock}/ema_100_200",
            },
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/download_chart/{stock}/{chart_type}")
def download_chart(stock: str, chart_type: str):
    chart_path = f"charts/{stock}_{chart_type}.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png", filename=os.path.basename(chart_path))
    return JSONResponse(status_code=404, content={"error": "Chart not found"})


# Run on Render / localhost
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
