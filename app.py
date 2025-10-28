from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = FastAPI(title="Stock Data API", version="1.0")

# ---------- Utility Functions ----------

def generate_stock_data(stock_name: str):
    """Generate random demo stock data for simulation."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200)
    close_prices = np.random.randint(100, 200, size=200).astype(float)
    df = pd.DataFrame({"Date": dates, "Close": close_prices})
    df.set_index("Date", inplace=True)

    # Add moving averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()

    return df


def save_chart(df, stock_name):
    """Save line chart and return file path."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["Close"], label="Closing Price", color="blue")
    plt.plot(df["SMA_10"], label="SMA 10", color="green")
    plt.plot(df["SMA_30"], label="SMA 30", color="red")
    plt.title(f"{stock_name} - Closing Price & Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    chart_path = f"outputs/{stock_name}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path


# ---------- API Endpoints ----------

@app.get("/")
def root():
    """Root endpoint for testing."""
    return {"message": "Welcome to the FastAPI Stock Data API 🚀"}


@app.post("/generate_stock_data")
def generate_stock_endpoint(stock: str = Form("DEMO")):
    """
    Generate mock stock data for the given symbol.
    Returns JSON with summary, chart path, and CSV link.
    """
    df = generate_stock_data(stock)
    chart_path = save_chart(df, stock)

    csv_path = f"outputs/{stock}_data.csv"
    df.to_csv(csv_path)

    # Create summary (convert to dict)
    summary = df.describe().to_dict()

    return JSONResponse({
        "stock": stock,
        "message": f"Stock data generated for {stock}",
        "chart_path": chart_path,
        "csv_path": csv_path,
        "summary": summary
    })


@app.get("/download_csv/{stock_name}")
def download_csv(stock_name: str):
    """Download CSV file for a stock."""
    file_path = f"outputs/{stock_name}_data.csv"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=f"{stock_name}_data.csv")
    return JSONResponse({"error": "CSV file not found."}, status_code=404)


@app.get("/download_chart/{stock_name}")
def download_chart(stock_name: str):
    """Download chart image for a stock."""
    file_path = f"outputs/{stock_name}_chart.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=f"{stock_name}_chart.png")
    return JSONResponse({"error": "Chart file not found."}, status_code=404)
