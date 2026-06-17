API LINK=https://stock-price-prediction-7.onrender.com/docs#/default/analyze_stock_analyze_stock_post
# 📈 AI Stock Price Prediction Platform

A production-ready stock forecasting application built using Deep Learning, FastAPI, and Streamlit. The platform predicts the next 60 trading-day closing prices for stocks using a pre-trained LSTM model and provides interactive visualizations through a modern web interface.

## 🚀 Live Demo

Live Application: [YOUR_STREAMLIT_LINK]

API Documentation: https://stock-price-prediction-7.onrender.com/docs

## ✨ Features

* Predicts next 60 trading-day stock prices
* Supports US and Indian stock symbols
* Interactive Streamlit dashboard
* FastAPI REST API backend
* Automatic stock data retrieval using Yahoo Finance
* Downloadable prediction charts
* CSV export for future predictions
* Docker-ready deployment
* Cloud deployment on Render

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* TensorFlow / Keras
* LSTM Neural Network
* Scikit-Learn

### Data Source

* Yahoo Finance (yfinance)

### DevOps & Deployment

* Docker
* Render
* GitHub

## 📂 Project Architecture

User → Streamlit Frontend → FastAPI API → LSTM Model → Predictions

## API Endpoints

### Analyze Stock

POST `/analyze_stock`

Input:

{
"stock": "AAPL"
}

### Get Predictions

GET `/get_predictions/{stock}`

Example:

/get_predictions/AAPL

### Download Prediction Chart

GET `/download_chart/{stock}/prediction`

Example:

/download_chart/AAPL/prediction

## Example Symbols

### US Stocks

* AAPL
* MSFT
* TSLA
* NVDA
* META

### Indian Stocks

* RELIANCE.NS
* TCS.NS
* INFY.NS
* HDFCBANK.NS
* BAJFINANCE.NS

## Future Improvements

* Real-time stock monitoring
* Confidence intervals
* Multi-stock comparison
* Sentiment analysis integration
* MLOps pipeline with CI/CD
* AWS deployment

## Author

Aditya Singh
