import streamlit as st
import requests
import pandas as pd

# ========== CONFIG ==========
API_URL = "https://stock-price-prediction-7.onrender.com"

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ========== CUSTOM CSS FOR BEAUTIFUL UI ==========
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e1e2f, #191970); color: white; }
    .glass-card {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: 900;
        color: #00eaff;
        text-shadow: 0px 0px 20px #00eaff;
        margin-bottom: 10px;
    }
    .subtitle { text-align: center; color: #cce7ff; font-size: 18px; margin-top: -10px; margin-bottom: 30px; }
    .stTextInput > div > input { background-color: rgba(255,255,255,0.15); color: white; }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("<div class='title'>📈 AI Stock Price Predictor Using LSTM</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict next 60 days using Deep Learning (LSTM)</div>", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Settings")
    stock = st.text_input("Enter Stock Symbol (e.g., TCS.NS, RELIANCE.NS)")
    run_btn = st.button("🚀 Predict Now")

# ========== IF BUTTON CLICKED ==========
if run_btn:
    if stock.strip() == "":
        st.error("❌ Please enter a valid stock symbol.")
    else:
        with st.spinner("⏳ Analyzing stock... Please wait..."):
            try:
                # API CALL → FastAPI expects FORM DATA
                response = requests.post(f"{API_URL}/analyze_stock", data={"stock": stock})

                if response.status_code != 200:
                    st.error("❌ Error: " + response.json().get("error", "Unknown error"))
                else:
                    result = response.json()

                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.success(f"✔ Prediction complete for **{stock.upper()}**")

                    # Preview
                    st.subheader("🔍 Prediction Preview (Next 10 Days)")
                    df_preview = pd.DataFrame(result["data_preview"])
                    st.dataframe(df_preview, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # ========== SHOW CHARTS ==========
                    st.subheader("📊 Charts")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(f"{API_URL}{result['charts']['prediction']}", caption="Prediction (60 Days)")
                        st.image(f"{API_URL}{result['charts']['ema_20_50']}", caption="EMA 20/50")

                    with col2:
                        st.image(f"{API_URL}{result['charts']['ema_100_200']}", caption="EMA 100/200")
                        st.image(f"{API_URL}{result['charts']['crossover_signals']}", caption="Buy & Sell Signals")

                    # ========== DOWNLOAD CSV ==========
                    st.markdown("### 📥 Download Full Predictions CSV")
                    st.markdown(
                        f"[🟦 Download CSV]({API_URL}/charts/{stock}_predictions.csv)",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"🔥 Error: {e}")
