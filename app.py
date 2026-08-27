import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostRegressor

# --- System Path Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import generate_technical_indicators

# --- Page Configuration ---
st.set_page_config(
    page_title="DSE Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_dse.cbm")

# --- Custom Enterprise CSS Styling ---
st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #F8FAFC;
        }

        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
        }
        [data-testid="stSidebar"] * {
            color: #F8FAFC !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            border-color: #334155 !important;
            color: #FFFFFF !important;
        }

        /* Sidebar Selection Badge */
        .sidebar-selected-box {
            background-color: #1E293B;
            padding: 12px 14px;
            border-radius: 6px;
            margin-top: 12px;
            border-left: 4px solid #2563EB;
        }
        .sidebar-selected-label {
            font-size: 0.75rem;
            color: #94A3B8;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .sidebar-selected-value {
            font-size: 1.25rem;
            color: #FFFFFF;
            font-weight: 700;
        }

        /* Top Header Active Indicator Badge */
        .selected-badge {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF;
            padding: 4px 14px;
            border-radius: 16px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-block;
            margin-left: 8px;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        }

        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .metric-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748B;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F172A;
        }
        .delta-positive {
            color: #166534;
            background-color: #DCFCE7;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 6px;
        }
        .delta-negative {
            color: #991B1B;
            background-color: #FEE2E2;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 6px;
        }

        .dashboard-header {
            color: #0F172A;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .dashboard-subtitle {
            color: #64748B;
            font-size: 0.95rem;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
        }

        .stExpander {
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            background-color: #FFFFFF !important;
        }
    </style>
''', unsafe_allow_html=True)

# --- Data & Model Loaders ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH): return None
    df = pd.read_csv(DATA_PATH)
    return generate_technical_indicators(df)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

df = load_data()
model = load_model()

if df is None or model is None:
    st.error("System Error: Required data artifacts (dataset/model) were not found.")
    st.stop()

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("### 📊 Market Navigation")
    tickers = sorted(df["Trading_Code"].unique().tolist())
    default_index = tickers.index("GP") if "GP" in tickers else 0
    
    selected_ticker = st.selectbox(
        "Select Instrument",
        tickers,
        index=default_index,
        help="Select a Dhaka Stock Exchange trading symbol to analyze."
    )
    
    # 1. Live Display Box below Selection
    st.markdown(f'''
        <div class="sidebar-selected-box">
            <div class="sidebar-selected-label">ACTIVE SELECTION</div>
            <div class="sidebar-selected-value">📌 {selected_ticker}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Model Specs**")
    st.caption("Algorithm: CatBoost Regressor")
    st.caption("Features: RSI, MACD, SMA 20, Volume")
    st.caption("Target: Next-Day Close Price")

# --- Main Dashboard ---
st.markdown(f'<div class="dashboard-header">Dhaka Stock Exchange — Quantitative Forecast</div>', unsafe_allow_html=True)

# 2. Main Banner Active Stock Indicator
st.markdown(f'''
    <div class="dashboard-subtitle">
        Machine Learning Price Prediction System | Currently Analyzing: <span class="selected-badge">📌 {selected_ticker}</span>
    </div>
''', unsafe_allow_html=True)

stock_df = df[df["Trading_Code"] == selected_ticker].sort_values("Date").copy()

if stock_df.empty:
    st.warning(f"No records available for ticker symbol '{selected_ticker}'.")
    st.stop()

latest_record = stock_df.iloc[-1]
last_date = latest_record["Date"]
last_close = latest_record["Close"]

feature_columns = ["RSI", "MACD", "SMA_20", "Volume"]
input_features = latest_record[feature_columns].to_frame().T

# Inference
predicted_close = model.predict(input_features)[0]
price_change = predicted_close - last_close
percent_change = (price_change / last_close) * 100

delta_class = "delta-positive" if price_change >= 0 else "delta-negative"
arrow = "▲" if price_change >= 0 else "▼"

# --- Metrics Grid ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">TRADING SYMBOL</div>
            <div class="metric-value" style="color: #2563EB;">{selected_ticker}</div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">LAST RECORD DATE</div>
            <div class="metric-value">{str(last_date)}</div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">LAST CLOSE PRICE</div>
            <div class="metric-value">BDT {last_close:.2f}</div>
        </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">FORECAST CLOSE</div>
            <div class="metric-value">BDT {predicted_close:.2f}</div>
            <div class="{delta_class}">{arrow} {price_change:+.2f} BDT ({percent_change:+.2f}%)</div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Plotly Visualization ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["Close"],
    mode='lines',
    name=f'{selected_ticker} Historical Close',
    line=dict(color='#2563EB', width=2)
))

if "SMA_20" in stock_df.columns:
    fig.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["SMA_20"],
        mode='lines',
        name='20-Day Moving Avg',
        line=dict(color='#F59E0B', width=1.5, dash='dash')
    ))

fig.update_layout(
    title=dict(
        text=f"Historical Price Performance & Trend — {selected_ticker}",
        font=dict(size=16, color="#0F172A", family="Inter")
    ),
    xaxis_title="Date",
    yaxis_title="Price (BDT)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#475569")
    ),
    hovermode="x unified",
    template="plotly_white",
    height=520,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(gridcolor='#F1F5F9'),
    yaxis=dict(gridcolor='#F1F5F9')
)

st.plotly_chart(fig, use_container_width=True)

# --- Data Table Expander ---
with st.expander(f"📂 View Historical Data Records for {selected_ticker}"):
    display_df = stock_df[['Date', 'Trading_Code', 'Close', 'SMA_20', 'RSI', 'MACD', 'Volume']].tail(25)
    st.dataframe(display_df, use_container_width=True)
