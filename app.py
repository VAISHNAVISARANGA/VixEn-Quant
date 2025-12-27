import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta

# Import your backend logic (ensure these files are in the same directory)
from data_manager import fetch_market_data
from feature_engine2 import create_features
from vix_filter import apply_vix_strategy
from evaluator import calculate_sharpe, calculate_rolling_sharpe, calculate_max_drawdown, calculate_CAGR

import pandas_market_calendars as mcal
from datetime import datetime
import pytz
import streamlit as st
import sklearn
print("SKLEARN VERSION:", sklearn.__version__)

def get_automated_market_status():
    # 1. Load the NYSE Calendar
    nyse = mcal.get_calendar('NYSE')
    
    # 2. Get current time in New York
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # 3. Get the market schedule for today
    # This automatically handles holidays and early closes!
    schedule = nyse.schedule(start_date=now_ny, end_date=now_ny)
    
    if schedule.empty:
        # If today isn't in the schedule, it's a holiday or weekend
        return "🔴 Market Closed (Holiday/Weekend)", "N/A"
    
    # Extract opening and closing times for today
    market_open = schedule.iloc[0]['market_open'].to_pydatetime()
    market_close = schedule.iloc[0]['market_close'].to_pydatetime()
    
    # 4. Logic to determine status
    if now_ny < market_open:
        wait_time = market_open - now_ny
        return "🟡 Pre-Market", f"Opens in {str(wait_time).split('.')[0]}"
    
    elif now_ny > market_close:
        return "🔴 Market Closed", "N/A"
    
    else:
        time_left = market_close - now_ny
        return "🟢 Market Open", f"Closes in {str(time_left).split('.')[0]}"

from fpdf import FPDF
import tempfile
import os

def create_pdf_report(df_slice, metrics, inputs, figs):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "VixEn Quant: Strategy Intelligence Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # Section 1: Configuration
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Strategy Configuration", ln=True)
    pdf.set_font("Arial", '', 12)
    for key, val in inputs.items():
        pdf.cell(0, 8, f"- {key}: {val}", ln=True)
    pdf.ln(5)

    # Section 2: Performance Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Performance Metrics", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"- Annualized Sharpe: {metrics['sharpe']:.2f}", ln=True)
    pdf.cell(0, 8, f"- CAGR: {metrics['cagr']*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"- Max Drawdown: {metrics['mdd']*100:.2f}%", ln=True)
    pdf.ln(5)

    # Section 3: Visual Analytics (Graphs)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Visual Analytics", ln=True)
    
    # Save Plotly figures to temporary images and add to PDF
    for i, fig in enumerate(figs):
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmpfile.close()
        fig.write_image(tmpfile.name)
        pdf.image(tmpfile.name, x=10, w=180)
        pdf.ln(2)
        os.remove(tmpfile.name)


    return pdf.output(dest='S').encode('latin-1')
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VixEn Quant",page_icon="🦊", layout="wide")

import os

@st.cache_resource
def load_production_assets():
    # Get the absolute path to the folder this script is in
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "trading_ensemble_prod.pkl")
    
    if not os.path.exists(model_path):
        # This will print in your Streamlit logs so you can debug
        st.error(f"Critical Error: {model_path} not found!")
        st.write("Files currently in directory:", os.listdir(base_path))
        st.stop()
        
    return joblib.load(model_path)

@st.cache_data
def get_processed_data():
    """Fetch and prepare fresh market data."""
    raw = fetch_market_data()
    featured = create_features(raw)
    return raw, featured

# --- LOAD DATA ---
try:
    assets = load_production_assets()
    raw_data, featured_data = get_processed_data()
except Exception as e:
    st.error(f"Error loading assets or data: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---

st.sidebar.markdown("<h3 style='font-size:40px;'>🛠️ Strategy Controls</h3>", unsafe_allow_html=True)
st.autorefresh(interval=1000, key="market_timer")
status, countdown = get_automated_market_status()


st.sidebar.subheader("🕒 Live Market Clock")

st.sidebar.markdown(
    f"<p style='font-size:14px; margin-bottom:4px;'>Status</p>"
    f"<p style='font-size:18px; font-weight:600;'> {status}</p>",
    unsafe_allow_html=True
)

if countdown != "N/A":
    st.sidebar.markdown(
        f"<p style='font-size:13px;'>⏱️ {countdown}</p>",
        unsafe_allow_html=True
    )


# 1. Ticker Selection (Fixed to SPY)
st.sidebar.selectbox("Market Ticker", options=["SPY"], index=0, help="Fixed to S&P 500 ETF (SPY) for this strategy.")

# 2. Date Selection (Start and End Inputs)
st.sidebar.subheader("Analysis Period")
# 1. Safety Check: Ensure data exists before trying to find the min/max date
if featured_data is not None and not featured_data.empty:
    # Convert Pandas Timestamp to standard Python date using .date()
    min_date = featured_data.index.min().date()
    
    # max_date should be today's date
    max_date = datetime.today().date()
    
    # Ensure default_start (5 years ago) is a date object and not before min_date
    five_years_ago = max_date - timedelta(days=365*5)
    default_start = max(min_date, five_years_ago)
else:
    st.error("⚠️ Data Fetch Failed. Check your data_manager.py and internet connection.")
    st.stop()

col_start, col_end = st.sidebar.columns(2)

# 2. Use the sanitized date objects in the widget
start_date = col_start.date_input(
    "Start Date", 
    value=default_start, 
    min_value=min_date, 
    max_value=max_date
)

end_date = col_end.date_input(
    "End Date", 
    value=max_date, 
    min_value=min_date, 
    max_value=max_date
)

# 3. Risk Tolerance (VIX Threshold)
st.sidebar.subheader("Risk Management")
vix_threshold = st.sidebar.slider(
    "VIX Threshold (Risk Off)", 
    min_value=15.0, 
    max_value=40.0, 
    value=30.0, 
    step=0.5,
    help="The strategy moves to cash if VIX exceeds this value."
)

# 4. Model Weights (Expander with Correlated Sliders)
with st.sidebar.expander("Ensemble Model Weights", expanded=False):
    st.write("Adjust the influence of each AI model.")
    
    # Session State for correlated sliders
    if 'xgb_w' not in st.session_state:
        st.session_state.xgb_w = 0.4

    def update_lgb():
        st.session_state.lgb_w = round(1.0 - st.session_state.xgb_w, 2)
    def update_xgb():
        st.session_state.xgb_w = round(1.0 - st.session_state.lgb_w, 2)

    xgb_weight = st.slider("XGBoost Influence", 0.0, 1.0, key='xgb_w', on_change=update_lgb)
    lgb_weight = st.slider("LightGBM Influence", 0.0, 1.0, key='lgb_w', value=1.0-st.session_state.xgb_w, on_change=update_xgb)

# 5. Show Details Toggle
show_details = st.sidebar.checkbox("Show Hyperparameters", value=False)

# 6. Run Model Button
run_model = st.sidebar.button("🚀 Run Strategy", use_container_width=True)

# --- EXECUTION LOGIC ---
period_sharpe = np.nan
period_cagr = np.nan
period_mdd = np.nan
final_signals = np.array([0])
strat_returns = pd.Series(dtype=float)
mkt_returns = pd.Series(dtype=float)
vix_mask_slice = pd.Series(dtype=int)
signals = np.array([0])

# --- EXECUTION LOGIC ---
if run_model:
    # 1. DATA SLICING & PREPARATION
    feat_mask = (featured_data.index >= pd.Timestamp(start_date)) & \
                (featured_data.index <= pd.Timestamp(end_date))
    df_slice = featured_data.loc[feat_mask].copy()
    df_slice = df_slice.ffill()

    raw_mask = (raw_data.index >= pd.Timestamp(start_date)) & \
               (raw_data.index <= pd.Timestamp(end_date))
    
    vix_series = raw_data.loc[raw_mask, 'VIX']
    vix_mask_slice = (vix_series < vix_threshold).astype(int)
    vix_mask_slice = vix_mask_slice.reindex(df_slice.index, method='ffill').fillna(0)

    # 2. MODEL INFERENCE
    assets['model'].set_weights({'xgb': xgb_weight, 'lgb': lgb_weight})
    X_input = df_slice[assets['top_features']]
    signals, probas = assets['model'].predict(X_input)

    # 3. STRATEGY CALCULATIONS
    final_signals = apply_vix_strategy(vix_mask_slice, signals)
    mkt_returns = df_slice['Close'].pct_change().fillna(0)
    strat_returns = final_signals * mkt_returns

    # 4. METRICS GENERATION
    period_sharpe, _, _ = calculate_sharpe(df_slice, final_signals)
    period_mdd, _ = calculate_max_drawdown(strat_returns)
    period_cagr = calculate_CAGR(strat_returns)

    # 5. CREATE FIGURES FOR UI AND REPORT
    # A. Equity Curve Figure
    equity_curve = (1 + strat_returns).cumprod()
    market_curve = (1 + mkt_returns).cumprod()
    comparison_df = pd.DataFrame({
        "AI Strategy": equity_curve,
        "Market (Benchmark)": market_curve
    })
    fig_equity = px.line(comparison_df, title="Cumulative Strategy Growth", 
                         labels={"value": "Cumulative Return", "index": "Date"})

    # B. Rolling Sharpe Figure
    strat_roll, mkt_roll = calculate_rolling_sharpe(strat_returns, mkt_returns, window=252, plot=False)
    roll_df = pd.DataFrame({
        "Strategy Sharpe": strat_roll,
        "Market Sharpe": mkt_roll
    })
    fig_roll = px.line(roll_df, title="Rolling 252-Day Sharpe Ratio",
                        labels={"value": "Sharpe Ratio", "index": "Date"})

    # C. Feature Importance
    feat_df = pd.DataFrame({
        "Feature Name": assets['top_features'],
        "Selection Rank": range(1, 21)
    }).sort_values("Selection Rank", ascending=False)
    fig_feat = px.bar(feat_df, x="Selection Rank", y="Feature Name", orientation='h',
                      title="Top 20 Strategy Drivers", color="Selection Rank", color_continuous_scale="Viridis")

    # D. Ensemble Weights
    pie_data = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM"],
        "Weight": [xgb_weight, lgb_weight]
    })
    fig_pie = px.pie(pie_data, values='Weight', names='Model', hole=0.4, title="Model Weights",
                     color_discrete_sequence=px.colors.sequential.RdBu)

    # 6. PREPARE INPUTS & DOWNLOAD BUTTON
    current_inputs = {
        "Analysis Start": start_date,
        "Analysis End": end_date,
        "VIX Threshold": vix_threshold,
        "XGB Weight": xgb_weight,
        "LGB Weight": lgb_weight
    }
    metrics_dict = {'sharpe': period_sharpe, 'cagr': period_cagr, 'mdd': period_mdd}

    try:
        # Include all 4 figures in the PDF
        figs_to_include = [fig_equity, fig_roll, fig_feat, fig_pie] 
        pdf_data = create_pdf_report(df_slice, metrics_dict, current_inputs, figs_to_include)

        st.sidebar.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_data,
            file_name=f"VixEn_Full_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"PDF Error: {e}")

# --- MAIN DASHBOARD UI ---
st.title("🦊 VixEn Quant")
st.markdown("### *Volatility-Aware Ensemble Intelligence*")

# --- 1. KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Sharpe", f"{period_sharpe:.2f}", 
            delta=f"{period_sharpe - assets['stats']['annualized_sharpe']:.2f} vs Hist" if run_model else None)
col2.metric("CAGR (%)", f"{period_cagr*100:.2f}%" if run_model else "0.00%")
col3.metric("Max Drawdown", f"{period_mdd*100:.2f}%" if run_model else "0.00%")

if run_model:
    is_vix_safe = len(vix_mask_slice) > 0 and vix_mask_slice.iloc[-1] == 1
    ai_bullish = signals[-1] == 1

    if not is_vix_safe:
        main = "CASH"
        reason = "VIX High"
    elif not ai_bullish:
        main = "CASH"
        reason = "AI Bearish"
    else:
        main = "BUY(LONG)"
        reason = ""

    col4.metric("Live Signal", main)

    if reason:
        col4.markdown(
            f"<p style='font-size:16px; margin-top:-16px; color:gray;'>({reason})</p>",
            unsafe_allow_html=True
        )


st.divider()

# --- 2. CHARTS TABS ---
if run_model:
    tab1, tab2 = st.tabs(["📈 Equity Curve", "🛡️ Rolling Risk Analysis"])
    with tab1:
        st.plotly_chart(fig_equity, use_container_width=True)
    with tab2:
        st.plotly_chart(fig_roll, use_container_width=True)

    st.divider()

    # --- 3. MODEL INSIGHTS ---
    st.header("🔍 Model Architecture & Intelligence")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(fig_feat, use_container_width=True)
    with col_right:
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("👈 Adjust the parameters in the sidebar and click 'Run Strategy' to view performance.")

if show_details:
    with st.expander("See Optimized Hyperparameters"):
        st.write("**XGBoost Params:**", assets['params']['xgb'])
        st.write("**LightGBM Params:**", assets['params']['lgb'])

st.caption("Disclaimer: This tool is for educational purposes. Performance reflects current ensemble weights and VIX risk management.")