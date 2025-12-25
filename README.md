🦊 VixEn Quant: Volatility-Aware Ensemble Intelligence
VixEn Quant is a high-performance algorithmic trading dashboard that combines Ensemble Machine Learning (XGBoost + LightGBM) with a dynamic VIX Volatility Shield. It is designed to navigate the US Equity markets (SPY) by prioritizing capital preservation during high-volatility regimes while capturing alpha during stable trends.

🚀 Core Strategy: "The VixEn Logic"
The system operates on a dual-filter architecture to decide between BUY (Long) and CASH (Protective) positions:

AI Ensemble (The Brain): An ensemble of XGBoost and LightGBM models analyzes over 20 technical indicators (RSI, MACD, Moving Averages, etc.) to predict the probability of a positive return for the next trading day.

VIX Filter (The Shield): A real-time volatility "circuit breaker." If the CBOE Volatility Index (VIX) exceeds a user-defined threshold (e.g., 30), the strategy forces a move to cash, regardless of the AI's bullishness.

🛠️ Key Features
Live Market Pulse: Real-time US market clock with automatic holiday and early-close detection (optimized for global users in time zones like IST).

Adaptive Backtesting: Interactive date sliders and risk-tolerance controls to test the strategy against 30+ years of historical market data.

Explainable AI (XAI): Visualizes the "Selection Rank" of top features, showing exactly which technical indicators are driving the current market forecast.

Automated Intelligence Reports: Generates professional-grade PDF reports including Equity Curves, Rolling Sharpe Ratios, and Model Architecture summaries.

Performance Metrics: Real-time calculation of Annualized Sharpe Ratio, CAGR, and Maximum Drawdown.

📊 Performance Benchmark
Benchmark: S&P 500 (SPY)

Strategy Goal: Outperform the benchmark on a risk-adjusted basis (Higher Sharpe Ratio) by significantly reducing drawdowns during market crashes (2008, 2020, 2022).

📦 Tech Stack
Language: Python

Framework: Streamlit (Web Dashboard)

ML Libraries: Scikit-learn, XGBoost, LightGBM, Joblib

Data Engine: Yahoo Finance (yfinance), Pandas Market Calendars

Visualization: Plotly Express

Reporting: FPDF, Kaleido

🚦 Getting Started
Clone the Repo:

Bash

git clone https://github.com/your-username/vixen-quant.git
Install Requirements:
## Prerequisites
- Python 3.8+ (project uses a virtualenv at `market` in the repo)
- PowerShell or Command Prompt on Windows

## Activate virtual environment (Windows)
PowerShell:
```powershell
& .\market\Scripts\Activate.ps1
```
Command Prompt:
```cmd
market\Scripts\activate.bat
```
Bash

pip install -r requirements.txt
Run the App:

Bash

streamlit run app.py
Disclaimer: VixEn Quant is an educational tool and does not constitute financial advice. Past performance is not indicative of future results.
