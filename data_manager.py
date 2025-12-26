import yfinance as yf
import streamlit as st
import pandas as pd
import os

@st.cache_data(ttl=3600) # Cache for 1 hour to prevent hitting rate limits
def fetch_market_data(ticker="SPY", start_date=None, end_date=None, period="max"):
    backup_file = "historical_backup.csv"
    
    try:
        # 1. Attempt Live Download
        if start_date and end_date:
            raw_data = yf.download([ticker, "^VIX"], start=start_date, end=end_date, progress=False)
        else:
            raw_data = yf.download([ticker, "^VIX"], period=period, progress=False)

        # If rate limited, raw_data might be empty or missing tickers
        if raw_data.empty or ticker not in raw_data.columns.get_level_values(1):
            raise ValueError("Rate limited by Yahoo Finance")

        # 2. Extract Data (Handling Multi-Index)
        # We use 'Adj Close' for SPY because it accounts for dividends (standard for Quants)
        # We use 'Close' for VIX because it doesn't have dividends
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = pd.DataFrame({
                'Close': raw_data['Adj Close'][ticker],
                'VIX': raw_data['Close']['^VIX']
            })
        else:
            df = raw_data[['Adj Close', 'Close']].copy()
            df.columns = ['Close', 'VIX']

    except Exception as e:
        print(f"DEBUG: Live Fetch Failed: {e}. Attempting Backup...")
        
        # 3. Fallback to Backup File
        if os.path.exists(backup_file):
            st.sidebar.warning("⚠️ Using Historical Backup (Yahoo Rate Limit active)")
            # Load CSV and handle the header levels
            df_backup = pd.read_csv(backup_file, header=[0, 1], index_col=0, parse_dates=True)
            
            df = pd.DataFrame({
                'Close': df_backup['Adj Close'][ticker],
                'VIX': df_backup['Close']['^VIX']
            })
            
            # Filter by date if user selected a range
            if start_date and end_date:
                df = df.loc[start_date:end_date]
        else:
            st.error("No live data or backup file found.")
            return pd.DataFrame()

    # Final Cleaning
    df['VIX'] = df['VIX'].ffill()
    df['VIX_Safe'] = (df['VIX'] < 25).astype(int)
    
    return df.dropna()