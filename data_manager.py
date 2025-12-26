
import yfinance as yf
import streamlit as st
import pandas as pd

def fetch_market_data(ticker="SPY", start_date=None, end_date=None, period="max"):

    try:
        # Download data
        if start_date and end_date:
            raw_data = yf.download([ticker, "^VIX"], start=start_date, end=end_date, progress=False)
        else:
            raw_data = yf.download([ticker, "^VIX"], period=period, progress=False)

        
        # DEBUG: This will show up in your "Manage App" logs
        print(f"DEBUG: Downloaded data columns: {raw_data.columns}")
        
        if raw_data.empty:
            print("DEBUG: raw_data is empty!")
            return pd.DataFrame()

        # ROOT CAUSE FIX: Handling the Multi-Index layers
        # Newer yfinance versions put the Price Type first, then the Ticker
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Safe extraction using .xs (Cross-Section)
            df = raw_data.xs('Close', level=0, axis=1)[[ticker, '^VIX']].copy()
            df.columns = ['Close', 'VIX']
        else:
            # Fallback for older versions or single-ticker returns
            df = raw_data[['Close', 'VIX']]

        # Fill any missing VIX values (VIX sometimes has gaps)
        df['VIX'] = df['VIX'].ffill()
        df['VIX_Safe'] = (df['VIX'] < 25).astype(int)
        
        return df.dropna()

    except Exception as e:
        print(f"DEBUG: Critical Fetch Error: {e}")
        return pd.DataFrame()