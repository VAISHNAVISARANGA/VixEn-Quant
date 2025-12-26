
import yfinance as yf
import streamlit as st
@st.cache_data(ttl=3600)
def fetch_market_data(ticker="SPY", start_date=None, end_date=None, period="35y"):
   
    if start_date and end_date:
        raw_data = yf.download([ticker, "^VIX"], start=start_date, end=end_date, progress=False)
    else:
        raw_data = yf.download([ticker, "^VIX"], period=period, progress=False)

    # Extract ticker data
    df = raw_data.xs(ticker, level=1, axis=1).copy()
    
    # Extract VIX Close
    vix_close = raw_data.xs('^VIX', level=1, axis=1)['Close']
    df['VIX'] = vix_close

    df['VIX'] = df['VIX'].ffill() 
    
    df['VIX_Safe'] = (df['VIX'] < 25).astype(int)
    return df.dropna()
