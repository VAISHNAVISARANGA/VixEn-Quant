

import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_market_data(ticker="SPY", period="35y", start_date=None, end_date=None):
    try:
        if start_date and end_date:
            raw = yf.download([ticker, "^VIX"], start=start_date, end=end_date, progress=False)
        else:
            raw = yf.download([ticker, "^VIX"], period=period, progress=False)

        if raw.empty:
            raise Exception

        df = raw.xs(ticker, level=1, axis=1)
        vix = raw.xs("^VIX", level=1, axis=1)["Close"]
        df["VIX"] = vix.ffill()
    except:
        df = pd.read_csv("spy.csv", parse_dates=["Date"], index_col="Date")
        vix = pd.read_csv("vix.csv", parse_dates=["Date"], index_col="Date")["Close"]
        df["VIX"] = vix.reindex(df.index).ffill()

    df["VIX_Safe"] = (df["VIX"] < 25).astype(int)
    return df.dropna()
