import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker, start_date, end_date):
    try:
        spy = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            prepost=False,
            progress=False
        )

        vix = yf.download(
            "^VIX",
            start=start_date,
            end=end_date,
            interval="1d",
            prepost=False,
            progress=False
        )["Close"]

        spy.loc[:, "VIX"] = vix.reindex(spy.index).ffill()

    except:
        spy = pd.read_csv("spy.csv", parse_dates=["Date"], index_col="Date")
        vix = pd.read_csv("vix.csv", parse_dates=["Date"], index_col="Date")["Close"]
        spy.loc[:, "VIX"] = vix.reindex(spy.index).ffill()

    spy.loc[:, "VIX_Safe"] = (spy["VIX"] < 25).astype(int)
    return spy.dropna()
