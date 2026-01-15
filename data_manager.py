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

        # ✅ explicit validation (fix)
        required_cols = {
            (ticker, "Close"),
            ("^VIX", "Close")
        }
        if raw.empty or not required_cols.issubset(set(raw.columns)):
            raise Exception("Incomplete Yahoo data")

        df = raw.xs(ticker, level=1, axis=1).copy()
        vix = raw.xs("^VIX", level=1, axis=1)["Close"]
        df.loc[:, "VIX"] = vix.reindex(df.index).ffill()

    except Exception:
        df = pd.read_csv(
            "spy.csv",
            skiprows=3,
            names= ["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )
        vix = pd.read_csv(
            "vix.csv",
            skiprows=3,
            names= ["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )["Close"]

        df.loc[:, "VIX"] = vix.reindex(df.index).ffill()

    df.loc[:, "VIX_Safe"] = (df["VIX"] < 25).astype(int)
    return df.dropna()
