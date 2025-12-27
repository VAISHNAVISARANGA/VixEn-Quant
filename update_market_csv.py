import yfinance as yf
import pandas as pd
import os

def safe_update(ticker, filename):
    try:
        df = yf.download(ticker, period="35y", progress=False)
        if df.empty:
            return
        tmp = filename + ".tmp"
        df.to_csv(tmp)
        os.replace(tmp, filename)
    except:
        pass

if __name__ == "__main__":
    safe_update("SPY", "spy.csv")
    safe_update("^VIX", "vix.csv")
