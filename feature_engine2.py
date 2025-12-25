
import numpy as np
import pandas as pd

def create_features(df):

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # ---------------- Trend ----------------
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ---------------- Momentum ----------------
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['ROC'] = df['Close'].pct_change(10)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Stochastic Oscillator
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)

    # ---------------- Volatility ----------------
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close'].shift(1)
    df['Volatility'] = df['Close'].rolling(21).std().shift(1)

    # ATR

    # Bollinger Bands
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20

    # ---------------- Volume ----------------
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()

    # On-Balance Volume (OBV)
    direction = (df['Close'].diff() > 0).astype(int) - (df['Close'].diff() < 0).astype(int)
    obv_series = (df['Volume'] * direction).cumsum()
    df['OBV'] = obv_series.shift(1)

    # Compute Volume_Spike using numpy arrays to avoid pandas alignment issues
    vol = df['Volume'].to_numpy()
    vol_ma = df['Volume_MA20'].to_numpy()

    # where rolling window yields NaN for vol_ma, treat as False (0)
    vol_spike = (vol_ma != np.nan) & (vol > 1.5 * vol_ma)
    df['Volume_Spike'] = pd.Series(vol_spike.astype(int), index=df.index).shift(1)


    # ---------------- Price-based ----------------
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift())
    df['Gap'] = df['Open'] - df['Close'].shift()
    df['High_Low_Spread'] = df['High'] - df['Low']


    # ---------------- Lag Features ----------------
    df['Close_lag1'] = df['Close'].shift(1)
    df['Return_lag1'] = df['Return'].shift(1)
    df['Volume_lag1'] = df['Volume'].shift(1)
   #-------------------Trend Strength----------------
    
    # Price slope (trend strength)
    df['Price_Slope_20'] = (
        df['Close']
        .rolling(20)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    ).shift(1)

    # ---------------- Target ----------------
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()
