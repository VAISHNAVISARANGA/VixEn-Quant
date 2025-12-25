import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_sharpe(X_test, y_pred_series):
    """Calculate annualized Sharpe and return strategy returns.

    This function is robust to the type of `y_pred_series` (numpy array or pandas Series).
    It aligns the predictions to the tail of `X_test` by index, drops NaNs, and handles
    zero or NaN standard deviation cases.
    """
    # Ensure predictions are a pandas Series aligned to the tail of X_test
    if not isinstance(y_pred_series, pd.Series):
        y_pred_series = pd.Series(y_pred_series, index=X_test.index[-len(y_pred_series):])
    else:
        # if indices don't match, reindex to the tail
        if not y_pred_series.index.equals(X_test.index[-len(y_pred_series):]):
            y_pred_series = pd.Series(y_pred_series.values, index=X_test.index[-len(y_pred_series):])

    # compute returns for the same index range
    test_returns = X_test['Close'].pct_change().reindex(y_pred_series.index)
    strategy_returns = (y_pred_series * test_returns).dropna()

    if strategy_returns.empty:
        return np.nan, strategy_returns, np.nan

    mean_return = strategy_returns.mean()
    return_std = strategy_returns.std()

    if return_std == 0 or np.isnan(return_std):
        return np.nan, strategy_returns, np.nan

    daily_sharpe = mean_return / return_std
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    return annualized_sharpe, strategy_returns, daily_sharpe


def calculate_rolling_sharpe(strategy_returns, market_returns, window=252, plot=True):
    """Calculates and optionally plots rolling Annualized Sharpe Ratio for strategy and market.

    Returns both rolling series: (strat_rolling, market_rolling).
    """
    def rolling_sharpe(rets):
        rolling_mu = rets.rolling(window=window).mean()
        rolling_sigma = rets.rolling(window=window).std()
        # protect against division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = (rolling_mu / rolling_sigma) * np.sqrt(252)
        return vals

    strat_rolling = rolling_sharpe(strategy_returns)
    market_rolling = rolling_sharpe(market_returns)

    return strat_rolling, market_rolling

def calculate_max_drawdown(strategy_returns):
    """
    Calculates the maximum peak-to-trough decline of the strategy.
    """
    # 1. Calculate the cumulative equity curve
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # 2. Calculate the running maximum (the highest point reached so far)
    running_max = cumulative_returns.cummax()
    
    # 3. Calculate the drawdown (percent drop from the peak)
    drawdown = (cumulative_returns - running_max) / running_max
    
    # 4. Find the worst (minimum) drawdown value
    max_drawdown = drawdown.min()
    
    return max_drawdown, drawdown

def calculate_CAGR(strategy_returns):
    """
    Calculates the Compound Annual Growth Rate (CAGR) of the strategy.
    """
    total_periods = len(strategy_returns) / 252  # Assuming 252 trading days in a year
    total_return = (1 + strategy_returns).prod() - 1
    
    if total_periods <= 0:
        return np.nan
    
    cagr = (1 + total_return) ** (1 / total_periods) - 1
    return cagr