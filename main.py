if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    import joblib # For saving the production model

    # Import your custom modules
    from data_manager import fetch_market_data
    from feature_engine2 import create_features
    from preprocessing import split_data, normalize_features
    from vix_filter import apply_vix_strategy
    from ensemble import (
        EnsembleModel, 
        optimize_hyperparameters, 
        train_xgboost_with_params, 
        train_lightgbm_with_params, 
        select_top_features
    )
    from evaluator import calculate_sharpe, calculate_rolling_sharpe, calculate_max_drawdown, calculate_CAGR

    # --- 1. DATA PREPARATION ---
    raw_data = fetch_market_data()
    featured_data = create_features(raw_data)
    print("✓ Data Ingestion and Feature Engineering Complete")
    
    # Split for parameter optimization (using the most recent window)
    X_train, X_test, y_train, y_test = split_data(featured_data)
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    # Lock in Top 20 features
    top_features = select_top_features(X_train_norm, y_train, n_features=20)

    # --- 2. HYPERPARAMETER OPTIMIZATION ---
    print("\n" + "="*60 + "\nSTEP 1: OPTIMIZING MODEL HYPERPARAMETERS\n" + "="*60)
    best_params_xgb, _ = optimize_hyperparameters(X_train_norm, y_train, X_test_norm, y_test, top_features, model_type='xgboost')
    best_params_lgb, _ = optimize_hyperparameters(X_train_norm, y_train, X_test_norm, y_test, top_features, model_type='lightgbm')

    # --- 3. PRODUCTION MODEL (The 'Predict' Path) ---
    print("\n" + "="*60 + "\nSTEP 2: TRAINING PRODUCTION ENSEMBLE (FOR UI)\n" + "="*60)
    
    # Train individual models on the training set
    model_xgb, _, _ = train_xgboost_with_params(X_train_norm, X_test_norm, y_train, y_test, best_params_xgb, top_features)
    model_lgb, _, _ = train_lightgbm_with_params(X_train_norm, X_test_norm, y_train, y_test, best_params_lgb, top_features)
    
    # Initialize Ensemble
    ensemble = EnsembleModel(model_xgb, model_lgb, weights={'xgb': 0.4, 'lgb': 0.6})
    
    # Predict using the standard .predict() method
    y_pred_std, _ = ensemble.predict(X_test_norm[top_features])
    
    # Apply VIX Filter to standard prediction
    vix_mask_std = raw_data.loc[X_test_norm.index, 'VIX_Safe']
    y_pred_std_vix = apply_vix_strategy(vix_mask_std, y_pred_std)
    
    acc_std = accuracy_score(y_test, y_pred_std_vix)
    print(f"Production Model Accuracy (with VIX Filter): {acc_std:.4f}")
    print(classification_report(y_test, y_pred_std_vix))

    # --- 4. BACKTEST (The 'Walk-Forward' Path) ---
    print("\n" + "="*60 + "\nSTEP 3: RUNNING 35-YEAR WALK-FORWARD BACKTEST\n" + "="*60)
    
    X_full_norm = pd.concat([X_train_norm, X_test_norm])
    y_full = featured_data['Target']
    tscv = TimeSeriesSplit(n_splits=8)

    # Generate the 'Honest' stitched signals
    honest_signals = ensemble.walk_forward_predict(
        X=X_full_norm, 
        y=y_full, 
        tscv=tscv, 
        xgb_params=best_params_xgb, 
        lgb_params=best_params_lgb, 
        top_features=top_features
    )

    # Apply VIX Filter to stitched signals
    vix_mask_full = raw_data.loc[honest_signals.index, 'VIX_Safe']
    final_backtest_signals = honest_signals * vix_mask_full

    # --- 5. FINAL EVALUATION & PLOTTING ---
    print("\n" + "="*60 + "\nSTEP 4: FINAL PERFORMANCE EVALUATION\n" + "="*60)
    
    mkt_returns = raw_data.loc[honest_signals.index, 'Close'].pct_change().fillna(0)
    strat_returns = final_backtest_signals * mkt_returns

    # Plot the 35-year Rolling Sharpe
    strat_roll, mkt_roll = calculate_rolling_sharpe(strat_returns, mkt_returns, window=252, plot=True)
    
    print(f"Average Rolling Strategy Sharpe: {strat_roll.mean():.4f}")
    print(f"Average Rolling Market Sharpe: {mkt_roll.mean():.4f}")

    # --- 4. FINAL PERFORMANCE METRICS (Using your specific function) ---
    print("\n" + "="*60 + "\nFINAL PERFORMANCE SUMMARY\n" + "="*60)

    # To use your calculate_sharpe, we need a dataframe with a 'Close' column
    # We create a temporary DF for the backtest period
    backtest_period_df = raw_data.loc[honest_signals.index]

    # Calling your function exactly as defined
    ann_sharpe, strat_returns, daily_sharpe = calculate_sharpe(
        X_test=backtest_period_df, 
        y_pred_series=final_backtest_signals
    )
    mdd_value, _ = calculate_max_drawdown(strat_returns)

    cagr = calculate_CAGR(strat_returns)


    print(f"Daily Sharpe Ratio:      {daily_sharpe:.6f}")
    print(f"Annualized Sharpe Ratio: {ann_sharpe:.4f}")
    print(f"Annualized Return (CAGR): {cagr * 100:.2f}%")
    print(f"Maximum Drawdown:        {mdd_value * 100:.2f}%")
    print("============================================================")

# --- STEP 6: SAVE PRODUCTION ASSETS ---
    print("\n" + "="*60 + "\nSAVING ASSETS FOR DEPLOYMENT\n" + "="*60)
    
    # Bundle everything into a production dictionary
    production_assets = {
        'model': ensemble,
        'top_features': top_features,
        'params': {
            'xgb': best_params_xgb,
            'lgb': best_params_lgb
        },
        'stats': {
            'annualized_sharpe': ann_sharpe,
            'max_drawdown': mdd_value,
            'CAGR': cagr
        }
    }

    # Save to a file
    model_filename = "trading_ensemble_prod.pkl"
    joblib.dump(production_assets, model_filename)
    
    print(f"✓ Production model and metadata saved to {model_filename}")
    print("✓ Ready for Streamlit deployment.")
