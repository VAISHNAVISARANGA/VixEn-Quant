from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler
def split_data(df):
    X=df.drop(columns=['Target'])
    y=df['Target']
    tscv = TimeSeriesSplit(n_splits=8)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return  X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    # determine feature columns (handle case where 'VIX_Safe' may not exist)
    if 'VIX_Safe' in X_train.columns:
        feature_cols = X_train.columns.drop('VIX_Safe')
    else:
        feature_cols = X_train.columns

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train[feature_cols])
    test_scaled = scaler.transform(X_test[feature_cols])

    train = pd.DataFrame(train_scaled, columns=feature_cols, index=X_train.index)
    test = pd.DataFrame(test_scaled, columns=feature_cols, index=X_test.index)

    # attach VIX_Safe back if present (preserve the original index)
    if 'VIX_Safe' in X_train.columns:
        train['VIX_Safe'] = X_train['VIX_Safe'].reindex(train.index)
    if 'VIX_Safe' in X_test.columns:
        test['VIX_Safe'] = X_test['VIX_Safe'].reindex(test.index)

    return train, test


