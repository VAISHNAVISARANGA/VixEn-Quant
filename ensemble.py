import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def select_top_features(X_train, y_train, target_col='Target', n_features=20):
    df = pd.concat([X_train, y_train], axis=1)
    corr_with_target = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(n_features).index.tolist()
    
    print(f"Selected top {len(top_features)} features:")
    print(top_features)
    print("\nCorrelations:")
    print(corr_with_target.head(n_features))
    
    return top_features


def objective_xgb(trial, X_train, X_test, y_train, y_test, top_features):
    """
    Optuna objective function for XGBoost hyperparameter optimization.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train[top_features], y_train, eval_set=[(X_test[top_features], y_test)], verbose=False)

    y_pred = model.predict(X_test[top_features])
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def objective_lgb(trial, X_train, X_test, y_train, y_test, top_features):
    """
    Optuna objective function for LightGBM hyperparameter optimization.
    """

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train[top_features], y_train, eval_set=[(X_test[top_features], y_test)])
    
    y_pred = model.predict(X_test[top_features])
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def optimize_hyperparameters(X_train, y_train, X_test, y_test,top_features, model_type='xgboost', n_trials=50):
    """
    Use Optuna to find the best hyperparameters.
    """
    print(f"\nOptimizing {model_type.upper()} hyperparameters with {len(top_features)} features")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    if model_type.lower() == 'xgboost':
        objective_func = objective_xgb
    else:
        objective_func = objective_lgb
    
    print(f"Running optimization with {n_trials} trials...")
    study.optimize(
        lambda trial: objective_func(trial, X_train, X_test, y_train, y_test, top_features),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_accuracy = study.best_value
    
    print(f"\n--- Optimization Results ({model_type.upper()}) ---")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params, study


def train_xgboost_with_params(X_train, X_test, y_train, y_test, params, top_features):
 
   
    print(f"\nTraining XGBoost with optimized parameters")
  
    # Avoid passing duplicate random_state/verbosity if present in params
    params_clean = params.copy()
    params_clean.pop('random_state', None)
    params_clean.pop('verbosity', None)
    model = xgb.XGBClassifier(**params_clean, random_state=42, verbosity=0)
    print("Training XGBoost model...")
    model.fit(X_train[top_features], y_train, eval_set=[(X_test[top_features], y_test)], verbose=False)

    y_pred = model.predict(X_test[top_features])
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- XGBoost Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, X_test[top_features], y_test


def train_lightgbm_with_params(X_train,X_test, y_train, y_test, params, top_features):
    

    print(f"\nTraining LightGBM with optimized parameters")

    params_clean = params.copy()
    params_clean.pop('random_state', None)
    params_clean.pop('verbose', None)
    model = lgb.LGBMClassifier(**params_clean, random_state=42, verbose=-1)
    print("Training LightGBM model...")
    model.fit(X_train[top_features], y_train, eval_set=[(X_test[top_features], y_test)])

    y_pred = model.predict(X_test[top_features])
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- LightGBM Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, X_test[top_features], y_test


# ... (keep your existing imports and objective functions)

class EnsembleModel:
    def __init__(self, xgb_model=None, lgb_model=None, weights=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        # Default weights
        self.weights = weights if weights else {'xgb': 0.4, 'lgb': 0.6}
    
    def predict(self, X_test):
        """Standard prediction using pre-trained models"""
        xgb_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_test)[:, 1]
        
        ensemble_pred_proba = (
            self.weights['xgb'] * xgb_pred +
            self.weights['lgb'] * lgb_pred 
        )
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        return ensemble_pred, ensemble_pred_proba

    def walk_forward_predict(self, X, y, tscv, xgb_params, lgb_params, top_features):
        """
        Modified to perform chronological 'stitching' of unseen predictions.
        This handles the 25+ years of data in sequence without 'cheating'.
        """
        all_preds = []
        all_indices = []

        print(f"Starting Walk-Forward Validation across {tscv.n_splits} folds...")

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Split data based on current fold
            X_train_fold, X_test_fold = X.iloc[train_idx][top_features], X.iloc[test_idx][top_features]
            y_train_fold = y.iloc[train_idx]

            # 1. Initialize models with your optimized params
            fold_xgb = xgb.XGBClassifier(**xgb_params, random_state=42, verbosity=0)
            fold_lgb = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)

            # 2. Train on the historical 'train' block
            fold_xgb.fit(X_train_fold, y_train_fold)
            fold_lgb.fit(X_train_fold, y_train_fold)

            # 3. Predict the 'unseen' future block
            xgb_p = fold_xgb.predict_proba(X_test_fold)[:, 1]
            lgb_p = fold_lgb.predict_proba(X_test_fold)[:, 1]
            
            # 4. Ensemble the predictions for this fold
            fold_ensemble_proba = (self.weights['xgb'] * xgb_p) + (self.weights['lgb'] * lgb_p)
            fold_ensemble_pred = (fold_ensemble_proba > 0.5).astype(int)

            # Store results
            all_preds.append(fold_ensemble_pred)
            all_indices.append(X.index[test_idx])
            
            print(f"Fold {i+1} complete. Processed up to {X.index[test_idx][-1].year}")

        # Stitch everything together into a chronological Series
        stitched_signals = pd.Series(
            np.concatenate(all_preds), 
            index=pd.to_datetime(np.concatenate(all_indices))
        )
        return stitched_signals
    
    def set_weights(self, weights):
        self.weights = weights





