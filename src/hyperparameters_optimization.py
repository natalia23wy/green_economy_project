"""
Hyperparameter Optimization Module
Optimizes all models and stores best parameters as global variables.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from xgboost import XGBRegressor


def r2_scorer(y_true, y_pred):
    # R² scorer for GridSearchCV.
    return r2_score(y_true, y_pred)


def optimize_linear_models(X_train_scaled, y_train, X_val_scaled, y_val):
    """Optimize linear models and return best parameters."""
    print("Optimizing linear models...")
    
    # OLS
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)
    ols_r2 = ols_model.score(X_val_scaled, y_val)
    print(f"✓ OLS R²: {ols_r2:.4f}")
    
    # Ridge
    ridge_alphas = np.logspace(-5, 2, 15)
    best_ridge_r2 = -np.inf
    best_ridge_alpha = 0.1
    
    for alpha in ridge_alphas:
        ridge_model = Ridge(alpha=alpha, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        val_r2 = ridge_model.score(X_val_scaled, y_val)
        if val_r2 > best_ridge_r2:
            best_ridge_r2 = val_r2
            best_ridge_alpha = alpha
    
    print(f"✓ Ridge alpha: {best_ridge_alpha:.6f} (R²: {best_ridge_r2:.4f})")
    
    # Lasso
    lasso_alphas = np.logspace(-5, 2, 12)
    best_lasso_r2 = -np.inf
    best_lasso_alpha = 0.1
    
    for alpha in lasso_alphas:
        try:
            lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            lasso_model.fit(X_train_scaled, y_train)
            val_r2 = lasso_model.score(X_val_scaled, y_val)
            if val_r2 > best_lasso_r2:
                best_lasso_r2 = val_r2
                best_lasso_alpha = alpha
        except:
            continue
    
    print(f"✓ Lasso alpha: {best_lasso_alpha:.6f} (R²: {best_lasso_r2:.4f})")
    
    # Store directly in global variables
    global RIDGE_ALPHA, LASSO_ALPHA
    RIDGE_ALPHA = best_ridge_alpha
    LASSO_ALPHA = best_lasso_alpha
    
    return {
        'ols': {'val_r2': ols_r2},
        'ridge': {'alpha': best_ridge_alpha, 'val_r2': best_ridge_r2},
        'lasso': {'alpha': best_lasso_alpha, 'val_r2': best_lasso_r2}
    }


def optimize_tree_models(X_train, y_train):
    """Optimize tree models with GridSearchCV and return best parameters."""
    print("Optimizing tree models...")
    
    # Random Forest
    rf_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        rf_grid,
        scoring=make_scorer(r2_scorer),
        cv=3,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    rf_search.fit(X_train, y_train)
    
    cv_results = pd.DataFrame(rf_search.cv_results_)
    cv_results['overfitting_gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']
    best_idx = cv_results.sort_values(['mean_test_score', 'overfitting_gap'], 
                                   ascending=[False, True]).index[0]
    best_rf_params = cv_results.loc[best_idx, 'params']
    best_rf_score = cv_results.loc[best_idx, 'mean_test_score']
    
    print(f"✓ Random Forest CV R²: {best_rf_score:.4f}")
    
    # XGBoost
    xgb_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb_search = GridSearchCV(
        XGBRegressor(random_state=42),
        xgb_grid,
        scoring=make_scorer(r2_scorer),
        cv=3,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    xgb_search.fit(X_train, y_train)
    
    cv_results = pd.DataFrame(xgb_search.cv_results_)
    cv_results['overfitting_gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']
    best_idx = cv_results.sort_values(['mean_test_score', 'overfitting_gap'], 
                                   ascending=[False, True]).index[0]
    best_xgb_params = cv_results.loc[best_idx, 'params']
    best_xgb_score = cv_results.loc[best_idx, 'mean_test_score']
    
    print(f"✓ XGBoost CV R²: {best_xgb_score:.4f}")
    
    # Gradient Boosting
    gb_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_grid,
        scoring=make_scorer(r2_scorer),
        cv=3,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    gb_search.fit(X_train, y_train)
    
    cv_results = pd.DataFrame(gb_search.cv_results_)
    cv_results['overfitting_gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']
    best_idx = cv_results.sort_values(['mean_test_score', 'overfitting_gap'], 
                                   ascending=[False, True]).index[0]
    best_gb_params = cv_results.loc[best_idx, 'params']
    best_gb_score = cv_results.loc[best_idx, 'mean_test_score']
    
    print(f"✓ Gradient Boosting CV R²: {best_gb_score:.4f}")
    
    return {
        'random_forest': {'best_params': best_rf_params, 'best_score': best_rf_score},
        'xgboost': {'best_params': best_xgb_params, 'best_score': best_xgb_score},
        'gradient_boosting': {'best_params': best_gb_params, 'best_score': best_gb_score}
    }


def optimize_all_models(X_train_scaled, y_train, X_val_scaled, y_val, run_tree_gridsearch=True):
    """
    Optimize all models and store best parameters in global variables.
    
    Args:
        X_train_scaled, y_train, X_val_scaled, y_val: Scaled data (consistent with models.py)
        run_tree_gridsearch: If False, skip tree models (much faster)
    
    Returns:
        Dictionary with best parameters for all models
    """
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Optimize linear models first (using scaled data)
    linear_results = optimize_linear_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Optimize tree models (using scaled data)
    if run_tree_gridsearch:
        print("\nRunning GridSearchCV for tree models...")
        tree_results = optimize_tree_models(X_train_scaled, y_train)
    else:
        print("\nSkipping tree models (use run_tree_gridsearch=True to enable)")
        tree_results = {
            'random_forest': {'best_params': {'n_estimators': 25, 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None}, 'best_score': -2.0},
            'xgboost': {'best_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6, 'colsample_bytree': 0.6}, 'best_score': -2.0},
            'gradient_boosting': {'best_params': {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 2, 'subsample': 1.0}, 'best_score': -2.0}
        }
    
    # Store tree parameters in global variables
    global RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, GRADIENT_BOOSTING_PARAMS
    RANDOM_FOREST_PARAMS = tree_results['random_forest']['best_params']
    XGBOOST_PARAMS = tree_results['xgboost']['best_params']
    GRADIENT_BOOSTING_PARAMS = tree_results['gradient_boosting']['best_params']
    
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS FOUND")
    print("=" * 60)
    print(f"Ridge: alpha={RIDGE_ALPHA:.6f}")
    print(f"Lasso: alpha={LASSO_ALPHA:.6f}")
    print(f"Random Forest: {RANDOM_FOREST_PARAMS}")
    print(f"XGBoost: {XGBOOST_PARAMS}")
    print(f"Gradient Boosting: {GRADIENT_BOOSTING_PARAMS}")
    
    # Find best model
    linear_scores = {
        'OLS': linear_results['ols']['val_r2'],
        'Ridge': linear_results['ridge']['val_r2'],
        'Lasso': linear_results['lasso']['val_r2']
    }
    
    best_linear = max(linear_scores, key=linear_scores.get)
    print(f"\nBest Linear: {best_linear} (R² = {linear_scores[best_linear]:.4f})")
    
    if run_tree_gridsearch:
        tree_scores = {
            'Random Forest': tree_results['random_forest']['best_score'],
            'XGBoost': tree_results['xgboost']['best_score'],
            'Gradient Boosting': tree_results['gradient_boosting']['best_score']
        }
        best_tree = max(tree_scores, key=tree_scores.get)
        print(f"Best Tree: {best_tree} (CV R² = {tree_scores[best_tree]:.4f})")
        
        if linear_scores[best_linear] > tree_scores[best_tree]:
            print(f"OVERALL WINNER: {best_linear}")
        else:
            print(f"OVERALL WINNER: {best_tree}")
    
    print("\n" + "=" * 60)
    print("PARAMETERS STORED IN GLOBAL VARIABLES")
    print("=" * 60)
    
    # Return best parameters
    best_params = {
        'ols': {},
        'ridge': {'alpha': RIDGE_ALPHA},
        'lasso': {'alpha': LASSO_ALPHA},
        'random_forest': RANDOM_FOREST_PARAMS,
        'xgboost': XGBOOST_PARAMS,
        'gradient_boosting': GRADIENT_BOOSTING_PARAMS
    }
    
    return best_params


# Global variables (will be filled after optimization)
RIDGE_ALPHA = 0.031623  # updated after optimization
LASSO_ALPHA = 100.0
RANDOM_FOREST_PARAMS = {'n_estimators': 25, 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None}
XGBOOST_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6, 'colsample_bytree': 0.6}
GRADIENT_BOOSTING_PARAMS = {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 2, 'subsample': 1.0}


if __name__ == "__main__":
    # Test the module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import build_dataset_france, train_val_test_split, scale_features
    
    print("Testing hyperparameter optimization...")
    
    # Load data
    df = build_dataset_france()
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = train_val_test_split(df)
    
    # Add year and scale
    X_train_with_year = X_train.copy()
    X_val_with_year = X_val.copy()
    X_test_with_year = X_test.copy()
    X_train_with_year['year_numeric'] = train_df['year']
    X_val_with_year['year_numeric'] = val_df['year']
    X_test_with_year['year_numeric'] = test_df['year']
    
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train_with_year, X_val_with_year, X_test_with_year)
    
    # Run optimization
    best_params = optimize_all_models(X_train_s, y_train, X_val_s, y_val, run_tree_gridsearch=True)
    
    print(f"\n✓ Optimization complete! Global variables updated.")
