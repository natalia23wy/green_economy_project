"""
Model definitions and training functions for CO2 emissions prediction.
This module implements four regression models with hyperparameters optimized for small dataset performance (n=25 training samples).
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from src.hyperparameters_optimization import (
    RIDGE_ALPHA, LASSO_ALPHA, RANDOM_FOREST_PARAMS, 
    XGBOOST_PARAMS, GRADIENT_BOOSTING_PARAMS
)

# OLS
def train_ols(X_train, y_train):
    """
    Train Ordinary Least Squares regression.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Ridge
def train_ridge(X_train, y_train, alpha=None):
    """
    Train Ridge regression (L2 REGULARIZATION).
    """
    if alpha is None:
        alpha = RIDGE_ALPHA
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


# Lasso
def train_lasso(X_train, y_train, alpha=None):
    """
    Train Lasso regression model (L1 REGULARIZATION).
    """
    if alpha is None:
        alpha = LASSO_ALPHA
    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    return model


# Random Forest
def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest regressor with optimized hyperparameters.
    """
    model = RandomForestRegressor(
n_estimators=RANDOM_FOREST_PARAMS['n_estimators'],
        max_depth=RANDOM_FOREST_PARAMS['max_depth'],
        min_samples_split=RANDOM_FOREST_PARAMS['min_samples_split'],
        min_samples_leaf=RANDOM_FOREST_PARAMS['min_samples_leaf'],
        max_features=RANDOM_FOREST_PARAMS['max_features'],
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# XGBoost
def train_xgboost(X_train, y_train, random_state=42):
    """
    Train XGBoost regressor with optimized hyperparameters.
    """
    model = XGBRegressor(
        n_estimators=XGBOOST_PARAMS['n_estimators'],
        learning_rate=XGBOOST_PARAMS['learning_rate'],
        max_depth=XGBOOST_PARAMS['max_depth'],
        subsample=XGBOOST_PARAMS['subsample'],
        colsample_bytree=XGBOOST_PARAMS['colsample_bytree'],
        random_state=random_state,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    
    return model


# Gradient Boosting
def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Train Gradient Boosting regressor with conservative parameters, more robust than
    XGBoost for small datasets).
    """
    model = GradientBoostingRegressor(
        n_estimators=GRADIENT_BOOSTING_PARAMS['n_estimators'],
        learning_rate=GRADIENT_BOOSTING_PARAMS['learning_rate'],
        max_depth=GRADIENT_BOOSTING_PARAMS['max_depth'],
        subsample=GRADIENT_BOOSTING_PARAMS['subsample'],
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# Generate predictions using trained model
def predict(model, X_test):
    return model.predict(X_test)