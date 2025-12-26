"""
Model definitions and training functions for CO2 emissions prediction.
This module implements four regression models with hyperparameters optimized for small dataset performance (n=25 training samples).
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# OLS
def train_ols(X_train, y_train, alpha=0.1):
    """
    Train Ordinary Least Squares regression.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Ridge
def train_ridge(X_train, y_train, alpha=0.1):
    """
    Train Ridge regression (L2 REGULARIZATION).
    """
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


# Lasso
def train_lasso(X_train, y_train, alpha=0.1):
    """
    Train Lasso regression model (L1 REGULARIZATION).
    """
    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    return model


# Random Forest
def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest regressor with conservative hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# XGBoost
def train_xgboost(X_train, y_train, random_state=42):
    """
    Train XGBoost regressor with regularization.
    """
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        min_child_weight=3,
        random_state=random_state,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    
    return model


# Generate predictions using trained model
def predict(model, X_test):
    return model.predict(X_test)