# Model definitions and training functions

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Train a linear regression (OLS) model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Train a Random Forest regressor
def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        max_depth=None,
    )
    model.fit(X_train, y_train)
    return model


# Generate predictions using trained model
def predict(model, X_test):
    return model.predict(X_test)