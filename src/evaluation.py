# Evaluation metrics for regression models

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_regression(y_true, y_pred):
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R-squared
    r2 = r2_score(y_true, y_pred)

    return {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
}