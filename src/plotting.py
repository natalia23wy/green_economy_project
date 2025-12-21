# Plotting utilities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_feature_importance_rf(rf_model, feature_names):
    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    return rf_importance


def compute_feature_importance_xgb(xgb_model, feature_names):
    xgb_importance = pd.Series(
        xgb_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    return xgb_importance


def plot_rf_feature_importance(rf_importance, output_path="results/rf_feature_importance.png"):
    plt.figure()
    plt.barh(rf_importance.index, rf_importance.values)
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_xgb_feature_importance(xgb_importance, output_path="results/xgb_feature_importance.png"):
    xgb_importance.plot(kind="bar")
    plt.title("XGBoost Feature Importance")
    plt.ylabel("Importance score")
    plt.xlabel("Features")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance_comparison(
    rf_importance,
    xgb_importance,
    output_path="results/rf_vs_xgb_feature_importance.png"
):
    # ensure same features order
    features = rf_importance.index
    rf_vals = rf_importance.values
    xgb_vals = xgb_importance[features].values

    x = np.arange(len(features))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, rf_vals, width, label="Random Forest")
    plt.bar(x + width/2, xgb_vals, width, label="XGBoost")

    plt.xticks(x, features, rotation=45)
    plt.ylabel("Feature importance")
    plt.title("Feature Importance Comparison: Random Forest vs XGBoost")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_standardized_trends(
    df,
    cols_to_plot,
    output_path="results/standardized_trends_co2_macro_france.png"
):
    # z-score: (x - mean) / std
    z = df[cols_to_plot].copy()
    z = (z - z.mean()) / z.std()

    plt.figure()
    for col in cols_to_plot:
        plt.plot(df["year"], z[col], label=col)

    plt.xlabel("Year")
    plt.ylabel("Standardized value (z-score)")
    plt.title("Standardized trends: CO₂ and explanatory variables (France)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_actual_vs_predicted_full_rf_xgb(
    df,
    rf_pred_full,
    xgb_pred_full,
    train_end_year=2020,
    output_path="results/actual_vs_predicted_rf_xgb.png",
):
    plt.figure(figsize=(10, 6))

    # Actual CO2
    plt.plot(
        df["year"],
        df["co2_million_tonnes"],
        label="Actual CO₂",
        color="black",
        linewidth=2,
    )

    # Predicted CO2
    plt.plot(
        df["year"],
        rf_pred_full,
        linestyle="--",
        color="blue",
        label="RF prediction",
    )
    plt.plot(
        df["year"],
        xgb_pred_full,
        linestyle="--",
        color="orange",
        label="XGB prediction",
    )

    # Train / test split line
    plt.axvline(
        x=train_end_year,
        color="gray",
        linestyle=":",
        label="Train / Test split",
    )

    plt.xlabel("Year")
    plt.ylabel("CO₂ emissions (million tonnes)")
    plt.title("Actual vs Predicted CO₂ Emissions (France)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()