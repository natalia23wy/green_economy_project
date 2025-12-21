from src.data_loader import build_dataset_france, train_test_split
from src.models import train_linear_regression, train_random_forest, train_xgboost, predict
from src.evaluation import evaluate_regression
from src.plotting import (
    compute_feature_importance_rf,
    compute_feature_importance_xgb,
    plot_feature_importance_comparison,
    plot_standardized_trends,
    plot_actual_vs_predicted_full_rf_xgb,
)


def main():
    # 1) Build merged dataset for France
    df = build_dataset_france()

    # 2) Train/test split (time-based)
    X_train, X_test, y_train, y_test, train_df, test_df = train_test_split(df)

    # 3) OLS baseline
    ols_model = train_linear_regression(X_train, y_train)
    ols_pred = predict(ols_model, X_test)
    ols_metrics = evaluate_regression(y_test, ols_pred)


    # 4) Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = predict(rf_model, X_test)
    rf_metrics = evaluate_regression(y_test, rf_pred)


    # 5) XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred = predict(xgb_model, X_test)
    xgb_metrics = evaluate_regression(y_test, xgb_pred)


    # 6) Feature importance values and plot
    rf_importance = compute_feature_importance_rf(rf_model, X_train.columns)
    xgb_importance = compute_feature_importance_xgb(xgb_model, X_train.columns)

    print("Random Forest feature importance:")
    print(rf_importance)
    print("XGBoost feature importance:")
    print(xgb_importance)

    plot_feature_importance_comparison(
        rf_importance,
        xgb_importance,
        "results/rf_vs_xgb_feature_importance.png"
    )


    # 7) Comparison plot: Actual vs Predicted CO2 emissions
    feature_cols = X_train.columns

    plot_actual_vs_predicted_full_rf_xgb(
        df=df,
        rf_pred_full=predict(rf_model, df[feature_cols]),
        xgb_pred_full=predict(xgb_model, df[feature_cols]),
        train_end_year=2020,
        output_path="results/actual_vs_predicted_rf_xgb.png",
    )

    # 8) Standardized trends plot
    cols_to_plot = [
        "co2_million_tonnes",
        "gdp_real_constant_usd",
        "unemployment_rate",
        "inflation_cpi",
    ]
    plot_standardized_trends(df, cols_to_plot, "results/standardized_trends_co2_macro_france.png")

    # 9) Print results
    print("Test years:", list(test_df["year"]))
    print("OLS metrics:", ols_metrics)
    print("RF metrics:", rf_metrics)
    print("XGB metrics:", xgb_metrics)


if __name__ == "__main__":
    main()
    
    # Optional sanity checks (keep commented)
    # print("Train years:", train_df["year"].min(), "-", train_df["year"].max())
    # print("Test years:", test_df["year"].min(), "-", test_df["year"].max())
    # print("Train size:", X_train.shape)
    # print("Test size:", X_test.shape)
    
    # Optional: see actual vs predicted (keep commented)
    # print("y_test:", [float(v) for v in y_test.values])
    # print("OLS y_pred:", [float(v) for v in ols_pred])
    # print("RF  y_pred:", [float(v) for v in rf_pred])