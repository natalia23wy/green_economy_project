from src.data_loader import build_dataset_france, train_test_split, train_val_test_split
from src.models import train_linear_regression, train_random_forest, train_xgboost, predict
from src.evaluation import evaluate_regression, regression_diagnostics, mape
from src.plotting import (
    compute_feature_importance_rf,
    compute_feature_importance_xgb,
    plot_feature_importance_comparison,
    plot_standardized_trends,
    plot_actual_vs_predicted_co2,
    plot_actual_vs_predicted_co2_train_val_test,
)


def main():
    # =========================================================================
    # 1. LOAD AND PREPARE DATAS
    # =========================================================================

    df = build_dataset_france()
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = train_val_test_split(df)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # =========================================================================
    # 2. TRAIN MODELS
    # =========================================================================

    ols_model = train_linear_regression(X_train_s, y_train, alpha=0.0)
    rf_model = train_random_forest(X_train_s, y_train)
    xgb_model = train_xgboost(X_train_s, y_train)


    ols_pred = predict(ols_model, X_test)
    ols_metrics = evaluate_regression(y_test, ols_pred)


    

        # Predictions
    rf_pred = predict(rf_model, X_test)
    rf_train_pred = predict(rf_model, X_train)

        # Test metrics (MAE / RMSE / R2)
    rf_metrics = evaluate_regression(y_test, rf_pred)

        # Train vs test diagnostics (R2 / RMSE / MAPE)
    rf_diag = regression_diagnostics(
        y_train, rf_train_pred,
        y_test, rf_pred
    )

        # MAPE on test and accuracy proxy
    rf_test_mape = mape(y_test, rf_pred)
    rf_test_accuracy = 1 - rf_test_mape


    # 5) XGBoost     
        # Predictions
    xgb_pred = predict(xgb_model, X_test)
    xgb_train_pred = predict(xgb_model, X_train)

        # Test metrics (MAE / RMSE / R2)
    xgb_metrics = evaluate_regression(y_test, xgb_pred)

        # Train vs test diagnostics (R2/RMSE/MAPE)
    xgb_diag = regression_diagnostics(
        y_train, xgb_train_pred,
        y_test, xgb_pred
    )

        # MAPE on test and accuracy proxy
    xgb_test_mape = mape(y_test, xgb_pred)
    xgb_test_accuracy = 1 - xgb_test_mape


    # 6) Feature importance values and plot
    rf_importance = compute_feature_importance_rf(rf_model, X_train.columns)
    xgb_importance = compute_feature_importance_xgb(xgb_model, X_train.columns)

    plot_feature_importance_comparison(
        rf_importance,
        xgb_importance,
        "results/rf_vs_xgb_feature_importance.png"
    )


    # 7) Comparison plot: Actual vs Predicted CO2 emissions
    feature_cols = X_train.columns

    plot_actual_vs_predicted_co2(
        df=df,
        rf_pred_full=predict(rf_model, df[feature_cols]),
        xgb_pred_full=predict(xgb_model, df[feature_cols]),
        train_end_year=2020,
        output_path="results/actual_vs_predicted_co2.png",
    )


    plot_actual_vs_predicted_co2_train_val_test(
        df=df,
        rf_pred_full=predict(rf_model, df[feature_cols]),
        xgb_pred_full=predict(xgb_model, df[feature_cols]),
        train_end_year=2015,
        val_end_year=2020,
        output_path="results/actual_vs_predicted_co2_train_val_test.png",
    )

    # 8) Standardized trends plot
    cols_to_plot = [
        "co2_million_tonnes",
        "gdp_real_constant_usd",
        "unemployment_rate",
        "inflation_cpi",
    ]
    plot_standardized_trends(
        df,
        cols_to_plot,
        output_path="results/standardized_trends_co2_macro_france.png"
    )



        # Validation analysis (optional)
    X_train_v, X_val, X_test_v, y_train_v, y_val, y_test_v, _, val_df, _ = (train_val_test_split(df))

    rf_model_val = train_random_forest(X_train_v, y_train_v)

    rf_val_pred = predict(rf_model_val, X_val)
    rf_test_pred = predict(rf_model_val, X_test_v)

    print("\nValidation vs Test performance (Random Forest)")
    print("Validation years:", list(val_df["year"]))
    print("Validation MAPE:", round(mape(y_val, rf_val_pred), 3))
    print("Test MAPE:", round(mape(y_test_v, rf_test_pred), 3))


    # Validation analysis (optional) — XGBoost
    X_train_v, X_val, X_test_v, y_train_v, y_val, y_test_v, _, val_df, _ = (
        train_val_test_split(df)
    )

    xgb_model_val = train_xgboost(X_train_v, y_train_v)

    xgb_val_pred = predict(xgb_model_val, X_val)
    xgb_test_pred = predict(xgb_model_val, X_test_v)

    print("\nValidation vs Test performance (XGBoost)")
    print("Validation years:", list(val_df["year"]))
    print("Validation MAPE:", round(mape(y_val, xgb_val_pred), 3))
    print("Test MAPE:", round(mape(y_test_v, xgb_test_pred), 3))

    # 9) Print results
    print("\n==================== RESULTS ====================\n")

    print("Test years:", list(test_df["year"]))
    print()

    print("OLS metrics")
    print("  MAE :", int(round(ols_metrics["MAE"])))
    print("  RMSE:", int(round(ols_metrics["RMSE"])))
    print("  R2  :", round(ols_metrics["R2"], 3))
    print()

    print("Random Forest metrics (test)")
    print("  MAE :", int(round(rf_metrics["MAE"])))
    print("  RMSE:", int(round(rf_metrics["RMSE"])))
    print("  R2  :", round(rf_metrics["R2"], 3))
    print("Random Forest diagnostics (train vs test)")
    print("  R2_train  :", round(rf_diag["R2_train"], 3))
    print("  R2_test   :", round(rf_diag["R2_test"], 3))
    print("  RMSE_train:", int(round(rf_diag["RMSE_train"])))
    print("  RMSE_test :", int(round(rf_diag["RMSE_test"])))
    print("  MAPE_train:", round(rf_diag["MAPE_train"], 3))
    print("  MAPE_test :", round(rf_diag["MAPE_test"], 3))
    print("  Accuracy (1 - MAPE) test:", round(rf_test_accuracy, 3))
    print()

    print("XGBoost metrics (test)")
    print("  MAE :", int(round(xgb_metrics["MAE"])))
    print("  RMSE:", int(round(xgb_metrics["RMSE"])))
    print("  R2  :", round(xgb_metrics["R2"], 3))
    print("XGBoost diagnostics (train vs test)")
    print("  R2_train  :", round(xgb_diag["R2_train"], 3))
    print("  R2_test   :", round(xgb_diag["R2_test"], 3))
    print("  RMSE_train:", int(round(xgb_diag["RMSE_train"])))
    print("  RMSE_test :", int(round(xgb_diag["RMSE_test"])))
    print("  MAPE_train:", round(xgb_diag["MAPE_train"], 3))
    print("  MAPE_test :", round(xgb_diag["MAPE_test"], 3))
    print("  Accuracy (1 - MAPE) test:", round(xgb_test_accuracy, 3))
    print()

    print("Feature importance (sorted)")
    print("  Random Forest:")
    for name, val in rf_importance.items():
        print(f"    {name}: {val:.3f}")
    print("  XGBoost:")
    for name, val in xgb_importance.items():
        print(f"    {name}: {val:.3f}")

    print("\n=================================================\n")

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