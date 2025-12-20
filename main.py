from src.data_loader import build_dataset_france, train_test_split
from src.models import train_linear_regression, train_random_forest, predict
from src.evaluation import evaluate_regression


def main():
    # 1) Build merged dataset for France
    df = build_dataset_france()

    # 2) Train/test split (time-based)
    X_train, X_test, y_train, y_test, train_df, test_df = train_test_split(df)

    # Optional sanity checks (keep commented)
    # print("Train years:", train_df["year"].min(), "-", train_df["year"].max())
    # print("Test years:", test_df["year"].min(), "-", test_df["year"].max())
    # print("Train size:", X_train.shape)
    # print("Test size:", X_test.shape)

    
    # 3) OLS baseline
    ols_model = train_linear_regression(X_train, y_train)
    ols_pred = predict(ols_model, X_test)
    ols_metrics = evaluate_regression(y_test, ols_pred)


    # 4) Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = predict(rf_model, X_test)
    rf_metrics = evaluate_regression(y_test, rf_pred)


    import matplotlib.pyplot as plt
    import pandas as pd

    # Feature importance from Random Forest

    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # Plot
    plt.figure()
    plt.barh(feature_importance.index, feature_importance.values)
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance")
    
    plt.tight_layout()
    plt.savefig("results/feature_importance_rf.png")
    plt.close()

    print("Feature importance (Random Forest):")
    print(feature_importance)


    # 5) Print results
    print("Test years:", list(test_df["year"]))
    print("OLS metrics:", ols_metrics)
    print("RF metrics:", rf_metrics)

    # Optional: see actual vs predicted (keep commented)
    # print("y_test:", [float(v) for v in y_test.values])
    # print("OLS y_pred:", [float(v) for v in ols_pred])
    # print("RF  y_pred:", [float(v) for v in rf_pred])


    # --- Standardized comparison (z-scores) ---
    cols_to_plot = [
        "co2_million_tonnes",
        "gdp_real_constant_usd",
        "unemployment_rate",
        "inflation_cpi",
    ]

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
    plt.savefig("results/standardized_trends_co2_macro_france.png")
    plt.close()


if __name__ == "__main__":
    main()