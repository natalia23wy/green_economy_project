"""
Main script for CO2 emissions prediction modeling.

This script orchestrates the complete machine learning pipeline:
    - Data loading and preprocessing for France economic indicators
    - Training multiple regression models (OLS, Ridge, Lasso, Random Forest, XGBoost, Gradient Boosting)
    - Model evaluation and performance comparison
    - Feature importance analysis using both built-in importance and SHAP values
    - Visualization generation for results interpretation

Usage:
    - python main.py
"""

from src.data_loader import build_dataset_france, train_val_test_split, scale_features
from src.models import train_ols, train_ridge, train_lasso, train_random_forest, train_xgboost, train_gradient_boosting
from src.hyperparameters_optimization import RIDGE_ALPHA, LASSO_ALPHA, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, GRADIENT_BOOSTING_PARAMS
from src.evaluation import (
    evaluate_model, 
    print_evaluation_report, 
    compare_models,
    create_r2_comparison_table,
    detect_overfitting,
    compare_feature_importance,
    explain_linear_model_shap,
    explain_tree_model_shap,
    compare_shap_importance
)

from src.plotting import (
    plot_temporal_overview,
    plot_predictions_timeline,
    plot_overfitting_diagnosis,
    plot_model_comparison,
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_comparison
)


def main():
    separator = "=" * 70

    # =========================================================================
    # 1. LOAD AND PREPARE DATAS
    # =========================================================================
    
    print(f"\n{separator}")
    print("1) LOADING AND PREPARING DATA")
    print(separator)
    print(" - Loading dataset for France...")
    df = build_dataset_france()
    print("   ✓ Dataset loaded successfully.")

    print(" - Splitting dataset into train/validation/test...")
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = train_val_test_split(df)
    print("   ✓ Split completed.")

    # Add year_numeric for better predictions
    print(" - Adding numeric year feature for modeling...")
    X_train_with_year = X_train.copy()
    X_val_with_year = X_val.copy()
    X_test_with_year = X_test.copy()
    
    X_train_with_year['year_numeric'] = train_df['year']
    X_val_with_year['year_numeric'] = val_df['year']
    X_test_with_year['year_numeric'] = test_df['year']
    print("   ✓ year_numeric added to training, validation, and test sets.")
    
    # Scale all features including year
    print(" - Scaling feature sets (with year)...")
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train_with_year, X_val_with_year, X_test_with_year)
    print("   ✓ Features scaled with year included.")

    # Also keep original features without year for feature importance analysis
    print(" - Preparing scaled sets without year for feature importance...")
    X_train_s_no_year, X_val_s_no_year, X_test_s_no_year, _ = scale_features(X_train, X_val, X_test)
    print("   ✓ Features scaled without year for importance analysis.\n")
    
    # Verify standardization
    # print("Standardized features statistics:")
    # print(X_train_s.describe().loc[['mean', 'std']].round(3))
    # print()


    # =========================================================================
    # 2. TRAIN MODELS
    # =========================================================================

    print(f"{separator}")
    print("2) TRAINING MODELS")
    print(separator)

    print(" - Training individual models...")
    ols_model = train_ols(X_train_s, y_train)
    print("   ✓ OLS trained.")
    ridge_model = train_ridge(X_train_s, y_train)
    print(f"   ✓ Ridge (alpha={RIDGE_ALPHA:.6f}) trained.")
    lasso_model = train_lasso(X_train_s, y_train)
    print(f"   ✓ Lasso (alpha={LASSO_ALPHA:.6f}) trained.")
    rf_model = train_random_forest(X_train_s, y_train)
    print("   ✓ Random Forest trained.")
    xgb_model = train_xgboost(X_train_s, y_train)
    print("   ✓ XGBoost trained.")
    gb_model = train_gradient_boosting(X_train_s, y_train)
    print("   ✓ Gradient Boosting trained.\n")

    # Train separate models without year for feature importance analysis
    ols_model_no_year = train_ols(X_train_s_no_year, y_train)
    ridge_model_no_year = train_ridge(X_train_s_no_year, y_train, alpha=RIDGE_ALPHA)
    lasso_model_no_year = train_lasso(X_train_s_no_year, y_train, alpha=LASSO_ALPHA)
    rf_model_no_year = train_random_forest(X_train_s_no_year, y_train)
    xgb_model_no_year = train_xgboost(X_train_s_no_year, y_train)
    gb_model_no_year = train_gradient_boosting(X_train_s_no_year, y_train)
    print("   ✓ Models without year feature trained.\n")

    # =========================================================================
    # 3. EVALUATE MODELS
    # =========================================================================

    print(f"{separator}")
    print("3) EVALUATING MODELS")
    print(separator)
    print(" - Running evaluation across train/val/test splits...")
    ols_results = evaluate_model(ols_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="OLS")
    ridge_results = evaluate_model(ridge_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Ridge")
    lasso_results = evaluate_model(lasso_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Lasso")
    rf_results = evaluate_model(rf_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Random Forest")
    xgb_results = evaluate_model(xgb_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="XGBoost")
    gb_results = evaluate_model(gb_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Gradient Boosting")

    # Evaluate models without year_numeric for feature importance analysis
    print(" - Evaluating models trained without year_numeric...")
    ols_results_no_year = evaluate_model(ols_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="OLS")
    ridge_results_no_year = evaluate_model(ridge_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="Ridge")
    lasso_results_no_year = evaluate_model(lasso_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="Lasso")
    rf_results_no_year = evaluate_model(rf_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="Random Forest")
    xgb_results_no_year = evaluate_model(xgb_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="XGBoost")
    gb_results_no_year = evaluate_model(gb_model_no_year, X_train_s_no_year, y_train, X_val_s_no_year, y_val, X_test_s_no_year, y_test, model_name="Gradient Boosting")
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")

    print_evaluation_report(ols_results)
    print_evaluation_report(ridge_results)
    print_evaluation_report(lasso_results)
    print_evaluation_report(rf_results)
    print_evaluation_report(xgb_results)
    print_evaluation_report(gb_results)
    
    comparison_df = compare_models([ols_results, ridge_results, lasso_results, rf_results, xgb_results, gb_results])


    # Create R² comparison table for models with year
    r2_df_with_year = create_r2_comparison_table([ols_results, ridge_results, lasso_results, rf_results, xgb_results, gb_results])
    
    print("\n" + "=" * 70)
    print("R² PERFORMANCE WITH YEAR_NUMERIC")
    print(r2_df_with_year.to_string(index=False, float_format="%.3f"))

    # Export R² performance table (train/val/test per model)
    performance_r2 = comparison_df.pivot(index="Model", columns="Split", values="R²")
    performance_r2 = performance_r2[["Train", "Val", "Test"]]
    performance_r2.to_csv("results/model_performance_r2.csv")
    print("   ✓ Saved: results/model_performance_r2.csv")

    print("\n" + "=" * 70)
    print("OVERFITTING DIAGNOSIS")

    for results in [ols_results, ridge_results, lasso_results, rf_results, xgb_results, gb_results]:
        diag = detect_overfitting(results)
        print(f"\n{results['model_name']}:")
        print(f"   Train R²: {diag['train_r2']:.3f}")
        print(f"   Val R²:   {diag['val_r2']:.3f}")
        print(f"   Gap:      {diag['r2_gap']:.3f}")
        print(f"   Severity: {diag['overfitting_severity']}")


    # =========================================================================
    # 4. GENERATE VISUALIZATIONS
    # =========================================================================

    print(f"\n{separator}")
    print("4) GENERATING VISUALIZATIONS")
    print(separator)
    print(" - Creating temporal overview plot...")
    # Temporal overview
    plot_temporal_overview(train_df, val_df, test_df, save_path='results/temporal_overview.png')
    print("   ✓ Saved: results/temporal_overview.png")

    # Predictions timeline for Ridge (best model)
    print(" - Building predictions timeline (Ridge)...")
    plot_predictions_timeline(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        ridge_results['train']['predictions'],
        ridge_results['val']['predictions'],
        ridge_results['test']['predictions'],
        model_name='Ridge',
        save_path='results/predictions_timeline.png'
    )
    print("   ✓ Saved: results/predictions_timeline.png")
    
    # Overfitting diagnosis
    print(" - Plotting overfitting diagnosis for all models...")
    models_dict = {
        'OLS': ols_model,
        'Ridge': ridge_model,
        'Lasso': lasso_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model
    }
    plot_overfitting_diagnosis(models_dict, X_train_s, y_train, X_val_s, y_val,
                               save_path='results/overfitting_diagnosis.png')
    print("   ✓ Saved: results/overfitting_diagnosis.png")
    
    # Model comparison
    print(" - Comparing model performance visually...")
    plot_model_comparison(comparison_df, save_path='results/model_comparison.png')
    print("   ✓ Saved: results/model_comparison.png")
    
    
    # Feature importance analysis (WITHOUT year to focus on economic indicators)
    print(" - Computing and plotting feature importances (without year)...")
    feature_names = list(X_train.columns)  # Original features without year
    

    # Create R² comparison table for models without year
    r2_df_no_year = create_r2_comparison_table([ols_results_no_year, ridge_results_no_year, lasso_results_no_year, rf_results_no_year, xgb_results_no_year, gb_results_no_year])
    
    print("\n" + "=" * 70)
    print("R² PERFORMANCE WITHOUT YEAR_NUMERIC")
    print(r2_df_no_year.to_string(index=False, float_format="%.3f"))
    
    models_importance_no_year = {
        'OLS': (ols_model_no_year, 'linear'),
        'Ridge': (ridge_model_no_year, 'linear'),
        'Lasso': (lasso_model_no_year, 'linear'),
        'Random Forest': (rf_model_no_year, 'tree'),
        'XGBoost': (xgb_model_no_year, 'tree'),
        'Gradient Boosting': (gb_model_no_year, 'tree')
    }
    
    importance_df = compare_feature_importance(models_importance_no_year, feature_names)
    plot_feature_importance(importance_df, save_path='results/feature_importance.png')
    print("   ✓ Saved: results/feature_importance.png")
    importance_df.to_csv('results/feature_importance.csv')
    print("   ✓ Saved: results/feature_importance.csv")

    # Print feature importance table
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    
    print("\nFeature importance (normalized to 100%):")
    print(importance_df.to_string(float_format="%.2f"))
    
    # Identify most important feature per model
    print("\n" + "-" * 70)
    print("Most predictive features per model:")
    for model_name in importance_df.columns:
        top_feature = importance_df[model_name].idxmax()
        top_value = importance_df[model_name].max()
        print(f"  {model_name}: {top_feature} ({top_value:.2f})")
    print("-" * 70)
    
    # =========================================================================
    # 5. SHAP ANALYSIS
    # =========================================================================

    # Separate models by type (WITHOUT year for feature importance focus)
    print(f"\n{separator}")
    print("5) RUNNING SHAP ANALYSIS")
    print(separator)
    print(" - Preparing linear and tree model groups for SHAP (without year)...")
    linear_models_no_year = {
        'OLS': ols_model_no_year,
        'Ridge': ridge_model_no_year,
        'Lasso': lasso_model_no_year
    }
    print("   ✓ Linear models ready (OLS, Ridge, Lasso) without year feature.")

    tree_models_no_year = {
        'Random Forest': rf_model_no_year,
        'XGBoost': xgb_model_no_year,
        'Gradient Boosting': gb_model_no_year
    }
    print("   ✓ Tree models ready (Random Forest, XGBoost, Gradient Boosting) without year feature.")
        
    # Generate SHAP comparison (focusing on economic indicators only)
    shap_comparison = compare_shap_importance(
        linear_models_no_year, tree_models_no_year, 
        X_train_s_no_year, X_train_s_no_year, 
        feature_names
    )
    print("   ✓ SHAP values computed using feature set without year_numeric.")
        
    # Plot SHAP comparison
    print(" - Plotting SHAP comparison...")
    plot_shap_comparison(shap_comparison, save_path='results/shap_comparison.png')
    print("   ✓ Saved: results/shap_comparison.png")
        
    print("\nSHAP-based feature importance comparison:")
    shap_comparison.index.name = "feature"
    print(shap_comparison.to_string(float_format="%.2f"))
    shap_comparison.to_csv('results/shap_feature_importance.csv')
    print("   ✓ Saved: results/shap_feature_importance.csv")
        
    # Generate individual SHAP plots for best model (Ridge)
    print(" - Generating SHAP summary for Ridge...")
    shap_values, explainer = explain_linear_model_shap(ridge_model, X_train_s, feature_names)
    plot_shap_summary(shap_values, X_train_s, feature_names, save_path='results/shap_summary_ridge.png')
    print("   ✓ Saved: results/shap_summary_ridge.png\n")
        
        
if __name__ == "__main__":
    main()
