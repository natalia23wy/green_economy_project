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
from src.evaluation import (
    evaluate_model, 
    print_evaluation_report, 
    compare_models,
    detect_overfitting,
    test_ridge_regularization,
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
    # =========================================================================
    # 1. LOAD AND PREPARE DATAS
    # =========================================================================

    df = build_dataset_france()
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = train_val_test_split(df)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)


    # =========================================================================
    # 2. TRAIN MODELS
    # =========================================================================

    ridge_test_results = test_ridge_regularization(
        train_ridge, X_train_s, y_train, X_val_s, y_val,
        alphas=[0, 0.1, 1.0, 10.0]
    )
    print(ridge_test_results.to_string(index=False))

    ols_model = train_ols(X_train_s, y_train)
    ridge_model = train_ridge(X_train_s, y_train, alpha=0.1)
    lasso_model = train_lasso(X_train_s, y_train, alpha=0.1)
    rf_model = train_random_forest(X_train_s, y_train)
    xgb_model = train_xgboost(X_train_s, y_train)
    gb_model = train_gradient_boosting(X_train_s, y_train)


    # =========================================================================
    # 3. EVALUATE MODELS
    # =========================================================================

    ols_results = evaluate_model(ols_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="OLS")
    ridge_results = evaluate_model(ridge_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Ridge")
    lasso_results = evaluate_model(lasso_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Lasso")
    rf_results = evaluate_model(rf_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="Random Forest")
    xgb_results = evaluate_model(xgb_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, model_name="XGBoost")

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")

    print_evaluation_report(ols_results)
    print_evaluation_report(ridge_results)
    print_evaluation_report(lasso_results)
    print_evaluation_report(rf_results)
    print_evaluation_report(xgb_results)
    
    comparison_df = compare_models([ols_results, ridge_results, lasso_results, rf_results, xgb_results])

    print("\n" + "=" * 70)
    print("VALIDATION SET COMPARISON")

    print(comparison_df[comparison_df['Split'] == 'Val'].to_string(
        index=False,
        float_format=lambda x: '%.3f' % x if abs(x) < 100 else '%.2e' % x)
        )

    print("\n" + "=" * 70)
    print("OVERFITTING DIAGNOSIS")

    for results in [ridge_results, rf_results, xgb_results]:
        diag = detect_overfitting(results)
        print(f"\n{results['model_name']}:")
        print(f"   Train R²: {diag['train_r2']:.3f}")
        print(f"   Val R²:   {diag['val_r2']:.3f}")
        print(f"   Gap:      {diag['r2_gap']:.3f}")
        print(f"   Severity: {diag['overfitting_severity']}")


    # =========================================================================
    # 4. GENERATE VISUALIZATIONS
    # =========================================================================

    # Temporal overview
    plot_temporal_overview(train_df, val_df, test_df, save_path='results/temporal_overview.png')

        # Predictions timeline (Ridge - best model)
    plot_predictions_timeline(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        ridge_results['train']['predictions'],
        ridge_results['val']['predictions'],
        ridge_results['test']['predictions'],
        model_name='Ridge (α=0.1)',
        save_path='results/predictions_timeline.png'
    )
    
    
    # Overfitting diagnosis
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
    
    
    # Model comparison
    plot_model_comparison(comparison_df, save_path='results/model_comparison.png')
    
    
    # Feature importance analysis
    feature_names = list(X_train.columns)
    models_importance = {
        'OLS': (ols_model, 'linear'),
        'Ridge': (ridge_model, 'linear'),
        'Lasso': (lasso_model, 'linear'),
        'Random Forest': (rf_model, 'tree'),
        'XGBoost': (xgb_model, 'tree'),
        'Gradient Boosting': (gb_model, 'tree')
    }
    importance_df = compare_feature_importance(models_importance, feature_names)
    plot_feature_importance(importance_df, save_path='results/feature_importance.png')
    
    
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
    # 5. SHAP ANALYSIS (OPTIONAL)
    # =========================================================================

    # Separate models by type
    linear_models = {
        'OLS': ols_model,
        'Ridge': ridge_model,
        'Lasso': lasso_model
    }
     
    tree_models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model
    }
      
    # Generate SHAP comparison
    shap_comparison = compare_shap_importance(
        linear_models, tree_models,
        X_train_s, X_train_s,
        feature_names
    )
       
    # Plot SHAP comparison
    plot_shap_comparison(shap_comparison, save_path='results/shap_comparison.png')
     

    print("\nSHAP-based feature importance comparison:")
    print(shap_comparison.to_string(float_format="%.2f"))

    # Generate individual SHAP plots for best model (Ridge)
    shap_values, explainer = explain_linear_model_shap(ridge_model, X_train_s, feature_names)
    plot_shap_summary(shap_values, X_train_s, feature_names, save_path='results/shap_summary_ridge.png')


if __name__ == "__main__":
    main()
