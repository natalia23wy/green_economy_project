"""
Model evaluation functions for CO2 emissions prediction.
This module provides functions to:
    - Calculate regression metrics (R², MAPE, MAE, RMSE)
    - Evaluate models on multiple datasets
    - Generate performance reports
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation.
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Model"):
    """
    Evaluate a model on train, validation and test sets.
    """
    # Generate predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    
    results = {
        'model_name': model_name,
        'train': {
            'predictions': train_pred,
            'metrics': train_metrics
        },
        'val': {
            'predictions': val_pred,
            'metrics': val_metrics
        },
        'test': {
            'predictions': test_pred,
            'metrics': test_metrics
        }
    }
    
    return results


def print_evaluation_report(results):
    """
    Print formatted evaluation report for a model.
    """
    model_name = results['model_name']
    
    print(f"\n{model_name}:")
    print(f"  Train - R²: {results['train']['metrics']['r2']:.3f} | "
          f"MAPE: {results['train']['metrics']['mape']:.3f} | "
          f"RMSE: {results['train']['metrics']['rmse']:.2f}")
    
    print(f"  Val   - R²: {results['val']['metrics']['r2']:.3f} | "
          f"MAPE: {results['val']['metrics']['mape']:.3f} | "
          f"RMSE: {results['val']['metrics']['rmse']:.2f}")
    
    print(f"  Test  - R²: {results['test']['metrics']['r2']:.3f} | "
          f"MAPE: {results['test']['metrics']['mape']:.3f} | "
          f"RMSE: {results['test']['metrics']['rmse']:.2f}")


def compare_models(results_list):
    """
    Compare multiple models and return summary DataFrame.
    """
    rows = []
    
    for results in results_list:
        model_name = results['model_name']
        
        for split in ['train', 'val', 'test']:
            metrics = results[split]['metrics']
            row = {
                'Model': model_name,
                'Split': split.capitalize(),
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse']
            }
            rows.append(row)
    
    df_comparison = pd.DataFrame(rows)
    return df_comparison


def detect_overfitting(results):
    """
    Detect overfitting by comparing train and validation performance.
    """
    train_r2 = results['train']['metrics']['r2']
    val_r2 = results['val']['metrics']['r2']
    
    r2_gap = train_r2 - val_r2
    
    # Simple heuristic
    if r2_gap > 0.5:
        severity = "SEVERE"
    elif r2_gap > 0.3:
        severity = "MODERATE"
    elif r2_gap > 0.1:
        severity = "MILD"
    else:
        severity = "NONE"
    
    diagnostics = {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'r2_gap': r2_gap,
        'overfitting_severity': severity,
        'is_overfitting': r2_gap > 0.1
    }
    
    return diagnostics


def test_ridge_regularization(train_func, X_train, y_train, X_val, y_val, alphas=[0, 0.1, 1.0, 10.0]):
    """
    Test different Ridge regularization strengths.
    """
    results = []
    
    for alpha in alphas:
        model = train_func(X_train, y_train, alpha=alpha)
        val_r2 = model.score(X_val, y_val)
        val_pred = model.predict(X_val)
        val_mape = mean_absolute_percentage_error(y_val, val_pred)
        
        results.append({
            'alpha': alpha,
            'val_r2': val_r2,
            'val_mape': val_mape
        })
    
    df_results = pd.DataFrame(results)
    return df_results


def get_feature_importance(model, feature_names, model_type='tree'):
    """
    Extract feature importance from a trained model.
    """
    if model_type == 'linear':
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_)
    elif model_type == 'tree':
        # For tree-based models, use built-in feature_importances_
        importances = model.feature_importances_
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    return df_importance


def compare_feature_importance(models_dict, feature_names):
    """
    Compare feature importance across multiple models.
    """
    importance_dfs = []
    
    for model_name, (model, model_type) in models_dict.items():
        df_imp = get_feature_importance(model, feature_names, model_type)
        df_imp = df_imp.rename(columns={'importance': model_name})
        importance_dfs.append(df_imp.set_index('feature'))
    
    # Merge all importance DataFrames
    df_comparison = pd.concat(importance_dfs, axis=1)
    
    return df_comparison
