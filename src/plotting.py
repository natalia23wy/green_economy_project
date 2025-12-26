"""
Visualization functions for CO2 emissions prediction project.

This module provides functions to:
- Plot temporal trends
- Generate actual vs predicted plots
- Create overfitting diagnostics
- Visualize model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_temporal_overview(train_df, val_df, test_df, save_path='results/temporal_overview.png'):
    """
    Plot temporal evolution of CO2 and GDP across train/val/test splits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dataset Overview: Temporal Trends', fontsize=14, fontweight='bold')
    
    # CO2 over time
    axes[0].plot(train_df['year'], train_df['co2_million_tonnes'], 'o-', 
                 label='Train', linewidth=2, markersize=6)
    axes[0].plot(val_df['year'], val_df['co2_million_tonnes'], 'o-', 
                 label='Val', linewidth=2, markersize=6)
    axes[0].plot(test_df['year'], test_df['co2_million_tonnes'], 'o-', 
                 label='Test', linewidth=2, markersize=6)
    axes[0].set_xlabel('Year', fontweight='bold')
    axes[0].set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold')
    axes[0].legend()
    axes[0].set_title('CO2 Emissions by Split')
    axes[0].grid(True, alpha=0.3)
    
    # GDP over time
    axes[1].plot(train_df['year'], train_df['gdp_real_constant_usd'], 'o-', 
                 color='green', linewidth=2, markersize=6, label='Train')
    axes[1].plot(val_df['year'], val_df['gdp_real_constant_usd'], 'o-', 
                 color='green', linewidth=2, markersize=6, label='Val')
    axes[1].plot(test_df['year'], test_df['gdp_real_constant_usd'], 'o-', 
                 color='green', linewidth=2, markersize=6, label='Test')
    axes[1].set_xlabel('Year', fontweight='bold')
    axes[1].set_ylabel('GDP (Constant USD)', fontweight='bold')
    axes[1].set_title('GDP Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_predictions_timeline(train_df, val_df, test_df, 
                               y_train, y_val, y_test,
                               train_pred, val_pred, test_pred,
                               model_name='Model',
                               save_path='results/predictions_timeline.png'):
    """
    Plot actual vs predicted CO2 emissions over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Concatenate data
    all_years = pd.concat([train_df['year'], val_df['year'], test_df['year']])
    all_actual = pd.concat([pd.Series(y_train.values), pd.Series(y_val.values), pd.Series(y_test.values)])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    
    # Plot
    ax.plot(all_years, all_actual, 'o-', label='Actual', 
            linewidth=2.5, markersize=7, color='black')
    ax.plot(all_years, all_pred, 's--', label=f'{model_name} Predicted', 
            alpha=0.7, linewidth=2, markersize=6, color='red')
    
    # Split lines
    ax.axvline(2015.5, color='blue', linestyle='--', alpha=0.5, 
               linewidth=2, label='Train/Val split')
    ax.axvline(2020.5, color='orange', linestyle='--', alpha=0.5, 
               linewidth=2, label='Val/Test split')
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=12)
    ax.set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold', fontsize=12)
    ax.set_title(f'Actual vs Predicted CO2 Emissions - {model_name}', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_overfitting_diagnosis(models_dict, X_train, y_train, X_val, y_val,
                                save_path='results/overfitting_diagnosis.png'):
    """
    Create scatter plots of actual vs predicted for overfitting diagnosis.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Overfitting Diagnosis: Actual vs Predicted', 
                 fontsize=14, fontweight='bold')
    
    for idx, (name, model) in enumerate(models_dict.items()):
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Scatter plots
        axes[idx].scatter(y_train, train_pred, alpha=0.7, s=100, 
                         label='Train', edgecolors='black', linewidths=1)
        axes[idx].scatter(y_val, val_pred, alpha=0.7, s=100, 
                         color='red', label='Val', edgecolors='black', linewidths=1)
        
        # Perfect prediction line
        min_val = min(y_train.min(), y_val.min())
        max_val = max(y_train.max(), y_val.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                      'k--', alpha=0.5, linewidth=2, label='Perfect fit')
        
        # Formatting
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        
        axes[idx].set_xlabel('Actual CO2', fontweight='bold')
        axes[idx].set_ylabel('Predicted CO2', fontweight='bold')
        axes[idx].set_title(f'{name}\nTrain R²={train_r2:.3f} | Val R²={val_r2:.3f}', 
                           fontweight='bold')
        axes[idx].legend(loc='upper left')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_model_comparison(comparison_df, save_path='results/model_comparison.png'):
    """
    Create bar chart comparing models across metrics.
    """
    # Filter only validation set for comparison
    df_val = comparison_df[comparison_df['Split'] == 'Val'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison on Validation Set', fontsize=14, fontweight='bold')
    
    # R² comparison
    df_val_sorted = df_val.sort_values('R²', ascending=True)
    axes[0].barh(df_val_sorted['Model'], df_val_sorted['R²'], 
                 color=['green' if x > 0 else 'red' for x in df_val_sorted['R²']])
    axes[0].set_xlabel('R² Score', fontweight='bold')
    axes[0].set_title('R² Score (Higher is Better)')
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # MAPE comparison
    df_val_sorted = df_val.sort_values('MAPE', ascending=False)
    axes[1].barh(df_val_sorted['Model'], df_val_sorted['MAPE'], 
                 color='orange')
    axes[1].set_xlabel('MAPE', fontweight='bold')
    axes[1].set_title('MAPE (Lower is Better)')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_residuals(y_true, y_pred, model_name='Model', 
                   save_path='results/residuals.png'):
    """
    Plot residuals analysis.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.7, s=80, edgecolors='black')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontweight='bold')
    axes[0].set_ylabel('Residuals', fontweight='bold')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_feature_importance(importance_df, save_path='results/feature_importance.png'):
    """
    Plot feature importance comparison across models.
    """
    # Normalize importance for each model (0-100 scale)
    df_normalized = importance_df.copy()
    for col in df_normalized.columns:
        max_val = df_normalized[col].max()
        if max_val > 0:
            df_normalized[col] = (df_normalized[col] / max_val) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    x = np.arange(len(df_normalized.index))
    width = 0.25
    n_models = len(df_normalized.columns)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    for i, (model_name, color) in enumerate(zip(df_normalized.columns, colors[:n_models])):
        offset = width * (i - (n_models - 1) / 2)
        bars = ax.bar(x + offset, df_normalized[model_name], width, 
                     label=model_name, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only show label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Features', fontweight='bold', fontsize=12)
    ax.set_ylabel('Relative Importance (%)', fontweight='bold', fontsize=12)
    ax.set_title('Feature Importance Comparison Across Models', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_normalized.index, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_shap_summary(shap_values, X_data, feature_names=None, 
                      save_path='results/shap_summary.png'):
    """
    Create SHAP summary plot (beeswarm plot).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    plt.figure(figsize=(10, 6))
    
    if isinstance(X_data, pd.DataFrame):
        shap.summary_plot(shap_values, X_data, show=False)
    else:
        shap.summary_plot(shap_values, X_data, 
                         feature_names=feature_names, show=False)
    
    plt.title('SHAP Summary: Feature Impact on CO2 Predictions', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_comparison(shap_comparison_df, save_path='results/shap_comparison.png'):
    """
    Plot SHAP-based feature importance comparison across models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    x = np.arange(len(shap_comparison_df.index))
    width = 0.25
    n_models = len(shap_comparison_df.columns)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    
    for i, (model_name, color) in enumerate(zip(shap_comparison_df.columns, colors[:n_models])):
        offset = width * (i - (n_models - 1) / 2)
        bars = ax.bar(x + offset, shap_comparison_df[model_name], width,
                     label=model_name, color=color, alpha=0.8, 
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Features', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean |SHAP Value| (Normalized %)', fontweight='bold', fontsize=12)
    ax.set_title('SHAP-based Feature Importance Comparison', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(shap_comparison_df.index, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, ax
