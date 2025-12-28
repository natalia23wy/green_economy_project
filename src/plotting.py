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
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def plot_temporal_overview(train_df, val_df, test_df, save_path='results/temporal_overview.png'):
    """
    Plot temporal evolution of all variables across train/val/test splits using seaborn.
    """
    # Combine data for seaborn
    all_data = pd.concat([
        train_df.assign(Split='Train'),
        val_df.assign(Split='Val'), 
        test_df.assign(Split='Test')
    ])
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Dataset Overview: Temporal Trends', fontsize=16, fontweight='bold')
    
    # CO2 over time
    sns.lineplot(data=all_data, x='year', y='co2_million_tonnes', 
                hue='Split', style='Split', markers=True, dashes=False,
                palette=['#2E86AB', '#FF6B6B', '#4ECDC4'], 
                linewidth=2.5, markersize=8, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Year', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('CO2 Emissions by Split', fontweight='bold')
    axes[0, 0].legend(title='Dataset Split')
    
    # GDP over time
    sns.lineplot(data=all_data, x='year', y='gdp_real_constant_usd', 
                hue='Split', style='Split', markers=True, dashes=False,
                palette=['#2E86AB', '#FF6B6B', '#4ECDC4'], 
                linewidth=2.5, markersize=8, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Year', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('GDP (Constant USD)', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('GDP Evolution', fontweight='bold')
    axes[0, 1].legend(title='Dataset Split')
    
    # Unemployment over time
    sns.lineplot(data=all_data, x='year', y='unemployment_rate', 
                hue='Split', style='Split', markers=True, dashes=False,
                palette=['#2E86AB', '#FF6B6B', '#4ECDC4'], 
                linewidth=2.5, markersize=8, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Year', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('Unemployment Rate (%)', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('Unemployment Rate Evolution', fontweight='bold')
    axes[1, 0].legend(title='Dataset Split')
    
    # Inflation over time
    sns.lineplot(data=all_data, x='year', y='inflation_cpi', 
                hue='Split', style='Split', markers=True, dashes=False,
                palette=['#2E86AB', '#FF6B6B', '#4ECDC4'], 
                linewidth=2.5, markersize=8, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Year', fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('Inflation (CPI)', fontweight='bold', fontsize=12)
    axes[1, 1].set_title('Inflation Evolution', fontweight='bold')
    axes[1, 1].legend(title='Dataset Split')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_predictions_timeline(train_df, val_df, test_df, 
                               y_train, y_val, y_test,
                               train_pred, val_pred, test_pred,
                               model_name='Model',
                               save_path='results/predictions_timeline.png'):
    """
    Plot actual vs predicted CO2 emissions over time using seaborn.
    """
    # Create dataframe for seaborn
    timeline_data = pd.DataFrame({
        'year': pd.concat([train_df['year'], val_df['year'], test_df['year']], ignore_index=True),
        'value': pd.concat([pd.Series(y_train.values), pd.Series(y_val.values), pd.Series(y_test.values)], ignore_index=True),
        'type': ['Actual'] * (len(y_train) + len(y_val) + len(y_test))
    })
    
    # Add predictions
    pred_data = pd.DataFrame({
        'year': pd.concat([train_df['year'], val_df['year'], test_df['year']], ignore_index=True),
        'value': np.concatenate([train_pred, val_pred, test_pred]),
        'type': [f'{model_name} Predicted'] * (len(train_pred) + len(val_pred) + len(test_pred))
    })
    
    all_data = pd.concat([timeline_data, pred_data])
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot with seaborn
    sns.lineplot(data=all_data, x='year', y='value', hue='type', 
                style='type', markers=True, dashes=False,
                palette=['black', '#FF6B6B'], 
                linewidth=2.5, markersize=8, ax=ax)
    
    # Split lines
    ax.axvline(2015.5, color='blue', linestyle='--', alpha=0.7, 
               linewidth=2, label='Train/Val split')
    ax.axvline(2020.5, color='orange', linestyle='--', alpha=0.7, 
               linewidth=2, label='Val/Test split')
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=12)
    ax.set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold', fontsize=12)
    ax.set_title(f'Actual vs Predicted CO2 Emissions - {model_name}', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_overfitting_diagnosis(models_dict, X_train, y_train, X_val, y_val,
                                save_path='results/overfitting_diagnosis.png'):
    """
    Create scatter plots of actual vs predicted for overfitting diagnosis.
    """
    n_models = len(models_dict)
    # 2 rows, 3 columns layout for better readability
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to easily iterate
    
    fig.suptitle('Overfitting Diagnosis: Actual vs Predicted', 
                 fontsize=16, fontweight='bold')
    
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
    Create bar chart comparing models across metrics using seaborn.
    """
    # Filter only validation set for comparison
    df_val = comparison_df[comparison_df['Split'] == 'Val'].copy()
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison on Validation Set', fontsize=16, fontweight='bold')
    
    # R² comparison
    df_val_sorted_r2 = df_val.sort_values('R²', ascending=True)
    colors_r2 = ['green' if x > 0 else 'red' for x in df_val_sorted_r2['R²']]
    sns.barplot(data=df_val_sorted_r2, x='R²', y='Model', hue='Model', palette=colors_r2, 
                ax=axes[0], alpha=0.8, edgecolor='black', linewidth=1, legend=False)
    axes[0].set_xlabel('R² Score', fontweight='bold', fontsize=12)
    axes[0].set_title('R² Score (Higher is Better)', fontweight='bold')
    axes[0].axvline(0, color='black', linewidth=1.5, linestyle='--')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df_val_sorted_r2['R²']):
        axes[0].text(v + 0.5 if v > 0 else v - 0.5, i, f'{v:.3f}', 
                    ha='left' if v > 0 else 'right', va='center', fontweight='bold')
    
    # MAPE comparison
    df_val_sorted_mape = df_val.sort_values('MAPE', ascending=False)
    sns.barplot(data=df_val_sorted_mape, x='MAPE', y='Model', hue='Model',
                palette='Oranges_r', ax=axes[1], alpha=0.8, edgecolor='black', linewidth=1, legend=False)
    axes[1].set_xlabel('MAPE', fontweight='bold', fontsize=12)
    axes[1].set_title('MAPE (Lower is Better)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df_val_sorted_mape['MAPE']):
        axes[1].text(v + 0.01, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_residuals(y_true, y_pred, model_name='Model', 
                   save_path='results/residuals.png'):
    """
    Plot residuals analysis using seaborn.
    """
    residuals = y_true - y_pred
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, s=100, 
                   edgecolor='black', linewidth=1, ax=axes[0])
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Residuals', fontweight='bold', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontweight='bold')
    
    # Residuals distribution
    sns.histplot(residuals, bins=15, edgecolor='black', alpha=0.8, 
                kde=True, color='skyblue', ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_feature_importance(importance_df, save_path='results/feature_importance.png'):
    """
    Plot feature importance comparison across models using seaborn.
    """
    # Reshape data for seaborn
    df_melted = importance_df.reset_index().melt(id_vars='feature', var_name='model', value_name='importance')
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create beautiful bar plot with seaborn
    sns.barplot(data=df_melted, x='feature', y='importance', hue='model', 
                palette=['#2E86AB', '#A23B72', '#F18F01', '#E63946', '#06FFA5', '#FFB700'],
                alpha=0.9, edgecolor='black', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Features', fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature Importance (%)', fontweight='bold', fontsize=13)
    ax.set_title('Feature Importance Comparison Across Models', 
                fontweight='bold', fontsize=15, pad=20)
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9, fontweight='bold', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_shap_summary(shap_values, X_data, feature_names=None, 
                      save_path='results/shap_summary.png'):
    """
    Create SHAP summary plot (beeswarm plot).
    """
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
    Plot SHAP-based feature importance comparison across models using seaborn.
    """
    # Reshape data for seaborn
    df_melted = shap_comparison_df.reset_index().melt(id_vars='index', var_name='model', value_name='shap_importance')
    df_melted = df_melted.rename(columns={'index': 'feature'})
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create beautiful bar plot with seaborn
    sns.barplot(data=df_melted, x='feature', y='shap_importance', hue='model', 
                palette=['#2E86AB', '#A23B72', '#F18F01', '#E63946', '#06FFA5', '#FFB700'],
                alpha=0.9, edgecolor='black', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Features', fontweight='bold', fontsize=13)
    ax.set_ylabel('SHAP Importance (%)', fontweight='bold', fontsize=13)
    ax.set_title('SHAP-based Feature Importance Comparison Across Models', 
                fontweight='bold', fontsize=15, pad=20)
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9, fontweight='bold', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
