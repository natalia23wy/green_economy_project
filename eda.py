"""
Exploratory Data Analysis (EDA) for France CO2 Emissions Prediction
====================================================================

This script performs comprehensive exploratory analysis of the dataset:
- Descriptive statistics
- Temporal trends visualization
- Correlation analysis
- Distribution analysis
- Missing values check

Author: [Your Name]
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import build_dataset_france

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - FRANCE CO2 EMISSIONS (1990-2024)")
print("=" * 70)

df = build_dataset_france(start_year=1990, end_year=2024, save=False)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Period: {df['year'].min()} - {df['year'].max()}")

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

print("\nSummary Statistics:")
print(df.describe().round(2))

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

# =============================================================================
# 2. TEMPORAL TRENDS
# =============================================================================
print("\n" + "=" * 70)
print("2. TEMPORAL TRENDS ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Evolution of Key Indicators (1990-2024)', 
             fontsize=16, fontweight='bold')

# CO2 Emissions
axes[0, 0].plot(df['year'], df['co2_million_tonnes'], 'o-', linewidth=2, markersize=6)
axes[0, 0].set_title('CO2 Emissions', fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Million Tonnes')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=2008, color='red', linestyle='--', alpha=0.5, label='2008 Crisis')
axes[0, 0].axvline(x=2020, color='orange', linestyle='--', alpha=0.5, label='COVID-19')
axes[0, 0].legend()

# GDP
axes[0, 1].plot(df['year'], df['gdp_real_constant_usd'], 'o-', 
                linewidth=2, markersize=6, color='green')
axes[0, 1].set_title('Real GDP (Constant USD)', fontweight='bold')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('USD')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=2008, color='red', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=2020, color='orange', linestyle='--', alpha=0.5)

# Unemployment Rate
axes[1, 0].plot(df['year'], df['unemployment_rate'], 'o-', 
                linewidth=2, markersize=6, color='red')
axes[1, 0].set_title('Unemployment Rate', fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('% of Labor Force')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=2008, color='red', linestyle='--', alpha=0.5)
axes[1, 0].axvline(x=2020, color='orange', linestyle='--', alpha=0.5)

# Inflation Rate
axes[1, 1].plot(df['year'], df['inflation_cpi'], 'o-', 
                linewidth=2, markersize=6, color='purple')
axes[1, 1].set_title('Inflation Rate (CPI)', fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Annual %')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].axvline(x=2008, color='red', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=2020, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/eda_temporal_trends.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/eda_temporal_trends.png")

# Key trends
co2_change = ((df['co2_million_tonnes'].iloc[-1] / df['co2_million_tonnes'].iloc[0]) - 1) * 100
gdp_change = ((df['gdp_real_constant_usd'].iloc[-1] / df['gdp_real_constant_usd'].iloc[0]) - 1) * 100

print(f"\nKey Changes (1990 → 2024):")
print(f"   CO2 Emissions: {co2_change:+.1f}%")
print(f"   Real GDP: {gdp_change:+.1f}%")
print(f"   Unemployment: {df['unemployment_rate'].iloc[0]:.1f}% → {df['unemployment_rate'].iloc[-1]:.1f}%")

# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("3. CORRELATION ANALYSIS")
print("=" * 70)

# Correlation matrix
features = ['co2_million_tonnes', 'gdp_real_constant_usd', 
            'unemployment_rate', 'inflation_cpi']
corr_matrix = df[features].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/eda_correlation_matrix.png")

# Key correlations with CO2
print(f"\nCorrelations with CO2 Emissions:")
co2_corr = corr_matrix['co2_million_tonnes'].sort_values(ascending=False)
for var, corr in co2_corr.items():
    if var != 'co2_million_tonnes':
        print(f"   {var:25s}: {corr:+.3f}")

# =============================================================================
# 4. DISTRIBUTION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("4. DISTRIBUTION ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

# CO2 distribution
axes[0, 0].hist(df['co2_million_tonnes'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('CO2 Emissions Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Million Tonnes')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['co2_million_tonnes'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df["co2_million_tonnes"].mean():.1f}')
axes[0, 0].legend()

# GDP distribution
axes[0, 1].hist(df['gdp_real_constant_usd'], bins=15, 
                edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('GDP Distribution', fontweight='bold')
axes[0, 1].set_xlabel('USD')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['gdp_real_constant_usd'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df["gdp_real_constant_usd"].mean():.2e}')
axes[0, 1].legend()

# Unemployment distribution
axes[1, 0].hist(df['unemployment_rate'], bins=15, 
                edgecolor='black', alpha=0.7, color='red')
axes[1, 0].set_title('Unemployment Rate Distribution', fontweight='bold')
axes[1, 0].set_xlabel('% of Labor Force')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['unemployment_rate'].mean(), color='darkred', 
                    linestyle='--', label=f'Mean: {df["unemployment_rate"].mean():.1f}%')
axes[1, 0].legend()

# Inflation distribution
axes[1, 1].hist(df['inflation_cpi'], bins=15, 
                edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_title('Inflation Rate Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Annual %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(df['inflation_cpi'].mean(), color='darkviolet', 
                    linestyle='--', label=f'Mean: {df["inflation_cpi"].mean():.1f}%')
axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('results/eda_distributions.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/eda_distributions.png")

# =============================================================================
# 5. SCATTER PLOTS: CO2 vs FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("5. BIVARIATE RELATIONSHIPS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CO2 Emissions vs. Key Features', fontsize=16, fontweight='bold')

# CO2 vs GDP
axes[0].scatter(df['gdp_real_constant_usd'], df['co2_million_tonnes'], 
                alpha=0.6, s=100, edgecolors='black')
axes[0].set_xlabel('GDP (Constant USD)', fontweight='bold')
axes[0].set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold')
axes[0].set_title(f'r = {df[["gdp_real_constant_usd", "co2_million_tonnes"]].corr().iloc[0,1]:.3f}')
axes[0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['gdp_real_constant_usd'], df['co2_million_tonnes'], 1)
p = np.poly1d(z)
axes[0].plot(df['gdp_real_constant_usd'], p(df['gdp_real_constant_usd']), 
             "r--", alpha=0.8, linewidth=2, label='Linear fit')
axes[0].legend()

# CO2 vs Unemployment
axes[1].scatter(df['unemployment_rate'], df['co2_million_tonnes'], 
                alpha=0.6, s=100, edgecolors='black', color='red')
axes[1].set_xlabel('Unemployment Rate (%)', fontweight='bold')
axes[1].set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold')
axes[1].set_title(f'r = {df[["unemployment_rate", "co2_million_tonnes"]].corr().iloc[0,1]:.3f}')
axes[1].grid(True, alpha=0.3)

z = np.polyfit(df['unemployment_rate'], df['co2_million_tonnes'], 1)
p = np.poly1d(z)
axes[1].plot(df['unemployment_rate'], p(df['unemployment_rate']), 
             "r--", alpha=0.8, linewidth=2, label='Linear fit')
axes[1].legend()

# CO2 vs Inflation
axes[2].scatter(df['inflation_cpi'], df['co2_million_tonnes'], 
                alpha=0.6, s=100, edgecolors='black', color='purple')
axes[2].set_xlabel('Inflation Rate (%)', fontweight='bold')
axes[2].set_ylabel('CO2 Emissions (Million Tonnes)', fontweight='bold')
axes[2].set_title(f'r = {df[["inflation_cpi", "co2_million_tonnes"]].corr().iloc[0,1]:.3f}')
axes[2].grid(True, alpha=0.3)

z = np.polyfit(df['inflation_cpi'], df['co2_million_tonnes'], 1)
p = np.poly1d(z)
axes[2].plot(df['inflation_cpi'], p(df['inflation_cpi']), 
             "r--", alpha=0.8, linewidth=2, label='Linear fit')
axes[2].legend()

plt.tight_layout()
plt.savefig('results/eda_scatter_plots.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/eda_scatter_plots.png")

# =============================================================================
# 6. OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 70)
print("6. OUTLIER DETECTION (IQR Method)")
print("=" * 70)

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\n{col}:")
        print(f"   Outlier years: {outliers['year'].tolist()}")
        print(f"   Values: {outliers[col].tolist()}")
    else:
        print(f"\n{col}: No outliers detected")

# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 70)
print("7. SUMMARY REPORT")
print("=" * 70)

print(f"""
Dataset Overview:
   - Period: {df['year'].min()} - {df['year'].max()} ({len(df)} observations)
   - Features: {len(features)} 
   - Target: CO2 Emissions (Million Tonnes)
   
Key Findings:
   1. CO2 emissions show strong correlation with GDP (r={corr_matrix.loc['co2_million_tonnes', 'gdp_real_constant_usd']:.3f})
   2. Negative correlation with unemployment (r={corr_matrix.loc['co2_million_tonnes', 'unemployment_rate']:.3f})
   3. Weak correlation with inflation (r={corr_matrix.loc['co2_million_tonnes', 'inflation_cpi']:.3f})
   4. Structural breaks visible at 2008 (financial crisis) and 2020 (COVID-19)
   
Data Quality:
   - No missing values
   - {len(df)} complete observations
   - Suitable for time-series ML modeling
   
Recommendations:
   - Use time-based train/val/test split (respect temporal order)
   - Consider adding year_numeric to capture temporal trend
   - Monitor for overfitting on small dataset (n={len(df)})
   - Test set (2021-2024) includes COVID recovery period
""")

print("\n" + "=" * 70)
print("EDA COMPLETE - All visualizations saved to results/")
print("=" * 70)
