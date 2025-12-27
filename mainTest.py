# Dans votre terminal Python ou notebook
from src.data_loader import build_dataset_france, train_val_test_split, scale_features
from src.models import train_linear_regression, train_random_forest, train_xgboost, predict
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error


# Load et split data
df = build_dataset_france()
X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = train_val_test_split(df)

# Scale
X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

# Train models
ols = train_linear_regression(X_train_s, y_train)
rf = train_random_forest(X_train_s, y_train)
xgb = train_xgboost(X_train_s, y_train)


# Fonction rapide pour évaluer
def eval_model(model, name):
    train_pred = predict(model, X_train_s)
    val_pred = predict(model, X_val_s)
    test_pred = predict(model, X_test_s)
    
    print(f"\n{name}:")
    print(f"  Train - R²: {r2_score(y_train, train_pred):.3f} | MAPE: {mean_absolute_percentage_error(y_train, train_pred):.3f}")
    print(f"  Val   - R²: {r2_score(y_val, val_pred):.3f} | MAPE: {mean_absolute_percentage_error(y_val, val_pred):.3f}")
    print(f"  Test  - R²: {r2_score(y_test, test_pred):.3f} | MAPE: {mean_absolute_percentage_error(y_test, test_pred):.3f}")

# Évaluer tous les modèles
eval_model(ols, "OLS")
eval_model(rf, "Random Forest")
eval_model(xgb, "XGBoost")


print(f"Train size: {len(X_train)}")
print(f"Val size: {len(X_val)}")
print(f"Test size: {len(X_test)}")
print(f"Features: {X_train.columns.tolist()}")

import pandas as pd
import matplotlib.pyplot as plt

# Dans votre mainTest.py, ajoutez:
_, _, _, _, _, _, train_df, val_df, test_df = train_val_test_split(df)

plt.figure(figsize=(12, 4))

# CO2 over time
plt.subplot(1, 3, 1)
plt.plot(train_df['year'], train_df['co2_million_tonnes'], 'o-', label='Train')
plt.plot(val_df['year'], val_df['co2_million_tonnes'], 'o-', label='Val')
plt.plot(test_df['year'], test_df['co2_million_tonnes'], 'o-', label='Test')
plt.xlabel('Year')
plt.ylabel('CO2 (million tonnes)')
plt.legend()
plt.title('CO2 Emissions over time')

# GDP over time
plt.subplot(1, 3, 2)
plt.plot(train_df['year'], train_df['gdp_real_constant_usd'], 'o-')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP over time')

# Predictions
ols = train_linear_regression(X_train_s, y_train)
plt.subplot(1, 3, 3)
all_years = pd.concat([train_df['year'], val_df['year'], test_df['year']])
all_y = pd.concat([y_train, y_val, y_test])
all_X_scaled = pd.concat([X_train_s, X_val_s, X_test_s])
all_pred = predict(ols, all_X_scaled)

plt.plot(all_years, all_y, 'o-', label='Actual', linewidth=2)
plt.plot(all_years, all_pred, 's--', label='OLS Pred', alpha=0.7)
plt.axvline(2015, color='red', linestyle='--', alpha=0.5, label='Train/Val split')
plt.axvline(2020, color='orange', linestyle='--', alpha=0.5, label='Val/Test split')
plt.xlabel('Year')
plt.ylabel('CO2')
plt.legend()
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.savefig('diagnostic.png', dpi=150, bbox_inches='tight')
print("Saved diagnostic.png")

# Après avoir entraîné les modèles
import matplotlib.pyplot as plt

models = {
    'OLS': ols,
    'Random Forest': rf,
    'XGBoost': xgb
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models.items()):
    # Prédictions
    train_pred = model.predict(X_train_s)
    val_pred = model.predict(X_val_s)
    
    # Plot train
    axes[idx].scatter(y_train, train_pred, alpha=0.6, label='Train', s=50)
    # Plot val
    axes[idx].scatter(y_val, val_pred, alpha=0.6, label='Val', s=50, color='red')
    
    # Ligne parfaite
    min_val = min(y_train.min(), y_val.min())
    max_val = max(y_train.max(), y_val.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    axes[idx].set_xlabel('Actual CO2')
    axes[idx].set_ylabel('Predicted CO2')
    axes[idx].set_title(f'{name}\nTrain R²={model.score(X_train_s, y_train):.3f}, Val R²={model.score(X_val_s, y_val):.3f}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/overfitting_diagnosis.png', dpi=300)
print("Saved: results/overfitting_diagnosis.png")
