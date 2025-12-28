# Green Economy Project: Forecasting France’s CO₂ Emissions

## Research question
How accurately can annual CO₂ emissions in France (1990‑2024) be predicted from macroeconomic indicators such as GDP, unemployment, and inflation, and which of these indicators contribute the most to the prediction?

## Setup

### 1) Clone the repository
```bash
git clone https://github.com/natalia23wy/green_economy_project.git
cd green_economy_project
```

### 2) Create the environment (Conda)
```bash
conda env create -f environment.yml
conda activate green-econ-env
```

### 3) Usage
```bash
python main.py
```
The script downloads public data (World Bank, Our World in Data), builds the France dataset (1990‑2024), trains and evaluates six regression models, and saves outputs in `data/processed/` and `results/`.

## Project structure

```
├── main.py                     # Orchestrates data prep, training, evaluation, plots
├── src
│   ├── data_loader.py          # Data fetching & preprocessing (France 1990‑2024)
│   ├── models.py               # Model definitions (OLS, Ridge, Lasso, RF, XGBoost, GB)
│   ├── evaluation.py           # Metrics, comparison, overfitting checks, SHAP
│   └── plotting.py             # Visualizations (temporal, predictions, feature importance, SHAP)
├── data
│   └── processed
│       └── france_1990_2024.csv  # Consolidated dataset (generated and saved by the pipeline)
├── results                     # Generated figures (after running main.py)
└── environment.yml             # Reproducible Conda environment
```

## Results

R² scores by split and model:

| Model              | Train R² | Val R² | Test R² |
|--------------------|---------:|-------:|--------:|
| OLS                | 0.84 | 0.54 | -0.08 |
| **Ridge**          | 0.82 | **0.58** | -1.60 |
| Lasso              | 0.84 | 0.54 | -0.08 |
| Random Forest      | 0.83 | -3.99 | -19.67 |
| XGBoost            | 0.60 | -7.94 | -28.58 |
| Gradient Boosting  | 0.94 | -2.52 | -17.08 |

- Best validation performance: **Ridge** (Val R² ≈ 0.58 on the stored dataset).  
- Feature importance (mean, normalized across models without the year feature):  
  1. GDP (real, constant USD)  
  2. Unemployment rate  
  3. Inflation (CPI)


## Requirements

- Python: **3.10+**
- Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `shap`, `seaborn`, `matplotlib`, `requests`
- Create the Conda environment with `environment.yml`