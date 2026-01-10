# Green Economy Project: Forecasting France’s CO₂ Emissions

## Research question
How accurately can annual CO₂ emissions in France (1991‑2024) be predicted from macroeconomic indicators such as GDP, unemployment, and inflation, and which of these indicators contribute the most to the prediction?

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
The script downloads public data (World Bank, Our World in Data), builds the France dataset (1991‑2024), trains and evaluates six regression models, and saves outputs in `data/processed/` and `results/`.

## Project structure

```
├── main.py                     # Orchestrates data prep, training, evaluation, plots
├── src
│   ├── data_loader.py          # Data fetching & preprocessing (France 1991‑2024)
│   ├── models.py               # Model definitions (OLS, Ridge, Lasso, RF, XGBoost, GB)
│   ├── hyperparameters_optimization.py # Validation-based hyperparameters tuning with overfitting control
│   ├── evaluation.py           # Metrics, comparison, overfitting checks, SHAP
│   └── plotting.py             # Visualizations (temporal, predictions, feature importance, SHAP)
├── data
│   └── processed
│       └── france_1991_2024.csv  # Consolidated dataset (generated and saved by the pipeline)
├── results                     # Generated figures (after running main.py)
└── environment.yml             # Reproducible Conda environment
```

## Results

Model performance is evaluated on three splits: train (1991–2015), validation (2016–2020) and test (2021–2024).

R² scores by split and model:

| Model              | Train R² | Val R² | Test R² |
|--------------------|---------:|-------:|--------:|
| OLS                | 0.840 | 0.539 | -0.083 |
| **Ridge**          | 0.836 | **0.765** | -0.435 |
| Lasso              | 0.840 | 0.539 | -0.083 |
| Random Forest      | 0.898 | -2.236 | -13.911 |
| XGBoost            | 0.998 | -1.887 | -15.637 |
| Gradient Boosting  | 0.999 | -1.047 | -11.759 |

Key findings:
- Best validation performance: **Ridge** (Val R² ≈ 0.77).
- Tree-based models strongly overfit, with high training R² and negative validation R².
- Feature importance and SHAP analyse (mean, normalized across models without the year feature) and SHAP analyses consistently rank predictors as:  
  1. real GDP (constant USD)  
  2. Unemployment rate  
  3. Inflation (CPI)


## Requirements

- Python: **3.10+**
- Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `shap`, `seaborn`, `matplotlib`, `requests`
- Create the Conda environment with `environment.yml`

## AI Usage
Use of ChatGPT:
ChatGPT was used as a learning and support tool throughout the project.

Use of GitHub Copilot:
GitHub Copilot was used occasionally during coding.
