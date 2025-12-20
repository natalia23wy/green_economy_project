# Green Economy Project : Predicting CO2 emissions from macroeconomic indicators.

## Problem statement and motivation
The Green Economy Dashboard aims to explore how macroeconomic indicators such as GDP, unemployment, and inflation can help forecast annual CO2 emissions for France from 1990 to 2024, and identify the most influential drivers.

As an economics student deeply interested in sustainability and macroeconomics dynamics, I believe that integrating ecological concerns into economic analysis is essential for long-term viability. Current economic models often treat the environment as an externality, while in reality, economic growth and ecological health are tightly connected.

This project will take a predictive approach to this relationship, using machine learning models to estimate future CO2 emissions based on economic trends. By combining data analysis and visualization, it will identify which economic factors most strongly influence environmental outcomes. The goal is to provide clear, data-driven insights that make the link between growth and sustainability more tangible, and to highlight how economic indicators can be used to anticipate environmental change.

## Planned approach and technologies
The project will be written in Python and will use:
- Pandas/ NumPy for data cleaning and manipulation
- Matplotlib/ Seaborn for visualizations
- scikit-learn for modeling pipeline:
o baseline: OLS linear regression, for interpretability
o tree-based: Random Forest, for non-linear interactions
o boosting: XGBoost, for high prediction power and controllable over-fitting
- Jupyter Notebook/ Streamlit for the dashboard

The project will rely on public datasets covering for France between 1990 and 2024:
- Real GDP: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=FR
- Unemployment rate: https://www.oecd.org/en/data/indicators/unemployment-rate.html
- CPI based inflation: https://www.oecd.org/en/data/indicators/inflation-cpi.html
- Total CO2 & per capita: https://ourworldindata.org/co2-and-greenhouse-gas-emissions

## Expected challenges and how to address them
First, economic and environmental data do not always share the same reporting periods or completeness, which can lead to inconsistencies. To address this, all variables will be aligned by year, missing values will be handled through linear interpolation if needed.

Second, some indicators are highly correlated, which can distort regression results. I will apply regularization methods to reduce multicollinearity and stabilize estimates.

Third, tree-based models like random Forest or XGBoost risk overfitting when trained on limited data. A strict temporal split will ensure reliable evaluation (1990-2015 for training, 2016-2020 for validation and 2021-2024 for testing).

Finally since complex models can be harder to interpret, tools such as SHAP and feature-importance plots will be used to explain the main drivers of CO2 emissions.

## Success criteria
The dashboard content will include historical overview with a line chart of actual vs. predicted CO2, model comparison with a bar chart et error metrics for the three models, feature importance summary plot to show which economic variables most affect predictions and lastly a scenario explorer with sliders for GDP, unemployment, etc. showing the model’s CO2 forecast in real time.

## Strech goals
Testing an LSTM model to compare deep-learning performance with traditional ML methods.