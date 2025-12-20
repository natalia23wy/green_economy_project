import json
from urllib.request import urlopen

import pandas as pd

# Import CO2 datas
OWID_CO2_URL = (
    "https://ourworldindata.org/grapher/"
    "annual-co2-emissions-per-country.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

def fetch_owid_co2_france(start_year=1990, end_year=2024):
    df = pd.read_csv(
        OWID_CO2_URL,
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"}
    )

    # Filter France
    df = df[df["Entity"] == "France"].copy()

    # Keep relevant columns
    df = df[["Year", "emissions_total"]]

    # Rename
    df = df.rename(
        columns={
            "Year": "year",
            "emissions_total": "co2_million_tonnes"
        }
    )

    # Filter years
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # Drop missing values
    df = df.dropna().reset_index(drop=True)

    return df


# Import GDP datas
def fetch_worldbank_gdp_france(start_year=1990, end_year=2024):
    url = (
        "https://api.worldbank.org/v2/country/FRA/"
        "indicator/NY.GDP.MKTP.KD?format=json&per_page=20000"
    )

    with urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload[1]  # second element contains observations

    rows = []
    for obs in data:
        year = int(obs["date"])
        value = obs["value"]
        if start_year <= year <= end_year:
            rows.append({"year": year, "gdp_real_constant_usd": value})

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df["gdp_real_constant_usd"] = pd.to_numeric(df["gdp_real_constant_usd"], errors="coerce")
    df = df.dropna()
    return df


# Import unemployment rate datas
def fetch_worldbank_unemployment_france(start_year=1990, end_year=2024):
    url = (
        "https://api.worldbank.org/v2/country/FRA/"
        "indicator/SL.UEM.TOTL.ZS?format=json&per_page=20000"
    )

    with urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload[1]

    rows = []
    for obs in data:
        year = int(obs["date"])
        value = obs["value"]
        if start_year <= year <= end_year:
            rows.append({
                "year": year,
                "unemployment_rate": value
            })

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")
    df = df.dropna()
    return df


# Import CPI based inflation
def fetch_worldbank_inflation_france(start_year=1990, end_year=2024):
    url = (
        "https://api.worldbank.org/v2/country/FRA/"
        "indicator/FP.CPI.TOTL.ZG?format=json&per_page=20000"
    )

    with urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload[1]

    rows = []
    for obs in data:
        year = int(obs["date"])
        value = obs["value"]
        if start_year <= year <= end_year:
            rows.append({
                "year": year,
                "inflation_cpi": value
            })

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df["inflation_cpi"] = pd.to_numeric(df["inflation_cpi"], errors="coerce")
    df = df.dropna()
    return df


# Merge all datasets into a single dataframe and create the final dataset
def build_dataset_france(start_year=1990, end_year=2024, save=True):
    gdp = fetch_worldbank_gdp_france(start_year, end_year)
    co2 = fetch_owid_co2_france(start_year, end_year)
    unemployment = fetch_worldbank_unemployment_france(start_year, end_year)
    inflation = fetch_worldbank_inflation_france(start_year, end_year)

    df = (
        gdp
        .merge(co2, on="year", how="inner")
        .merge(unemployment, on="year", how="inner")
        .merge(inflation, on="year", how="inner")
    )

    if save:
        df.to_csv("data/processed/france_1990_2024.csv", index=False)

    return df

# Split the dataset into a fixed time train/test split
def train_test_split(
    df,
    target_col="co2_million_tonnes",
    train_end_year=2020,
):

    # Train: up to train_end_year
    # Test: after train_end_year
    df = df.sort_values("year").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["year", target_col]]

    train_df = df[df["year"] <= train_end_year]
    test_df = df[df["year"] > train_end_year]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df