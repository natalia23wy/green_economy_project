from urllib.request import urlopen
from sklearn.preprocessing import StandardScaler

import json
import pandas as pd
import requests

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


# Import renewable energy share datas
def fetch_owid_renewable_france(start_year=1990, end_year=2024):
    url = (
        "https://ourworldindata.org/grapher/"
        "share-of-final-energy-consumption-from-renewable-sources.csv"
        "?v=1&csvType=full&useColumnShortNames=true"
    )
    
    df = pd.read_csv(
        url,
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"}
    )
    
    # Filter France
    df = df[df["Entity"] == "France"].copy()
    
    # Keep relevant columns
    df = df[["Year", "_7_2_1__eg_fec_rnew"]]
    
    # Rename
    df = df.rename(
        columns={
            "Year": "year",
            "_7_2_1__eg_fec_rnew": "renewable_energy_pct"
        }
    )
    
    # Filter years
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    
    return df


def fetch_eurostat_ipi_france(start_year=1990, end_year=2024):    
    # Eurostat API endpoint
    url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inpr_m"
    
    # Query parameters: France, non-seasonally adjusted (NSA)
    params = {
        "geo": "FR",
        "s_adj": "NSA",
    }
    
    # Request data from Eurostat
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    # Extract dimension metadata
    dims = data["dimension"]
    dim_ids = data["id"]
    size = data["size"]
    values = data.get("value", {})
    
    # Helper function: decode flat index to multi-dimensional coordinates
    # Eurostat stores series in a flat index format
    def decode(flat_i, size_list):
        coords = []
        for s in reversed(size_list):
            coords.append(flat_i % s)
            flat_i //= s
        return list(reversed(coords))
    
    # Build inverse lookup for each dimension: index -> label
    inv_maps = {}
    for dim_id in dim_ids:
        cat = dims[dim_id]["category"]["index"]  # label -> idx
        inv_maps[dim_id] = {v: k for k, v in cat.items()}  # idx -> label
    
    # Parse all observations
    rows = []
    for flat_i_str, val in values.items():
        flat_i = int(flat_i_str)
        coords = decode(flat_i, size)
        rec = {}
        for dim_id, coord in zip(dim_ids, coords):
            rec[dim_id] = inv_maps[dim_id][coord]
        rec["value"] = val
        rows.append(rec)
    
    # Create DataFrame from parsed data
    df = pd.DataFrame(rows)
    
    # Extract year from time dimension (format: "1990M01" -> 1990)
    df["year"] = df["time"].str.slice(0, 4).astype(int)
    
    # Filter by requested year range
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    
    # Eurostat may return multiple units/series
    # Pick the most frequent unit to keep one consistent series
    if "unit" in df.columns:
        unit_choice = df["unit"].value_counts().idxmax()
        df = df[df["unit"] == unit_choice].copy()
    
    # Aggregate monthly IPI to annual mean
    df = (
        df.groupby("year", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "industrial_production_index"})  # Renamed for consistency
        .sort_values("year")
        .reset_index(drop=True)
    )
    
    return df


# Merge all datasets into a single dataframe and create the final dataset
def build_dataset_france(start_year=1990, end_year=2024, save=True, interpolate=False):
    
    """
    Build the complete dataset for France years 1990 to 2024.
    Returns: pd.DataFrame, the merged dataset with all indicators
    """
    
    gdp = fetch_worldbank_gdp_france(start_year, end_year)
    co2 = fetch_owid_co2_france(start_year, end_year)
    unemployment = fetch_worldbank_unemployment_france(start_year, end_year)
    inflation = fetch_worldbank_inflation_france(start_year, end_year)
    #ipi = fetch_eurostat_ipi_france(start_year, end_year)
    #renewable = fetch_owid_renewable_france(start_year, end_year)

    # Start with outer merge to keep all years
    df = (
        gdp
        .merge(co2, on="year", how="outer")
        .merge(unemployment, on="year", how="outer")
        .merge(inflation, on="year", how="outer")
        #.merge(ipi, on="year", how="outer")
        #.merge(renewable, on="year", how="outer")
    )

    # drop gdp try to keep only IPI to avoid multicolinearity
    #df = df.drop(columns=['gdp_real_constant_usd'])

    # Sort by year
    df = df.sort_values("year").reset_index(drop=True)

    # Handle missing values
    if interpolate:
        # Linear interpolation for missing values
        df = df.interpolate(method='linear')
        df = df.dropna()  # Drop any remaining NaN (edges)
    else:
        # Drop rows with any missing values
        df = df.dropna()

    df = df.reset_index(drop=True)

    # year as numeric feature to capture time trend
    df['year_numeric'] = df['year']

    if save:
        df.to_csv("data/processed/france_1990_2024.csv", index=False)

    return df


# Split the dataset into a fixed time train/validation/test split
def train_val_test_split(
    df,
    target_col="co2_million_tonnes",
    train_end_year=2015,
    val_end_year=2020,
):
    """
    Split the dataset into train, validation and test sets based on year.

    Parameters:
        df: pd.DataFrame, Dataset to split
        target_col: str, Name of the target column
        train_end_year: int, Last year is included in training set (1990-2015)
        val_end_year: int, Last year is included in validation set (2016-2020)
    
    Returns: 
        tuple (X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df)
    """

    # Train: years <= train_end_year
    # Validation: train_end_year < years <= val_end_year
    # Test: years > val_end_year

    df = df.sort_values("year").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["year", target_col]]

    train_df = df[df["year"] <= train_end_year]
    val_df = df[(df["year"] > train_end_year) & (df["year"] <= val_end_year)]
    test_df = df[df["year"] > val_end_year]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_df, val_df, test_df
    )


# Scale features using StandardScaler
def scale_features(X_train, X_val=None, X_test=None):
    """
    Standardize features using StandardScaler.

    The scaler is fit only on the training set to avoid data leakage.
    the same scaler is then used to transform validation and test sets.

    Parameters:
        X_train: pd.DataFrame, Training features
        X_val: pd.DataFrame, optional, Validation features
        X_test: pd.DataFrame, optional, Test features
    
    Returns:
        tuples
        - if only X_train: (X_train_scaled, scaler)
        - if X_train and X_val: (X_train_scaled, X_val_scaled, scaler)
        - if all three: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()

    # Fit scaler on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Transform validation and test
    results = [X_train_scaled]

    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        results.append(X_val_scaled)
    
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        results.append(X_test_scaled)

    # return scaler as last element
    results.append(scaler)

    return tuple(results)