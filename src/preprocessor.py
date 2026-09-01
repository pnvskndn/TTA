import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from data_loader import (
  load_country_metadata,
  PROCESSED_PATH
)

TARGET_COL = "renewable_energy_pct"

FEATURED_COLS = [
  "gdp_per_capita",
  "urbanization_pct",
  "tertiary_education_pct",
  "coal_electricity_pct",
  "electricity_consumption_pc",
  "renewable_energy_total_pct",
  "gdp_growth_rate",
]

TEST_CUTOFF_YEAR = 2019

def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # year column check
    if "year" not in df.columns:
        raise ValueError("Expected 'year' column. Check data_loader output.")
    df["year"] = df["year"].astype(int)

    # country code mapping
    meta = load_country_metadata()
    name_to_code = dict(zip(
            meta["country_name"].str.lower(),
            meta["country_code"]
        ))
    if "country" in df.columns:
        df["country_code"] = df["country"].str.lower().map(name_to_code)
        df["country_name"] = df["country"]
    elif "country_code" not in df.columns:
        raise ValueError("No country identifier found in DataFrame.")
    
    # drop rows where columns couldnt map country
    df = df.dropna(subset=["country_code"])

    # filter year range
    df = df[(df["year"] >= 2000) & (df["year"] <= 2023)]

    # drop rows missing target
    df = df.dropna(subset=[TARGET_COL])

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy().sort_values(["country_code", "year"])

    for col in FEATURED_COLS:
        if col not in df.columns:
            continue
        # forward fill within country
        df[col] = df.groupby("country_code")[col].transform(
            lambda x: x.ffill().bfill()
        )
        # remaining NANs to global median 
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

    return df 

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["country_code", "year"])

    grp = df.groupby("country_code")[TARGET_COL]

    df["renewable_lag_1y"]  = grp.shift(1)   # last year
    df["renewable_lag_3y"]  = grp.shift(3)   # 3 years ago
    df["renewable_change_1y"] = df[TARGET_COL] - df["renewable_lag_1y"]  # YoY change
    df["renewable_change_3y"] = df[TARGET_COL] - df["renewable_lag_3y"]  # 3-year change

    # Log GDP — heavy right skew
    df["log_gdp_per_capita"] = np.log1p(df["gdp_per_capita"])

    # Drop rows where lag features are NaN (first few years per country)
    df = df.dropna(subset=["renewable_lag_1y"])

    return df

def merge_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = load_country_metadata()
    df = df.merge(meta, on="country_code", how="left")
    return df


def build_feature_matrix(df: pd.DataFrame):
    model_features = [
        "renewable_lag_1y",
        "renewable_lag_3y",
        "renewable_change_1y",
        "log_gdp_per_capita",
        "urbanization_pct",
        "tertiary_education_pct",
        "coal_electricity_pct",
        "electricity_consumption_pc",
        "renewable_energy_total_pct",
        "gdp_growth_rate", 
        "year",                   # captures global trend over time
    ]

    # Only keep features that exist
    model_features = [f for f in model_features if f in df.columns]

    X = df[model_features]
    y = df[TARGET_COL]

    return X, y, model_features


def temporal_train_test_split(df: pd.DataFrame):
    
    train_df = df[df["year"] < TEST_CUTOFF_YEAR].copy()
    test_df  = df[df["year"] >= TEST_CUTOFF_YEAR].copy()

    print(f"Train: {len(train_df)} rows "
          f"({train_df['year'].min()}–{train_df['year'].max()})")
    print(f"Test:  {len(test_df)} rows "
          f"({test_df['year'].min()}–{test_df['year'].max()})")

    return train_df, test_df


def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("Step 1: Cleaning...")
    df = clean(raw_df)
    print(f"  After cleaning: {df.shape}")

    print("Step 2: Handling missing values...")
    df = handle_missing(df)
    print(f"  After missing value treatment: {df.shape}")

    print("Step 3: Adding lag features...")
    df = add_lag_features(df)
    print(f"  After lag features: {df.shape}")

    print("Step 4: Merging country metadata...")
    df = merge_metadata(df)
    print(f"  After metadata merge: {df.shape}")

    # Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved processed dataset to: {PROCESSED_PATH}")

    return df