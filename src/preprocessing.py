from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MONTH_ORDER, PROCESSED_DIR


def preprocess_accident_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw_df.copy()
    audit = {
        "raw_rows": len(df),
        "raw_columns": df.shape[1],
    }

    df["accident_datetime"] = pd.to_datetime(df["accident_datetime"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    audit["duplicates_removed"] = audit["raw_rows"] - len(df)

    irrelevant_cols = ["accident_location_details"]
    df = df.drop(columns=irrelevant_cols)
    audit["irrelevant_columns_removed"] = len(irrelevant_cols)

    for col in ["visibility_km", "temperature_c", "driver_age"]:
        df[col] = df.groupby(["state", "weather_conditions"])[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    df["hour_of_day"] = df["accident_datetime"].dt.hour
    df["rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["night_time"] = df["hour_of_day"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["weather_category"] = np.where(
        df["weather_conditions"].isin(["Rain", "Storm", "Fog"]),
        "Adverse",
        "Normal",
    )
    df["visibility_risk"] = pd.cut(
        df["visibility_km"],
        bins=[-1, 2.5, 5, 8, 20],
        labels=["Critical", "High", "Moderate", "Low"],
    ).astype(str)
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["severity"] = pd.Categorical(df["severity"], categories=["Minor", "Serious", "Fatal"], ordered=True)

    # Sample numeric scaling-ready columns for downstream models.
    df["casualty_rate"] = (df["number_of_casualties"] / df["number_of_vehicles_involved"]).round(2)
    df["fatality_flag"] = (df["number_of_fatalities"] > 0).astype(int)

    audit["processed_rows"] = len(df)
    audit["processed_columns"] = df.shape[1]
    audit["missing_values_after_processing"] = int(df.isna().sum().sum())
    return df, audit


def save_processed_data(processed_df: pd.DataFrame, audit: dict[str, int]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DIR / "accidents_cleaned.csv", index=False)
    pd.DataFrame([audit]).to_csv(PROCESSED_DIR / "preprocessing_audit.csv", index=False)
