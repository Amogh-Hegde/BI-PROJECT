from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, STATE_CITY_MAP


@dataclass(frozen=True)
class BuildResult:
    accident_records_path: Path
    source_summary_path: Path


def _weighted_choice(rng: np.random.Generator, options: list[str], weights: list[float]) -> str:
    return str(rng.choice(options, p=np.array(weights) / np.sum(weights)))


def _severity_label(score: float) -> str:
    if score >= 0.78:
        return "Fatal"
    if score >= 0.50:
        return "Serious"
    return "Minor"


def generate_accident_dataset(seed: int = 42, n_rows: int = 4500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = list(STATE_CITY_MAP.keys())
    state_weights = np.array([0.14, 0.13, 0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08])
    years = np.arange(2018, 2024)

    weather_options = ["Clear", "Rain", "Fog", "Cloudy", "Storm", "Heat"]
    weather_weights = [0.44, 0.20, 0.08, 0.15, 0.05, 0.08]
    road_condition_options = ["Dry", "Wet", "Damaged", "Under Construction", "Slippery"]
    road_condition_weights = [0.54, 0.19, 0.12, 0.09, 0.06]
    road_type_options = ["Urban Arterial", "Highway", "Residential", "Intersection", "Rural Road"]
    road_type_weights = [0.26, 0.24, 0.16, 0.18, 0.16]
    vehicle_type_options = ["Car", "Two-Wheeler", "Truck", "Bus", "Auto-Rickshaw"]
    vehicle_type_weights = [0.25, 0.37, 0.15, 0.08, 0.15]
    traffic_control_options = ["Signal Present", "Stop Sign", "Uncontrolled", "Police Managed", "Roundabout"]
    traffic_control_weights = [0.34, 0.12, 0.29, 0.09, 0.16]
    lighting_options = ["Daylight", "Street Lights", "Poor Lighting", "No Lighting"]
    lighting_weights = [0.48, 0.24, 0.17, 0.11]
    junction_options = ["Non-Junction", "T-Junction", "Cross Junction", "Y-Junction", "Roundabout"]
    junction_weights = [0.39, 0.21, 0.18, 0.08, 0.14]
    gender_options = ["Male", "Female"]
    gender_weights = [0.87, 0.13]
    license_options = ["Valid", "Expired", "Learner", "Invalid"]
    license_weights = [0.81, 0.06, 0.08, 0.05]

    rows = []
    for idx in range(n_rows):
        state = str(rng.choice(states, p=state_weights))
        city, lat, lon = STATE_CITY_MAP[state][rng.integers(0, len(STATE_CITY_MAP[state]))]
        year = int(rng.choice(years, p=[0.14, 0.15, 0.16, 0.17, 0.18, 0.20]))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        hour_weights = np.array([
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.06, 0.04, 0.04,
            0.04, 0.04, 0.05, 0.06, 0.08, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01
        ])
        hour = int(rng.choice(np.arange(24), p=hour_weights / hour_weights.sum()))
        minute = int(rng.integers(0, 60))
        accident_dt = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        weekday = accident_dt.day_name()

        weather = _weighted_choice(rng, weather_options, weather_weights)
        road_condition = _weighted_choice(rng, road_condition_options, road_condition_weights)
        road_type = _weighted_choice(rng, road_type_options, road_type_weights)
        vehicle_type = _weighted_choice(rng, vehicle_type_options, vehicle_type_weights)
        traffic_control = _weighted_choice(rng, traffic_control_options, traffic_control_weights)
        lighting = _weighted_choice(rng, lighting_options, lighting_weights)
        junction = _weighted_choice(rng, junction_options, junction_weights)
        driver_gender = _weighted_choice(rng, gender_options, gender_weights)
        license_status = _weighted_choice(rng, license_options, license_weights)

        speed_limit = int(rng.choice([30, 40, 50, 60, 80, 100], p=[0.12, 0.20, 0.25, 0.19, 0.16, 0.08]))
        driver_age = int(np.clip(rng.normal(34, 11), 18, 72))
        vehicles = int(rng.choice([1, 2, 3, 4], p=[0.18, 0.49, 0.25, 0.08]))
        alcohol_flag = "Yes" if rng.random() < 0.11 else "No"

        visibility = float(np.clip(rng.normal(6.5, 2.1), 0.5, 12.0))
        if weather == "Fog":
            visibility = float(np.clip(rng.normal(2.0, 0.8), 0.2, 5.0))
        if weather == "Storm":
            visibility = float(np.clip(rng.normal(3.4, 1.0), 0.4, 6.0))

        base_temp = {
            "Delhi": 29.0,
            "Rajasthan": 31.0,
            "Tamil Nadu": 30.0,
            "Kerala": 28.0,
            "Karnataka": 26.0,
            "West Bengal": 29.0,
        }.get(state, 27.0)
        season_shift = {12: -4, 1: -5, 2: -3, 3: 1, 4: 4, 5: 6, 6: 3, 7: 1, 8: 1, 9: 1, 10: 0, 11: -2}[month]
        temperature = float(np.clip(rng.normal(base_temp + season_shift, 3.8), 9, 43))

        severity_score = 0.18
        severity_score += 0.16 if speed_limit >= 80 else 0.0
        severity_score += 0.11 if road_type == "Highway" else 0.0
        severity_score += 0.10 if alcohol_flag == "Yes" else 0.0
        severity_score += 0.08 if visibility <= 3 else 0.0
        severity_score += 0.07 if road_condition in {"Wet", "Slippery"} else 0.0
        severity_score += 0.07 if lighting in {"Poor Lighting", "No Lighting"} else 0.0
        severity_score += 0.05 if junction != "Non-Junction" else 0.0
        severity_score += 0.05 if weather in {"Fog", "Storm"} else 0.0
        severity_score += 0.04 if hour in {0, 1, 2, 3, 4, 22, 23} else 0.0
        severity_score += 0.05 if vehicle_type in {"Truck", "Bus"} else 0.0
        severity_score += float(rng.normal(0, 0.06))
        severity_score = float(np.clip(severity_score, 0.05, 0.97))
        severity = _severity_label(severity_score)

        casualties = 1
        if severity == "Serious":
            casualties = int(rng.choice([1, 2, 3, 4], p=[0.24, 0.44, 0.23, 0.09]))
        elif severity == "Fatal":
            casualties = int(rng.choice([1, 2, 3, 4, 5], p=[0.16, 0.29, 0.27, 0.18, 0.10]))
        else:
            casualties = int(rng.choice([1, 2, 3], p=[0.61, 0.30, 0.09]))

        fatalities = 0
        if severity == "Fatal":
            fatalities = int(rng.choice([1, 2, 3], p=[0.72, 0.22, 0.06]))

        rows.append(
            {
                "accident_id": f"IND-ACC-{idx + 1:05d}",
                "state": state,
                "city": city,
                "accident_datetime": accident_dt,
                "year": year,
                "month": accident_dt.month_name(),
                "weekday": weekday,
                "hour": hour,
                "weather_conditions": weather,
                "road_condition": road_condition,
                "road_type": road_type,
                "vehicle_type": vehicle_type,
                "traffic_control_presence": traffic_control,
                "junction_type": junction,
                "lighting_conditions": lighting,
                "speed_limit_kmph": speed_limit,
                "driver_age": driver_age,
                "driver_gender": driver_gender,
                "driver_license_status": license_status,
                "alcohol_involvement": alcohol_flag,
                "number_of_vehicles_involved": vehicles,
                "number_of_casualties": casualties,
                "number_of_fatalities": fatalities,
                "visibility_km": round(visibility, 2),
                "temperature_c": round(temperature, 1),
                "latitude": round(lat + rng.normal(0, 0.06), 5),
                "longitude": round(lon + rng.normal(0, 0.06), 5),
                "severity": severity,
                "accident_location_details": f"{road_type} near {junction.lower()} zone",
            }
        )

    df = pd.DataFrame(rows)

    # Add small, realistic data-quality issues for preprocessing demonstration.
    duplicate_rows = df.sample(40, random_state=seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    for col, pct in {"visibility_km": 0.05, "temperature_c": 0.04, "driver_age": 0.03}.items():
        mask = rng.random(len(df)) < pct
        df.loc[mask, col] = np.nan
    return df


def create_source_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": "Kaggle - India Road Accident Dataset Predictive Analysis",
                "source_type": "Public ML-ready accident records schema",
                "coverage": "India, 2018-2023, 3,000 record-level accidents",
                "usage_in_project": "Schema and field design for severity prediction layer",
                "link": "https://www.kaggle.com/datasets/khushikyad001/india-road-accident-dataset-predictive-analysis",
            },
            {
                "source_name": "MoRTH / Dataful collection",
                "source_type": "Official state/city aggregate road accident tables",
                "coverage": "India, 2013-2023, state and city views",
                "usage_in_project": "Context, validation, and academic source references for BI framing",
                "link": "https://dataful.in/collections/682/",
            },
            {
                "source_name": "OGD Platform India - Weather and junction resource pages",
                "source_type": "Government table resources",
                "coverage": "State- and city-level accident classification pages",
                "usage_in_project": "Reference metadata for weather and junction dimensions",
                "link": "https://www.data.gov.in/catalog/stateut-wise-number-accidents-persons-killed-and-injured-road-accidents",
            },
        ]
    )


def build_raw_assets(seed: int = 42, n_rows: int = 4500) -> BuildResult:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    accident_df = generate_accident_dataset(seed=seed, n_rows=n_rows)
    source_df = create_source_summary()

    accident_records_path = RAW_DIR / "india_accident_records.csv"
    source_summary_path = RAW_DIR / "source_catalog.csv"
    accident_df.to_csv(accident_records_path, index=False)
    source_df.to_csv(source_summary_path, index=False)
    return BuildResult(accident_records_path=accident_records_path, source_summary_path=source_summary_path)
