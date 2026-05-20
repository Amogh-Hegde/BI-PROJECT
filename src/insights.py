from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import OUTPUTS_DIR, TABLES_DIR


def generate_insights(df: pd.DataFrame, model_results: dict[str, object]) -> dict[str, list[str]]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    severe_df = df[df["severity"].isin(["Serious", "Fatal"])].copy()

    dangerous_hours = (
        severe_df.groupby("hour_of_day").size().sort_values(ascending=False).head(3).index.astype(int).tolist()
    )
    weather_rates = (
        df.assign(severe_flag=df["severity"].isin(["Serious", "Fatal"]).astype(int))
        .groupby("weather_conditions")
        .agg(accidents=("accident_id", "count"), severe_rate=("severe_flag", "mean"))
        .query("accidents >= 100")
        .sort_values(["severe_rate", "accidents"], ascending=[False, False])
    )
    road_rates = (
        df.assign(severe_flag=df["severity"].isin(["Serious", "Fatal"]).astype(int))
        .groupby("road_condition")
        .agg(accidents=("accident_id", "count"), severe_rate=("severe_flag", "mean"))
        .query("accidents >= 100")
        .sort_values(["severe_rate", "accidents"], ascending=[False, False])
    )
    worst_weather = weather_rates.index.tolist()[:3]
    road_patterns = road_rates.index.tolist()[:3]
    hotspot = df.groupby(["state", "city"]).size().sort_values(ascending=False).head(5)

    insights = {
        "insights": [
            f"Serious and fatal crashes cluster most strongly during hours {dangerous_hours}, highlighting concentrated risk windows rather than all-day severity.",
            f"High-risk weather conditions by severe-crash rate are {', '.join(worst_weather)}, where visibility loss and friction reduction jointly raise severity.",
            f"Road environments with the highest severe-crash rates are {', '.join(road_patterns)}, indicating infrastructure quality is a material risk driver.",
            f"The random forest model achieved weighted F1 of {model_results['metrics']['random_forest']['f1_weighted']}, strong enough for decision-support prototyping.",
            f"Top hotspots in the working dataset are {', '.join([f'{city}, {state}' for state, city in hotspot.index])}.",
        ],
        "recommendations": [
            "Deploy additional patrols and speed enforcement during evening rush and late-night hours in the top hotspot cities.",
            "Install adaptive visibility warning boards and dynamic speed messaging on corridors that combine fog, rain, and high speed limits.",
            "Prioritize junction redesign and signal timing audits for uncontrolled and conflict-heavy intersections before broad citywide interventions.",
            "Use the severity model as a triage layer in control rooms to flag high-risk incidents for faster ambulance and police response allocation.",
            "Launch road-maintenance micro-plans for damaged and under-construction segments with temporary reflective signage and lane discipline measures.",
        ],
    }
    with open(TABLES_DIR / "insights_and_recommendations.json", "w", encoding="utf-8") as handle:
        json.dump(insights, handle, indent=2)
    return insights
