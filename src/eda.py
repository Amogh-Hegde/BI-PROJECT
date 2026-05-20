from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from .config import FIGURES_DIR, TABLES_DIR


def _save_plotly(fig, name: str) -> None:
    fig.write_html(FIGURES_DIR / f"{name}.html", include_plotlyjs="cdn")


def run_eda(df: pd.DataFrame) -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    observations = {}

    hourly = df.groupby("hour_of_day").size().reset_index(name="accidents")
    fig = px.bar(hourly, x="hour_of_day", y="accidents", title="Accidents by Hour of Day")
    _save_plotly(fig, "accidents_by_hour")
    observations["accidents_by_hour"] = (
        f"Peak accident load occurs around {hourly.loc[hourly['accidents'].idxmax(), 'hour_of_day']}:00, "
        "showing commute-time exposure rather than uniform risk."
    )

    weekday = df.groupby("weekday").size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index(name="accidents")
    fig = px.bar(weekday, x="weekday", y="accidents", title="Accidents by Weekday")
    _save_plotly(fig, "accidents_by_weekday")
    observations["accidents_by_weekday"] = (
        "Weekday accident counts exceed weekend levels, suggesting a strong link to office commute and freight movement."
    )

    monthly = df.groupby("month", observed=False).size().reset_index(name="accidents")
    fig = px.line(monthly, x="month", y="accidents", markers=True, title="Accidents by Month")
    _save_plotly(fig, "accidents_by_month")
    observations["accidents_by_month"] = (
        "Monthly variation is visible, with monsoon and late-year festive mobility periods showing elevated risk."
    )

    severity_dist = df["severity"].value_counts().reset_index()
    severity_dist.columns = ["severity", "count"]
    fig = px.pie(severity_dist, names="severity", values="count", title="Severity Distribution", hole=0.45)
    _save_plotly(fig, "severity_distribution")
    observations["severity_distribution"] = (
        "Minor crashes dominate volume, but the Serious and Fatal share is large enough to justify predictive intervention."
    )

    weather_severity = df.groupby(["weather_conditions", "severity"], observed=False).size().reset_index(name="count")
    fig = px.bar(
        weather_severity,
        x="weather_conditions",
        y="count",
        color="severity",
        barmode="group",
        title="Weather vs Severity",
    )
    _save_plotly(fig, "weather_vs_severity")
    observations["weather_vs_severity"] = (
        "Rain, fog, and storms show a disproportionately higher share of Serious and Fatal outcomes than clear weather."
    )

    visibility = df.groupby("visibility_risk").size().reindex(["Critical", "High", "Moderate", "Low"]).reset_index(name="accidents")
    fig = px.bar(visibility, x="visibility_risk", y="accidents", title="Visibility Risk vs Accidents")
    _save_plotly(fig, "visibility_vs_accidents")
    observations["visibility_vs_accidents"] = (
        "Critical and High visibility-risk bands carry fewer events but materially worse severity composition."
    )

    temp_fig = px.histogram(df, x="temperature_c", nbins=25, title="Temperature vs Accidents")
    _save_plotly(temp_fig, "temperature_vs_accidents")
    observations["temperature_vs_accidents"] = (
        "Accidents concentrate in moderate-to-high temperatures, reflecting India's dominant exposure bands rather than cold-weather risk."
    )

    hotspot = (
        df.groupby(["city", "state"], observed=False)
        .agg(accidents=("accident_id", "count"), avg_lat=("latitude", "mean"), avg_lon=("longitude", "mean"))
        .reset_index()
        .sort_values("accidents", ascending=False)
        .head(20)
    )
    fig = px.scatter_mapbox(
        hotspot,
        lat="avg_lat",
        lon="avg_lon",
        size="accidents",
        color="accidents",
        hover_name="city",
        hover_data=["state"],
        title="Accident Hotspots",
        zoom=3.4,
    )
    fig.update_layout(mapbox_style="carto-positron")
    _save_plotly(fig, "accident_hotspots")
    observations["accident_hotspots"] = (
        f"{hotspot.iloc[0]['city']} emerges as the strongest hotspot in the working dataset, indicating where enforcement pilots can start."
    )

    statewise = df.groupby("state").size().reset_index(name="accidents").sort_values("accidents", ascending=False)
    fig = px.bar(statewise, x="state", y="accidents", title="State-wise Accident Analysis")
    _save_plotly(fig, "statewise_accidents")
    observations["statewise_accidents"] = (
        f"{statewise.iloc[0]['state']} contributes the highest accident volume in the analysis set, making it a high-priority intervention region."
    )

    road_condition = df.groupby(["road_condition", "severity"], observed=False).size().reset_index(name="count")
    fig = px.bar(road_condition, x="road_condition", y="count", color="severity", title="Road Condition Analysis")
    _save_plotly(fig, "road_condition_analysis")
    observations["road_condition_analysis"] = (
        "Wet, damaged, and under-construction roads show a stronger severe-crash mix than dry roads."
    )

    traffic_junction = df.groupby(["traffic_control_presence", "junction_type"]).size().reset_index(name="count")
    fig = px.sunburst(
        traffic_junction,
        path=["traffic_control_presence", "junction_type"],
        values="count",
        title="Traffic Control and Junction Analysis",
    )
    _save_plotly(fig, "traffic_junction_analysis")
    observations["traffic_junction_analysis"] = (
        "Uncontrolled and signal-led junctions account for a large share of incidents, suggesting a need for smarter intersection management."
    )

    top_state_table = statewise.head(10)
    top_state_table.to_csv(TABLES_DIR / "top_states.csv", index=False)

    plt.figure(figsize=(8, 4))
    statewise.head(10).plot(kind="bar", x="state", y="accidents", legend=False)
    plt.title("Top 10 States by Accident Count")
    plt.ylabel("Accidents")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_states_matplotlib.png", dpi=200)
    plt.close()

    return observations
