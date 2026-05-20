from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "accidents_cleaned.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_severity_model.joblib"
XAI_JSON = ROOT / "outputs" / "xai" / "local_explanations.json"


st.set_page_config(page_title="Smart Traffic Accident BI", page_icon="🚦", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def overview_page(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accidents", f"{len(df):,}")
    col2.metric("Fatal Accidents", f"{(df['severity'] == 'Fatal').sum():,}")
    col3.metric("States Covered", df["state"].nunique())
    col4.metric("Avg Casualties / Crash", round(df["number_of_casualties"].mean(), 2))

    st.plotly_chart(px.pie(df, names="severity", title="Severity Distribution", hole=0.5), use_container_width=True)
    trend = df.groupby(["year", "severity"]).size().reset_index(name="accidents")
    st.plotly_chart(px.line(trend, x="year", y="accidents", color="severity", markers=True, title="Trend by Year"), use_container_width=True)


def time_page(df: pd.DataFrame):
    left, right = st.columns(2)
    left.plotly_chart(px.bar(df.groupby("hour_of_day").size().reset_index(name="accidents"), x="hour_of_day", y="accidents", title="Hourly Trends"), use_container_width=True)
    right.plotly_chart(px.bar(df.groupby("month").size().reset_index(name="accidents"), x="month", y="accidents", title="Monthly Trends"), use_container_width=True)
    rush = df.groupby("rush_hour").size().reset_index(name="accidents")
    rush["rush_hour"] = rush["rush_hour"].map({0: "Non Rush Hour", 1: "Rush Hour"})
    st.plotly_chart(px.bar(rush, x="rush_hour", y="accidents", title="Rush Hour Analysis"), use_container_width=True)


def environmental_page(df: pd.DataFrame):
    st.plotly_chart(px.bar(df.groupby(["weather_conditions", "severity"]).size().reset_index(name="count"), x="weather_conditions", y="count", color="severity", barmode="group", title="Weather Impact"), use_container_width=True)
    st.plotly_chart(px.bar(df.groupby("visibility_risk").size().reset_index(name="accidents"), x="visibility_risk", y="accidents", title="Visibility Impact"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="temperature_c", color="severity", nbins=30, title="Temperature Impact"), use_container_width=True)


def geographic_page(df: pd.DataFrame):
    city_hotspots = df.groupby(["city", "state"]).agg(accidents=("accident_id", "count"), latitude=("latitude", "mean"), longitude=("longitude", "mean")).reset_index()
    geo_fig = px.scatter_mapbox(city_hotspots, lat="latitude", lon="longitude", size="accidents", color="accidents", hover_name="city", hover_data=["state"], zoom=3.4, title="Hotspot Map")
    geo_fig.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(geo_fig, use_container_width=True)
    state_summary = df.groupby("state").size().reset_index(name="accidents").sort_values("accidents", ascending=False)
    st.plotly_chart(px.bar(state_summary, x="state", y="accidents", title="State-wise Comparison"), use_container_width=True)


def prediction_page(df: pd.DataFrame):
    model = load_model()
    st.subheader("Severity Prediction")
    example = df.iloc[0]
    col1, col2 = st.columns(2)
    input_row = {}
    for field in [
        "state", "city", "hour_of_day", "rush_hour", "weekend", "night_time", "weather_conditions", "weather_category",
        "road_condition", "road_type", "vehicle_type", "traffic_control_presence", "junction_type", "lighting_conditions",
        "speed_limit_kmph", "driver_age", "driver_gender", "driver_license_status", "alcohol_involvement",
        "number_of_vehicles_involved", "number_of_casualties", "visibility_km", "visibility_risk", "temperature_c"
    ]:
        input_row[field] = example[field]

    input_row["speed_limit_kmph"] = col1.slider("Speed Limit (km/h)", 20, 120, int(example["speed_limit_kmph"]))
    input_row["driver_age"] = col1.slider("Driver Age", 18, 75, int(example["driver_age"]))
    input_row["number_of_vehicles_involved"] = col1.slider("Vehicles Involved", 1, 5, int(example["number_of_vehicles_involved"]))
    input_row["number_of_casualties"] = col2.slider("Casualties", 1, 6, int(example["number_of_casualties"]))
    input_row["visibility_km"] = col2.slider("Visibility (km)", 0.5, 12.0, float(example["visibility_km"]))
    input_row["temperature_c"] = col2.slider("Temperature (C)", 8.0, 45.0, float(example["temperature_c"]))

    if st.button("Predict Severity", type="primary"):
        pred_df = pd.DataFrame([input_row])
        prediction = model.predict(pred_df)[0]
        probabilities = model.predict_proba(pred_df)[0]
        st.success(f"Predicted Severity: {prediction}")
        st.write({cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)})
        if XAI_JSON.exists():
            with open(XAI_JSON, "r", encoding="utf-8") as handle:
                explanations = pd.DataFrame(__import__("json").load(handle))
            st.info("Example explanation artifacts are available in outputs/xai for viva presentation.")
            st.dataframe(explanations[["case_id", "predicted_severity"]], use_container_width=True)


def main():
    st.title("Smart Traffic Accident Analysis & Risk Prediction System")
    st.caption("Business Intelligence solution for traffic authorities and academic evaluation.")
    df = load_data()
    page = st.sidebar.radio(
        "Dashboard Pages",
        ["Overview", "Time Analysis", "Environmental Analysis", "Geographic Analysis", "Prediction"],
    )
    if page == "Overview":
        overview_page(df)
    elif page == "Time Analysis":
        time_page(df)
    elif page == "Environmental Analysis":
        environmental_page(df)
    elif page == "Geographic Analysis":
        geographic_page(df)
    else:
        prediction_page(df)


if __name__ == "__main__":
    main()
