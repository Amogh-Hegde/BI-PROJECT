from __future__ import annotations

import json

import pandas as pd

from src.data_builder import build_raw_assets
from src.eda import run_eda
from src.insights import generate_insights
from src.modeling import train_models
from src.preprocessing import preprocess_accident_data, save_processed_data
from src.reporting import create_notebooks, create_presentation_and_viva, create_report, create_source_notes
from src.xai import generate_xai_artifacts


def main() -> None:
    build_raw_assets(seed=42, n_rows=4500)
    raw_df = pd.read_csv("data/raw/india_accident_records.csv")
    cleaned_df, audit = preprocess_accident_data(raw_df)
    save_processed_data(cleaned_df, audit)

    eda_observations = run_eda(cleaned_df)
    model_bundle = train_models(cleaned_df)
    xai_summaries = generate_xai_artifacts(model_bundle)
    insights_bundle = generate_insights(cleaned_df, model_bundle)

    create_notebooks()
    create_report(audit, eda_observations, model_bundle["metrics"], xai_summaries, insights_bundle)
    create_presentation_and_viva(model_bundle["metrics"], insights_bundle)
    create_source_notes()

    summary = {
        "rows_after_preprocessing": audit["processed_rows"],
        "random_forest_metrics": model_bundle["metrics"]["random_forest"],
        "generated_outputs": [
            "data/processed/accidents_cleaned.csv",
            "outputs/figures/*.html",
            "outputs/models/*.joblib",
            "outputs/xai/*",
            "report/project_report.md",
            "ppt/presentation_content.md",
            "report/viva_qa.md",
            "notebooks/*.ipynb",
        ],
    }
    with open("outputs/project_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
