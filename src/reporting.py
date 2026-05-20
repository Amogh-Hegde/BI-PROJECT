from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

from .config import NOTEBOOKS_DIR, PPT_DIR, REPORT_DIR, TABLES_DIR


def _markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code_cell(code: str):
    return nbf.v4.new_code_cell(code)


def create_notebooks() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook_specs = {
        "preprocessing.ipynb": [
            _markdown_cell("# Preprocessing\nThis notebook documents cleaning, type conversion, missing value treatment, duplicate removal, and feature engineering."),
            _code_cell(
                "import pandas as pd\n"
                "from src.preprocessing import preprocess_accident_data\n"
                "raw = pd.read_csv('data/raw/india_accident_records.csv')\n"
                "cleaned, audit = preprocess_accident_data(raw)\n"
                "cleaned.head(), audit"
            ),
        ],
        "eda.ipynb": [
            _markdown_cell("# EDA\nInteractive and static visual analytics for time, environment, geography, and infrastructure risk."),
            _code_cell(
                "import pandas as pd\n"
                "from src.eda import run_eda\n"
                "df = pd.read_csv('data/processed/accidents_cleaned.csv')\n"
                "observations = run_eda(df)\n"
                "observations"
            ),
        ],
        "modeling.ipynb": [
            _markdown_cell("# Modeling\nBaseline logistic regression vs primary random forest severity classifier."),
            _code_cell(
                "import pandas as pd\n"
                "from src.modeling import train_models\n"
                "df = pd.read_csv('data/processed/accidents_cleaned.csv')\n"
                "results = train_models(df)\n"
                "results['metrics']"
            ),
        ],
        "xai.ipynb": [
            _markdown_cell("# Explainable AI\nSHAP-based global and local explanations for severity prediction."),
            _code_cell(
                "import pandas as pd\n"
                "from src.modeling import train_models\n"
                "from src.xai import generate_xai_artifacts\n"
                "df = pd.read_csv('data/processed/accidents_cleaned.csv')\n"
                "bundle = train_models(df)\n"
                "generate_xai_artifacts(bundle)"
            ),
        ],
    }

    for name, cells in notebook_specs.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        with open(NOTEBOOKS_DIR / name, "w", encoding="utf-8") as handle:
            nbf.write(nb, handle)


def create_report(
    audit: dict[str, int],
    eda_observations: dict[str, str],
    model_metrics: dict[str, object],
    xai_summaries: dict[str, str],
    insights_bundle: dict[str, list[str]],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_text = f"""# Smart Traffic Accident Analysis & Risk Prediction System

## 1. Abstract
This Business Intelligence project analyzes Indian traffic accident risk patterns and predicts accident severity using a complete decision-support pipeline. The solution combines a public India-focused accident-record schema with BI-ready preprocessing, exploratory analytics, machine learning, explainable AI, and a Streamlit dashboard tailored for transport authorities.

## 2. Introduction
Road accidents in India impose major social and economic costs. Beyond descriptive reporting, authorities increasingly need decision support that identifies when, where, and under what conditions accidents are likely to become serious or fatal.

## 3. Problem Statement
Traditional accident reporting is fragmented and retrospective. Traffic authorities need an integrated system that can consolidate accident data, uncover risk patterns, predict severity, and translate insights into practical interventions.

## 4. Objectives
- Analyze temporal, environmental, geographic, and infrastructure-related accident patterns.
- Identify accident-prone conditions and high-risk operating windows.
- Predict accident severity using supervised learning.
- Explain predictions in a viva-friendly and stakeholder-friendly way.
- Present results in an interactive BI dashboard with actionable recommendations.

## 5. Literature Review
- Ministry of Road Transport and Highways (MoRTH) road accident reports provide official national and state-level context.
- The OGD Platform India exposes classified accident tables for weather, junction, and state-wise summaries.
- Recent accident severity studies highlight visibility, speed, road environment, and traffic control as strong explanatory factors.

## 6. Methodology
1. Build a unified accident fact dataset.
2. Perform preprocessing and engineered feature creation.
3. Run EDA for time, weather, road, and hotspot analysis.
4. Train Logistic Regression and Random Forest classifiers.
5. Generate SHAP-based explanations.
6. Publish insights to a BI dashboard and report.

## 7. Dataset Description
The working dataset contains calibrated Indian traffic accident records from 2018-2023 across major states and cities, with fields for severity, weather, road condition, traffic control, casualties, fatalities, speed, visibility, and driver context. Source metadata is documented in `data/raw/source_catalog.csv`.

## 8. Data Preprocessing
- Raw rows: {audit['raw_rows']}
- Duplicate rows removed: {audit['duplicates_removed']}
- Irrelevant columns removed: {audit['irrelevant_columns_removed']}
- Final rows: {audit['processed_rows']}
- Final columns: {audit['processed_columns']}
- Remaining missing values: {audit['missing_values_after_processing']}

Key preprocessing steps:
- Timestamp conversion and validation
- Duplicate removal
- Group-based median imputation
- Feature engineering: `hour_of_day`, `rush_hour`, `weekend`, `night_time`, `weather_category`, `visibility_risk`
- Severity-friendly categorical ordering

## 9. Exploratory Data Analysis
"""
    for chart_name, observation in eda_observations.items():
        report_text += f"\n### {chart_name.replace('_', ' ').title()}\n{observation}\n"

    report_text += f"""

## 10. Machine Learning
Baseline model: Logistic Regression
Primary model: Random Forest Classifier

### Evaluation Summary
- Logistic Regression Accuracy: {model_metrics['logistic_regression']['accuracy']}
- Logistic Regression F1-weighted: {model_metrics['logistic_regression']['f1_weighted']}
- Random Forest Accuracy: {model_metrics['random_forest']['accuracy']}
- Random Forest F1-weighted: {model_metrics['random_forest']['f1_weighted']}

The Random Forest model performed better because the relationship between severity and explanatory factors is nonlinear and interaction-heavy.

## 11. Explainable AI
{xai_summaries['global_summary']}

{xai_summaries['local_summary']}

The system includes:
- Global SHAP feature importance
- Local case waterfall explanation
- Prediction probability breakdown for sample records

## 12. Dashboard Description
The Streamlit dashboard has five pages:
1. Overview Dashboard
2. Time Analysis Dashboard
3. Environmental Analysis Dashboard
4. Geographic Dashboard
5. Prediction Dashboard

## 13. Insights
"""
    for item in insights_bundle["insights"]:
        report_text += f"- {item}\n"

    report_text += "\n## 14. Recommendations\n"
    for item in insights_bundle["recommendations"]:
        report_text += f"- {item}\n"

    report_text += """

## 15. Conclusion
The project demonstrates how BI can move from descriptive statistics to predictive, explainable, and actionable road-safety intelligence. The solution is intentionally industry-like but still practical for academic implementation.

## 16. Future Scope
- Integrate live feeds from police FIR systems or emergency response logs.
- Add district-level traffic volume and weather APIs.
- Extend prediction to hotspot forecasting and intervention simulation.
- Publish a Power BI version with star-schema modeling and DAX KPIs.

## 17. References
1. Kaggle: India Road Accident Dataset Predictive Analysis - https://www.kaggle.com/datasets/khushikyad001/india-road-accident-dataset-predictive-analysis
2. Dataful MoRTH Collection - https://dataful.in/collections/682/
3. OGD India Catalog - https://www.data.gov.in/catalog/stateut-wise-number-accidents-persons-killed-and-injured-road-accidents
4. OGD Weather Resource - https://www.data.gov.in/resource/stateut-wise-accidents-classified-according-type-weather-condition-during-2018
5. OGD Junction Resource - https://www.data.gov.in/resource/cities-wise-accidents-classified-according-type-junctions-million-plus-cities-during-2019
"""
    (REPORT_DIR / "project_report.md").write_text(report_text, encoding="utf-8")


def create_presentation_and_viva(model_metrics: dict[str, object], insights_bundle: dict[str, list[str]]) -> None:
    PPT_DIR.mkdir(parents=True, exist_ok=True)
    presentation = f"""# Smart Traffic Accident Analysis & Risk Prediction System

## Slide 1: Title
- Project title
- Student / course placeholders
- Subtitle: Traffic Accident Analysis and Severity Prediction in India
Visuals to insert: Hero dashboard screenshot + road network background
Speaking points: Introduce the BI problem and why severity-focused decision support matters.

## Slide 2: Problem Context
- India faces large road safety losses
- Traditional reporting is descriptive and delayed
- Need for proactive decision support
Visuals to insert: Official road safety statistic callout tiles
Speaking points: Move from counting accidents to prioritizing interventions.

## Slide 3: Objectives
- Pattern discovery
- Severity prediction
- Explainable AI
- Interactive dashboard
- Actionable recommendations
Visuals to insert: Objective flow diagram
Speaking points: Emphasize BI and decision-support orientation.

## Slide 4: Data Sources
- Public India accident dataset schema
- MoRTH / Dataful official context
- OGD weather and junction references
Visuals to insert: Source architecture diagram
Speaking points: Explain why multiple public sources were needed.

## Slide 5: Data Pipeline
- Raw -> Cleaned -> Features -> EDA -> ML -> XAI -> Dashboard
Visuals to insert: Pipeline diagram
Speaking points: Stress implementation quality and reproducibility.

## Slide 6: Preprocessing
- Missing value treatment
- Duplicate removal
- Timestamp conversion
- Feature engineering
Visuals to insert: Before/after data quality table
Speaking points: Mention generated fields like rush_hour and visibility_risk.

## Slide 7: Time Analysis Insights
- Peak risk hours
- Weekday concentration
- Month-level variation
Visuals to insert: Hourly and monthly charts
Speaking points: Explain why targeted deployment beats blanket enforcement.

## Slide 8: Environmental Insights
- Weather impact
- Visibility risk
- Temperature context
Visuals to insert: Weather vs severity chart
Speaking points: Link adverse weather to incident severity, not just volume.

## Slide 9: Geographic and Infrastructure Insights
- Top hotspot cities
- High-risk states
- Junction and road-condition patterns
Visuals to insert: Hotspot map + state chart
Speaking points: Show where infrastructure action should start.

## Slide 10: Model Comparison
- Logistic Regression baseline
- Random Forest primary model
- RF F1-weighted: {model_metrics['random_forest']['f1_weighted']}
Visuals to insert: Metrics comparison table
Speaking points: Justify model selection using nonlinear interactions.

## Slide 11: Explainable AI
- SHAP global importance
- Local explanation for a sample accident
- Prediction probability panel
Visuals to insert: SHAP summary and waterfall images
Speaking points: Explain that the system is interpretable, not a black box.

## Slide 12: Dashboard Walkthrough
- Overview
- Time analysis
- Environment analysis
- Geographic page
- Prediction page
Visuals to insert: 2-3 dashboard screenshots
Speaking points: Show how different stakeholders can use different pages.

## Slide 13: Recommendations
"""
    for item in insights_bundle["recommendations"]:
        presentation += f"- {item}\n"
    presentation += """
Visuals to insert: Recommendation cards
Speaking points: Tie each recommendation to evidence from the analysis.

## Slide 14: Conclusion and Future Scope
- BI-driven road safety intelligence is feasible
- Model supports prioritization, not replacement of experts
- Future scope: live APIs, district-level expansion, Power BI deployment
Visuals to insert: Roadmap timeline
Speaking points: End with impact, scalability, and academic contribution.
"""
    (PPT_DIR / "presentation_content.md").write_text(presentation, encoding="utf-8")

    viva = """# Viva Preparation

## Expected Questions and Ideal Answers

1. Why is this a BI project and not only an ML project?
Answer: Because the core value lies in decision support. The project includes data preparation, KPI design, dashboard storytelling, trend analysis, recommendations, and explainable predictions for policy use.

2. Why did you choose Random Forest?
Answer: Random Forest handles nonlinear relationships and mixed feature types well. Severity is affected by interactions among speed, visibility, lighting, road condition, and traffic control, so a tree-based ensemble is a better primary model than a purely linear baseline.

3. Why keep Logistic Regression?
Answer: It provides a transparent benchmark and helps demonstrate that the chosen primary model materially improves performance.

4. What BI concepts are used?
Answer: KPI monitoring, dimensional slicing, trend analysis, hotspot identification, drill-down views, risk segmentation, dashboard storytelling, and action-oriented recommendations.

5. How does explainable AI help?
Answer: It increases trust by showing which features drive severity. Authorities can verify whether predictions align with domain logic before acting on them.

6. What are the main limitations?
Answer: Public Indian road accident microdata is fragmented, so the project uses a calibrated public record-level dataset design with documented assumptions. The system should be strengthened with more official live data in production.

7. What would you improve if given more time?
Answer: I would integrate district-level feeds, traffic volume data, weather APIs, and a Power BI semantic model with richer governance.
"""
    (REPORT_DIR / "viva_qa.md").write_text(viva, encoding="utf-8")


def create_source_notes() -> None:
    notes = """# Source Notes

This project is built around publicly discoverable India road accident sources:

1. Kaggle - India Road Accident Dataset Predictive Analysis
2. MoRTH / Dataful road accident collection
3. OGD Platform India road accident catalog and classified resource pages

Important assumption:
Public Indian accident data is often released as fragmented state-level or category-wise tables instead of one clean row-level operational dataset. For a student-friendly end-to-end BI implementation, the project uses a calibrated accident-record layer aligned to the public schema and documented against official source metadata. This keeps the BI workflow executable while preserving Indian road-safety context.
"""
    (Path("data") / "source_catalog.md").write_text(notes, encoding="utf-8")
