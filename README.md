# Smart Traffic Accident Analysis & Risk Prediction System

This project is a submission-ready Business Intelligence solution for analyzing Indian traffic accidents, identifying accident-prone conditions, predicting severity, and translating results into dashboard-ready decision support.

## Project Overview
- BI focus: descriptive analytics, KPI storytelling, hotspot identification, recommendations
- Predictive focus: accident severity prediction using Logistic Regression and Random Forest
- XAI focus: SHAP summary and local explanation artifacts
- Dashboard focus: Streamlit-based interactive multi-page application

## Folder Structure
```text
Project/
├── data/
├── notebooks/
├── dashboard/
├── report/
├── ppt/
├── src/
├── outputs/
├── requirements.txt
├── README.md
└── main.py
```

## Data Sources
- Kaggle: India Road Accident Dataset Predictive Analysis
- Dataful MoRTH road accident collection
- Government of India OGD road accident catalog pages

Source notes and assumptions are documented in [data/source_catalog.md](/Users/amoghhegde/Desktop/BI%20Project/data/source_catalog.md).

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- SHAP
- Streamlit

## Setup Instructions
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Steps
1. Run the full pipeline:
   ```bash
   python main.py
   ```
2. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## What Gets Generated
- Cleaned dataset in `data/processed/`
- Interactive HTML figures in `outputs/figures/`
- Trained models in `outputs/models/`
- SHAP explanation files in `outputs/xai/`
- Notebook files in `notebooks/`
- Academic report draft in `report/project_report.md`
- Presentation content in `ppt/presentation_content.md`
- Viva prep notes in `report/viva_qa.md`

## Screenshots Section
Add final screenshots here before submission:
- Overview dashboard
- Time analysis page
- Environmental analysis page
- Geographic hotspot page
- Prediction + explanation page

## Business Intelligence Highlights
- Decision-support focused recommendations, not just model metrics
- Interactive dashboard pages aligned with stakeholder questions
- Explainable severity prediction to support trust and actionability
- Clean modular pipeline suitable for student demonstration and viva

## Submission Notes
- The project uses a calibrated India-focused accident record layer because public Indian road accident microdata is fragmented across portals and resource pages.
- All assumptions are documented clearly for academic transparency.
