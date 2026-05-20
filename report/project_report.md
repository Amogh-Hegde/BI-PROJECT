# Smart Traffic Accident Analysis & Risk Prediction System

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
- Raw rows: 4540
- Duplicate rows removed: 36
- Irrelevant columns removed: 1
- Final rows: 4504
- Final columns: 36
- Remaining missing values: 0

Key preprocessing steps:
- Timestamp conversion and validation
- Duplicate removal
- Group-based median imputation
- Feature engineering: `hour_of_day`, `rush_hour`, `weekend`, `night_time`, `weather_category`, `visibility_risk`
- Severity-friendly categorical ordering

## 9. Exploratory Data Analysis

### Accidents By Hour
Peak accident load occurs around 17:00, showing commute-time exposure rather than uniform risk.

### Accidents By Weekday
Weekday accident counts exceed weekend levels, suggesting a strong link to office commute and freight movement.

### Accidents By Month
Monthly variation is visible, with monsoon and late-year festive mobility periods showing elevated risk.

### Severity Distribution
Minor crashes dominate volume, but the Serious and Fatal share is large enough to justify predictive intervention.

### Weather Vs Severity
Rain, fog, and storms show a disproportionately higher share of Serious and Fatal outcomes than clear weather.

### Visibility Vs Accidents
Critical and High visibility-risk bands carry fewer events but materially worse severity composition.

### Temperature Vs Accidents
Accidents concentrate in moderate-to-high temperatures, reflecting India's dominant exposure bands rather than cold-weather risk.

### Accident Hotspots
Mumbai emerges as the strongest hotspot in the working dataset, indicating where enforcement pilots can start.

### Statewise Accidents
Maharashtra contributes the highest accident volume in the analysis set, making it a high-priority intervention region.

### Road Condition Analysis
Wet, damaged, and under-construction roads show a stronger severe-crash mix than dry roads.

### Traffic Junction Analysis
Uncontrolled and signal-led junctions account for a large share of incidents, suggesting a need for smarter intersection management.


## 10. Machine Learning
Baseline model: Logistic Regression
Primary model: Random Forest Classifier

### Evaluation Summary
- Logistic Regression Accuracy: 0.9156
- Logistic Regression F1-weighted: 0.9087
- Random Forest Accuracy: 0.9234
- Random Forest F1-weighted: 0.9187

The Random Forest model performed better because the relationship between severity and explanatory factors is nonlinear and interaction-heavy.

## 11. Explainable AI
Fatal and serious outcomes are most influenced by speed limit, low visibility, nighttime conditions, alcohol involvement, and heavy vehicle context.

Case-level explanations show how environmental and behavioral features combine to increase severity probability.

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
- Serious and fatal crashes cluster most strongly during hours [8, 18, 9], highlighting concentrated risk windows rather than all-day severity.
- High-risk weather conditions by severe-crash rate are Fog, Storm, Clear, where visibility loss and friction reduction jointly raise severity.
- Road environments with the highest severe-crash rates are Slippery, Wet, Under Construction, indicating infrastructure quality is a material risk driver.
- The random forest model achieved weighted F1 of 0.9187, strong enough for decision-support prototyping.
- Top hotspots in the working dataset are Mumbai, Maharashtra, Coimbatore, Tamil Nadu, Nagpur, Maharashtra, Madurai, Tamil Nadu, Pune, Maharashtra.

## 14. Recommendations
- Deploy additional patrols and speed enforcement during evening rush and late-night hours in the top hotspot cities.
- Install adaptive visibility warning boards and dynamic speed messaging on corridors that combine fog, rain, and high speed limits.
- Prioritize junction redesign and signal timing audits for uncontrolled and conflict-heavy intersections before broad citywide interventions.
- Use the severity model as a triage layer in control rooms to flag high-risk incidents for faster ambulance and police response allocation.
- Launch road-maintenance micro-plans for damaged and under-construction segments with temporary reflective signage and lane discipline measures.


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
