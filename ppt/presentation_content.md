# Smart Traffic Accident Analysis & Risk Prediction System

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
- RF F1-weighted: 0.9187
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
- Deploy additional patrols and speed enforcement during evening rush and late-night hours in the top hotspot cities.
- Install adaptive visibility warning boards and dynamic speed messaging on corridors that combine fog, rain, and high speed limits.
- Prioritize junction redesign and signal timing audits for uncontrolled and conflict-heavy intersections before broad citywide interventions.
- Use the severity model as a triage layer in control rooms to flag high-risk incidents for faster ambulance and police response allocation.
- Launch road-maintenance micro-plans for damaged and under-construction segments with temporary reflective signage and lane discipline measures.

Visuals to insert: Recommendation cards
Speaking points: Tie each recommendation to evidence from the analysis.

## Slide 14: Conclusion and Future Scope
- BI-driven road safety intelligence is feasible
- Model supports prioritization, not replacement of experts
- Future scope: live APIs, district-level expansion, Power BI deployment
Visuals to insert: Roadmap timeline
Speaking points: End with impact, scalability, and academic contribution.
