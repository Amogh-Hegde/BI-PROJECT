# Viva Preparation

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
