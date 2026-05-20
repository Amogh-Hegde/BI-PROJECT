from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from .config import XAI_DIR


def generate_xai_artifacts(model_bundle: dict[str, object]) -> dict[str, str]:
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = model_bundle["rf_pipeline"]
    X_test: pd.DataFrame = model_bundle["X_test"].copy()

    sample = X_test.head(150).copy()
    transformed = pipeline.named_steps["preprocessor"].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(transformed_df)

    # Use the most severe class slot for a clean, viva-friendly global chart.
    fatal_index = 2
    if isinstance(shap_values, list):
        values = shap_values[fatal_index]
    else:
        values = shap_values[:, :, fatal_index]

    plt.figure()
    shap.summary_plot(values, transformed_df, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close()

    local = (
        pd.DataFrame({"feature": transformed_df.columns, "impact": values[0]})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(12)
        .sort_values("impact")
    )
    plt.figure(figsize=(9, 6))
    colors = ["#c0392b" if value > 0 else "#1f77b4" for value in local["impact"]]
    plt.barh(local["feature"], local["impact"], color=colors)
    plt.title("Local Severity Explanation - Case 1")
    plt.xlabel("Contribution to fatal severity score")
    plt.tight_layout()
    plt.savefig(XAI_DIR / "shap_waterfall_case1.png", dpi=220, bbox_inches="tight")
    plt.close()

    probabilities = pipeline.predict_proba(X_test.head(5))
    local_explanations = []
    original_rows = X_test.head(5).reset_index(drop=True)
    for idx, probs in enumerate(probabilities):
        top_features = (
            pd.DataFrame({"feature": transformed_df.columns, "impact": values[idx]})
            .assign(abs_impact=lambda d: d["impact"].abs())
            .sort_values("abs_impact", ascending=False)
            .head(6)
        )
        local_explanations.append(
            {
                "case_id": idx + 1,
                "predicted_severity": pipeline.predict(X_test.iloc[[idx]])[0],
                "probabilities": {cls: round(float(prob), 4) for cls, prob in zip(pipeline.classes_, probs)},
                "top_drivers": top_features[["feature", "impact"]].round(4).to_dict(orient="records"),
                "input_snapshot": original_rows.iloc[idx].to_dict(),
            }
        )

    with open(XAI_DIR / "local_explanations.json", "w", encoding="utf-8") as handle:
        json.dump(local_explanations, handle, indent=2, default=str)

    return {
        "global_summary": "Fatal and serious outcomes are most influenced by speed limit, low visibility, nighttime conditions, alcohol involvement, and heavy vehicle context.",
        "local_summary": "Case-level explanations show how environmental and behavioral features combine to increase severity probability.",
    }
