"""CLI Script to run Churn Modeling, SHAP Explainability, T-Learner Uplift, and Financial Engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision_engine.financial_engine import run_financial_decision_engine
from src.explainability.shap_explainer import run_shap_analysis
from src.models.churn_model import train_and_evaluate_churn_models
from src.models.uplift_model import run_uplift_pipeline
from src.utils.config_loader import get_logger, get_project_root, load_config

logger = get_logger("scripts.run_training_and_eval")


def main() -> None:
    config = load_config()
    project_root = get_project_root()
    processed_dir = project_root / config["etl"]["processed_dir"]

    df = pd.read_parquet(processed_dir / "customer_features_with_nlp.parquet")
    with open(project_root / "reports" / "nlp_complaint_summary.json", "r", encoding="utf-8") as f:
        nlp_summary = json.load(f)
    nlp_cols = nlp_summary["nlp_feature_columns"]

    # 1. Churn Prediction & Calibration (Logistic Regression vs XGBoost, with & without NLP)
    churn_res = train_and_evaluate_churn_models(
        df=df,
        nlp_feature_columns=nlp_cols,
        config=config,
        save_artifacts=True,
    )

    # 2. Explainable AI (SHAP Global Importance & Customer Drivers)
    shap_df, _, _ = run_shap_analysis(
        scored_df=churn_res["scored_df"],
        models_bundle=churn_res["models"],
        save_to_disk=True,
    )

    # 3. T-Learner Uplift Modeling & Statistical Policy Comparison
    uplift_df, _, _ = run_uplift_pipeline(
        scored_df=shap_df,
        feature_columns=churn_res["models"]["full_features"],
        config=config,
        save_to_disk=True,
    )

    # 4. Financial Decision Engine & Budget Optimization
    run_financial_decision_engine(
        df=uplift_df,
        config=config,
        save_to_disk=True,
    )
    logger.info("Full training, explainability, uplift, and financial pipeline completed successfully.")


if __name__ == "__main__":
    main()
