"""Automated tests for Churn Models, SHAP Explainability, and T-Learner Uplift Modeling."""

from __future__ import annotations

import json
import pandas as pd

from src.utils.config_loader import get_project_root


def test_churn_model_evaluation_metrics_and_calibration() -> None:
    root = get_project_root()
    with open(root / "reports" / "churn_evaluation_metrics.json") as f:
        metrics = json.load(f)

    for variant in [
        "logistic_regression_no_nlp",
        "logistic_regression_with_nlp",
        "xgboost_no_nlp",
        "xgboost_with_nlp",
    ]:
        assert variant in metrics
        test_m = metrics[variant]["test"]
        assert 0.75 <= test_m["roc_auc"] <= 0.95
        assert 0.70 <= test_m["pr_auc"] <= 0.95
        assert 0.10 <= test_m["brier_score"] <= 0.22
        assert test_m["expected_calibration_error"] < 0.05
        assert len(test_m["calibration_curve"]) == 10

    # Verify XGBoost outperforms Logistic Regression on ROC-AUC and Brier score
    assert metrics["xgboost_with_nlp"]["test"]["roc_auc"] > metrics["logistic_regression_with_nlp"]["test"]["roc_auc"]
    assert metrics["xgboost_with_nlp"]["test"]["brier_score"] < metrics["logistic_regression_with_nlp"]["test"]["brier_score"]


def test_shap_explainability_outputs() -> None:
    root = get_project_root()
    shap_global = pd.read_parquet(root / "data" / "processed" / "shap_global_importance.parquet")
    shap_cust = pd.read_parquet(root / "data" / "processed" / "customer_shap_values.parquet")

    assert len(shap_global) >= 29
    assert (shap_global["mean_abs_shap"] >= 0).all()
    assert abs(shap_global["relative_importance_pct"].sum() - 100.0) < 0.5
    assert len(shap_cust) == 20000


def test_t_learner_uplift_and_policy_ranking() -> None:
    root = get_project_root()
    with open(root / "reports" / "uplift_evaluation_metrics.json") as f:
        up_metrics = json.load(f)

    assert up_metrics["pearson_corr_with_ground_truth"] > 0.50
    assert up_metrics["pehe_rmse_vs_ground_truth"] < 0.10
    auuc = up_metrics["auuc_expected_saves"]
    assert auuc["Uplift_T_Learner"] > auuc["Churn_Risk_Targeting"]
    assert auuc["Uplift_T_Learner"] > auuc["Random_Targeting"]
    assert up_metrics["overall_rct_ate"]["p_value"] < 0.001
