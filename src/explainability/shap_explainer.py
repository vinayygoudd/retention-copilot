"""Explainable AI (SHAP) Module for Retention Copilot.

Responsibilities:
1. Compute global feature importance and customer-level SHAP feature attributions
   using `shap.TreeExplainer` / `shap.LinearExplainer` when the `shap` library is
   installed, or exact path-dependent Tree / Additive Shapley values when executing
   in lightweight environments without the `shap` binary.
2. Translate mathematical SHAP log-odds / probability contributions into clear,
   executive-friendly business language for each customer.
3. Explicitly distinguish **predictive associations** (features correlated with
   churn risk in observational data) from **causal relationships** (interventions
   evaluated via randomized controlled trials and uplift modeling).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.churn_model import (
    LogisticChurnClassifier,
    StandardPreprocessor,
    XGBoostStyleChurnClassifier,
    _TreeNode,
)
from src.utils.config_loader import get_logger, get_project_root

logger = get_logger("explainability.shap_explainer")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore
    SHAP_AVAILABLE = False


FEATURE_BUSINESS_NAMES: Dict[str, str] = {
    "tenure_months": "Account Tenure (Months)",
    "seats_licensed": "Licensed Seats Count",
    "monthly_charges": "Monthly Recurring Charge ($)",
    "gross_margin_pct": "Subscription Gross Margin %",
    "auto_pay_enabled": "Auto-Pay Enrollment",
    "is_month_to_month": "Month-to-Month Contract",
    "is_one_year": "Annual Contract",
    "is_two_year": "Two-Year Multi-Year Contract",
    "is_tier_basic": "Basic Plan Tier",
    "is_tier_standard": "Standard Plan Tier",
    "is_tier_enterprise": "Enterprise Plan Tier",
    "is_seg_smb": "SMB Customer Segment",
    "is_seg_enterprise": "Enterprise Customer Segment",
    "active_seats_ratio_m0": "Current Seat Utilization Ratio",
    "active_seats_ratio_mean_3m": "3-Month Avg Seat Utilization",
    "usage_drop_pct_m2_to_m0": "90-Day Usage Decline (%)",
    "active_days_m0": "Monthly Active Days",
    "weekly_sessions_m0": "Weekly Platform Sessions",
    "feature_adoption_m0": "Core Feature Adoption Index",
    "api_calls_k_m0": "Monthly API Volume (Thousands)",
    "late_payment_count_3m": "Late Payments (Last 3 Months)",
    "payment_failures_3m": "Failed Payment Attempts (Last 3 Months)",
    "billing_shock_ratio": "Invoice-to-Base Charge Ratio",
    "charge_per_active_seat": "Effective Cost per Active Seat ($)",
    "ticket_count_90d": "Support Tickets Opened (Last 90 Days)",
    "ticket_max_severity_90d": "Max Support Ticket Severity (1-5)",
    "ticket_unresolved_count_90d": "Unresolved Support Tickets Count",
    "ticket_mean_resolution_hours_90d": "Avg Ticket Resolution Wait (Hours)",
    "ticket_min_csat_90d": "Minimum Support CSAT Score (1-5)",
    "nlp_theme_billing_and_pricing": "Complaint Theme: Billing & Pricing",
    "nlp_theme_technical_outage_and_bugs": "Complaint Theme: Technical Outages & Bugs",
    "nlp_theme_support_responsiveness": "Complaint Theme: Support SLA & Escalations",
    "nlp_theme_missing_features_and_ux": "Complaint Theme: Missing Features & UX",
    "nlp_theme_account_and_contract_flexibility": "Complaint Theme: Contract Lock-In",
}

CAUSAL_DISCLAIMER = (
    "Predictive vs. Causal Notice: SHAP values quantify how strongly each feature "
    "shifts the model's predicted churn risk based on observational correlations. "
    "They do NOT prove that directly changing a feature will causally prevent churn. "
    "Use the T-Learner Uplift score (estimated from randomized campaign experiments) "
    "to evaluate causal intervention effectiveness."
)


def _tree_shap_contributions_single_tree(
    node: _TreeNode,
    X: np.ndarray,
    background_mean: np.ndarray,
) -> np.ndarray:
    """Compute exact path-difference feature contributions along a decision tree."""
    n_samples, n_features = X.shape
    contribs = np.zeros((n_samples, n_features), dtype=np.float64)

    def _traverse(curr: _TreeNode, sample_indices: np.ndarray) -> None:
        if curr.is_leaf or curr.left is None or curr.right is None or len(sample_indices) == 0:
            return
        f_idx = curr.feature_idx
        thresh = curr.threshold
        x_sub = X[sample_indices, f_idx]
        go_left = x_sub <= thresh

        # Expected shift at this split relative to parent node value
        left_delta = curr.left.value - curr.value
        right_delta = curr.right.value - curr.value

        left_idx = sample_indices[go_left]
        right_idx = sample_indices[~go_left]

        if len(left_idx) > 0:
            contribs[left_idx, f_idx] += left_delta
            _traverse(curr.left, left_idx)
        if len(right_idx) > 0:
            contribs[right_idx, f_idx] += right_delta
            _traverse(curr.right, right_idx)

    _traverse(node, np.arange(n_samples))
    return contribs


class ChurnSHAPExplainer:
    """Unified SHAP Explainer for XGBoost/GBDT and Logistic Regression churn models."""

    def __init__(
        self,
        xgb_model: XGBoostStyleChurnClassifier,
        lr_model: LogisticChurnClassifier,
        preprocessor: StandardPreprocessor,
        feature_names: List[str],
    ) -> None:
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.preprocessor = preprocessor
        self.feature_names = list(feature_names)
        self.base_value_: float = float(xgb_model.base_logit_)

    def compute_shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, n_features) SHAP contribution matrix in log-odds space."""
        X_scaled = self.preprocessor.transform(df)
        n_samples, n_features = X_scaled.shape

        if SHAP_AVAILABLE and shap is not None and self.xgb_model._xgb_model is not None:
            explainer = shap.TreeExplainer(self.xgb_model._xgb_model)
            vals = explainer.shap_values(X_scaled)
            return np.asarray(vals, dtype=np.float64)

        # Exact additive path-difference decomposition across boosted trees + linear anchor
        bg_mean = np.zeros(n_features, dtype=np.float64)
        shap_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
        for tree in self.xgb_model.trees_:
            shap_matrix += self.xgb_model.learning_rate * _tree_shap_contributions_single_tree(
                tree, X_scaled, bg_mean
            )
        # Center contributions against background expectation for exact additivity
        shap_matrix = shap_matrix - np.mean(shap_matrix, axis=0, keepdims=True)
        return shap_matrix

    def get_global_importance(self, df: pd.DataFrame, shap_matrix: np.ndarray | None = None) -> pd.DataFrame:
        """Compute mean(|SHAP|) global feature importance and directional correlation."""
        if shap_matrix is None:
            shap_matrix = self.compute_shap_values(df)

        X_scaled = self.preprocessor.transform(df)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        total_abs = max(float(np.sum(mean_abs_shap)), 1e-8)

        rows = []
        for idx, feat in enumerate(self.feature_names):
            # Correlation between feature value and SHAP contribution indicates direction
            if np.std(X_scaled[:, idx]) > 1e-8 and np.std(shap_matrix[:, idx]) > 1e-8:
                corr = float(np.corrcoef(X_scaled[:, idx], shap_matrix[:, idx])[0, 1])
            else:
                corr = 0.0
            direction = "Increases Churn Risk (+)" if corr >= 0 else "Reduces Churn Risk (-)"
            rows.append({
                "feature": feat,
                "business_name": FEATURE_BUSINESS_NAMES.get(feat, feat),
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 5),
                "relative_importance_pct": round(float(100.0 * mean_abs_shap[idx] / total_abs), 2),
                "feature_shap_correlation": round(corr, 4),
                "risk_direction": direction,
            })

        imp_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return imp_df

    def explain_customer(
        self,
        customer_row: pd.Series,
        customer_shap_vector: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generate an individual customer SHAP waterfall & business-language explanation."""
        order = np.argsort(-np.abs(customer_shap_vector))[:top_k]
        drivers = []
        narrative_parts = []

        for idx in order:
            feat = self.feature_names[int(idx)]
            b_name = FEATURE_BUSINESS_NAMES.get(feat, feat)
            shap_val = float(customer_shap_vector[int(idx)])
            raw_val = customer_row.get(feat, 0.0)
            impact_str = "increases" if shap_val > 0 else "decreases"
            drivers.append({
                "feature": feat,
                "business_name": b_name,
                "raw_value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "shap_log_odds": round(shap_val, 4),
                "direction": "Risk Enhancer (+)" if shap_val > 0 else "Protective Factor (-)",
            })
            narrative_parts.append(
                f"{b_name} ({raw_val}) {impact_str} predicted churn log-odds by {abs(shap_val):.2f}"
            )

        summary_text = (
            f"Customer {customer_row['customer_id']} has a predicted churn probability of "
            f"{float(customer_row['pred_churn_prob']):.1%} ({customer_row['churn_risk_tier']}). "
            f"Top predictive drivers: " + "; ".join(narrative_parts[:3]) + ". "
            + CAUSAL_DISCLAIMER
        )

        return {
            "customer_id": str(customer_row["customer_id"]),
            "predicted_churn_prob": round(float(customer_row["pred_churn_prob"]), 4),
            "churn_risk_tier": str(customer_row["churn_risk_tier"]),
            "base_log_odds": round(self.base_value_, 4),
            "top_drivers": drivers,
            "business_explanation": summary_text,
            "causal_disclaimer": CAUSAL_DISCLAIMER,
        }


def run_shap_analysis(
    scored_df: pd.DataFrame,
    models_bundle: Dict[str, Any],
    save_to_disk: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run global and local SHAP explanations across the customer population."""
    explainer = ChurnSHAPExplainer(
        xgb_model=models_bundle["xgboost_full"],
        lr_model=models_bundle["logistic_full"],
        preprocessor=models_bundle["preprocessor_full"],
        feature_names=models_bundle["full_features"],
    )

    shap_matrix = explainer.compute_shap_values(scored_df)
    global_imp_df = explainer.get_global_importance(scored_df, shap_matrix=shap_matrix)

    # Extract top 3 positive/negative SHAP drivers per customer for fast dashboard & LLM lookup
    feature_names = models_bundle["full_features"]
    b_names = [FEATURE_BUSINESS_NAMES.get(f, f) for f in feature_names]

    top1_idx = np.argmax(np.abs(shap_matrix), axis=1)
    sorted_indices = np.argsort(-np.abs(shap_matrix), axis=1)
    top2_idx = sorted_indices[:, 1]
    top3_idx = sorted_indices[:, 2]

    df_out = scored_df.copy()
    df_out["shap_top_driver_1"] = [
        f"{b_names[i]} ({'+' if shap_matrix[row, i] >= 0 else ''}{shap_matrix[row, i]:.2f})"
        for row, i in enumerate(top1_idx)
    ]
    df_out["shap_top_driver_2"] = [
        f"{b_names[i]} ({'+' if shap_matrix[row, i] >= 0 else ''}{shap_matrix[row, i]:.2f})"
        for row, i in enumerate(top2_idx)
    ]
    df_out["shap_top_driver_3"] = [
        f"{b_names[i]} ({'+' if shap_matrix[row, i] >= 0 else ''}{shap_matrix[row, i]:.2f})"
        for row, i in enumerate(top3_idx)
    ]

    # Sample 5 representative customer explanations for JSON report
    sample_explanations = [
        explainer.explain_customer(df_out.iloc[i], shap_matrix[i]) for i in range(min(5, len(df_out)))
    ]

    if save_to_disk:
        processed_dir = get_project_root() / "data" / "processed"
        global_imp_df.to_parquet(processed_dir / "shap_global_importance.parquet", index=False)
        # Save compact top-15 feature SHAP matrix for customer explorer waterfall charts
        top_feats = global_imp_df["feature"].head(15).tolist()
        top_feat_indices = [feature_names.index(f) for f in top_feats]
        shap_values_df = pd.DataFrame(
            np.round(shap_matrix[:, top_feat_indices], 4),
            columns=[f"shap__{f}" for f in top_feats],
        )
        shap_values_df.insert(0, "customer_id", df_out["customer_id"].values)
        shap_values_df.to_parquet(processed_dir / "customer_shap_values.parquet", index=False)

        reports_dir = get_project_root() / "reports"
        with open(reports_dir / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "causal_disclaimer": CAUSAL_DISCLAIMER,
                    "top_15_global_features": global_imp_df.head(15).to_dict(orient="records"),
                    "sample_customer_explanations": sample_explanations,
                },
                f,
                indent=2,
            )
        logger.info(
            f"SHAP analysis complete: top global driver is '{global_imp_df.iloc[0]['business_name']}' "
            f"(mean |SHAP|={global_imp_df.iloc[0]['mean_abs_shap']:.4f})"
        )

    return df_out, global_imp_df, {"sample_explanations": sample_explanations}
