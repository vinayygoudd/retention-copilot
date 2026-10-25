"""Feature Engineering and Temporal Aggregation Module for Retention Copilot.

Transforms raw multi-table customer, usage/billing, and support ticket records into
a rich analytical feature table while strictly enforcing `prediction_cutoff_date`
so zero post-cutoff records leak into feature vectors.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import get_logger

logger = get_logger("etl.feature_engineering")


NUMERIC_CORE_FEATURES: List[str] = [
    "tenure_months",
    "seats_licensed",
    "monthly_charges",
    "gross_margin_pct",
    "auto_pay_enabled",
    "is_month_to_month",
    "is_one_year",
    "is_two_year",
    "is_tier_basic",
    "is_tier_standard",
    "is_tier_enterprise",
    "is_seg_smb",
    "is_seg_enterprise",
    "active_seats_ratio_m0",
    "active_seats_ratio_mean_3m",
    "usage_drop_pct_m2_to_m0",
    "active_days_m0",
    "weekly_sessions_m0",
    "feature_adoption_m0",
    "api_calls_k_m0",
    "late_payment_count_3m",
    "payment_failures_3m",
    "billing_shock_ratio",
    "charge_per_active_seat",
    "ticket_count_90d",
    "ticket_max_severity_90d",
    "ticket_unresolved_count_90d",
    "ticket_mean_resolution_hours_90d",
    "ticket_min_csat_90d",
]


def engineer_customer_features(
    customers_df: pd.DataFrame,
    usage_billing_df: pd.DataFrame,
    support_tickets_df: pd.DataFrame,
    campaigns_outcomes_df: pd.DataFrame,
    cutoff_date: str = "2026-06-30",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate pre-cutoff history and engineer behavioral, billing, and ticket features.

    Returns:
        features_df: Customer-level modeling DataFrame with features, treatment, and outcomes.
        audit_counts: Dictionary of pre/post cutoff filtering audit counts.
    """
    # 1. Strict Temporal Cutoff Filtering (Target Leakage Prevention)
    total_usage_rows = len(usage_billing_df)
    valid_usage = usage_billing_df[usage_billing_df["snapshot_date"] <= cutoff_date].copy()
    dropped_usage_rows = total_usage_rows - len(valid_usage)

    total_ticket_rows = len(support_tickets_df)
    valid_tickets = support_tickets_df[support_tickets_df["created_at"] <= cutoff_date].copy()
    dropped_ticket_rows = total_ticket_rows - len(valid_tickets)

    logger.info(
        f"Temporal Leakage Guard (cutoff={cutoff_date}): "
        f"kept {len(valid_usage):,}/{total_usage_rows:,} usage rows (filtered {dropped_usage_rows:,} post-cutoff), "
        f"kept {len(valid_tickets):,}/{total_ticket_rows:,} tickets (filtered {dropped_ticket_rows:,} post-cutoff)"
    )

    # 2. Usage & Billing Aggregations across M-2, M-1, M0
    m0_date = valid_usage["snapshot_date"].max()
    m2_date = valid_usage["snapshot_date"].min()

    m0_df = valid_usage[valid_usage["snapshot_date"] == m0_date].set_index("customer_id")
    m2_df = valid_usage[valid_usage["snapshot_date"] == m2_date].set_index("customer_id")

    usage_agg = (
        valid_usage.groupby("customer_id", as_index=True)
        .agg(
            active_seats_ratio_mean_3m=("active_seats_ratio", "mean"),
            late_payment_count_3m=("late_payment_flag", "sum"),
            payment_failures_3m=("payment_failures", "sum"),
            billed_amount_mean_3m=("billed_amount", "mean"),
            billed_amount_max_3m=("billed_amount", "max"),
        )
    )

    # 3. Support Ticket Aggregations (Pre-Cutoff 90d Window)
    if len(valid_tickets) > 0:
        ticket_agg = (
            valid_tickets.groupby("customer_id", as_index=True)
            .agg(
                ticket_count_90d=("ticket_id", "count"),
                ticket_max_severity_90d=("severity", "max"),
                ticket_unresolved_count_90d=("is_unresolved", "sum"),
                ticket_mean_resolution_hours_90d=("resolution_time_hours", "mean"),
                ticket_min_csat_90d=("csat_score", "min"),
            )
        )
        # Concatenate pre-cutoff complaint texts per customer for downstream NLP module
        text_agg = (
            valid_tickets.sort_values("created_at")
            .groupby("customer_id")["complaint_text"]
            .apply(lambda s: " || ".join(s.tolist()))
            .rename("combined_complaint_text")
        )
    else:
        ticket_agg = pd.DataFrame()
        text_agg = pd.Series(dtype=str, name="combined_complaint_text")

    # 4. Assemble Master Feature Table
    df = customers_df.set_index("customer_id").copy()
    df["active_seats_ratio_m0"] = m0_df["active_seats_ratio"]
    df["active_seats_ratio_m2"] = m2_df["active_seats_ratio"]
    df["active_days_m0"] = m0_df["active_days_month"]
    df["weekly_sessions_m0"] = m0_df["weekly_sessions"]
    df["feature_adoption_m0"] = m0_df["feature_adoption_score"]
    df["api_calls_k_m0"] = m0_df["api_calls_k"]
    df["billed_amount_m0"] = m0_df["billed_amount"]

    df = df.join(usage_agg, how="left")
    df = df.join(ticket_agg, how="left")
    df = df.join(text_agg, how="left")

    # Fill defaults for customers with 0 support tickets
    df["ticket_count_90d"] = df["ticket_count_90d"].fillna(0).astype(int)
    df["ticket_max_severity_90d"] = df["ticket_max_severity_90d"].fillna(0).astype(int)
    df["ticket_unresolved_count_90d"] = df["ticket_unresolved_count_90d"].fillna(0).astype(int)
    df["ticket_mean_resolution_hours_90d"] = df["ticket_mean_resolution_hours_90d"].fillna(0.0).round(2)
    df["ticket_min_csat_90d"] = df["ticket_min_csat_90d"].fillna(5).astype(int)
    df["combined_complaint_text"] = df["combined_complaint_text"].fillna("")

    # 5. Domain-Engineered Ratios & Interaction Terms
    df["usage_drop_pct_m2_to_m0"] = np.round(
        np.clip(
            (df["active_seats_ratio_m2"] - df["active_seats_ratio_m0"])
            / np.maximum(df["active_seats_ratio_m2"], 0.05),
            -1.0,
            1.0,
        ),
        4,
    )
    df["billing_shock_ratio"] = np.round(
        df["billed_amount_max_3m"] / np.maximum(df["monthly_charges"], 1.0), 4
    )
    effective_active_seats = np.maximum(df["seats_licensed"] * df["active_seats_ratio_m0"], 0.5)
    df["charge_per_active_seat"] = np.round(df["monthly_charges"] / effective_active_seats, 2)

    # One-hot / binary indicators for interpretable linear & tree modeling
    df["is_month_to_month"] = (df["contract_type"] == "Month-to-Month").astype(int)
    df["is_one_year"] = (df["contract_type"] == "One-Year").astype(int)
    df["is_two_year"] = (df["contract_type"] == "Two-Year").astype(int)
    df["is_tier_basic"] = (df["plan_tier"] == "Basic").astype(int)
    df["is_tier_standard"] = (df["plan_tier"] == "Standard").astype(int)
    df["is_tier_enterprise"] = (df["plan_tier"] == "Enterprise").astype(int)
    df["is_seg_smb"] = (df["segment"] == "SMB").astype(int)
    df["is_seg_enterprise"] = (df["segment"] == "Enterprise").astype(int)

    # 6. Join Campaign Assignment & Post-Cutoff Outcomes (Targets only, never used as input features)
    outcomes_indexed = campaigns_outcomes_df.set_index("customer_id")[
        [
            "assigned_campaign",
            "treatment_flag",
            "intervention_cost",
            "churned",
            "retained",
            "true_baseline_churn_prob",
            "true_treated_churn_prob",
            "true_uplift_retention",
            "dominant_complaint_theme_gt",
        ]
    ]
    df = df.join(outcomes_indexed, how="inner").reset_index()

    audit_counts = {
        "total_customers": len(df),
        "total_usage_rows": total_usage_rows,
        "kept_pre_cutoff_usage_rows": len(valid_usage),
        "dropped_post_cutoff_usage_rows": dropped_usage_rows,
        "total_ticket_rows": total_ticket_rows,
        "kept_pre_cutoff_ticket_rows": len(valid_tickets),
        "dropped_post_cutoff_ticket_rows": dropped_ticket_rows,
    }
    return df, audit_counts
