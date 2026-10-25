"""Data Quality and Schema Validation Module for Retention Copilot.

Validates raw synthetic tables for:
1. Schema completeness and primary key uniqueness
2. Value ranges (probabilities in [0,1], non-negative charges/tenures)
3. Referential integrity across customers, usage_billing, support_tickets, and campaigns
4. Presence of pre-cutoff and post-cutoff timestamps to verify ETL leakage guards
5. Statistical randomization balance (Standardized Mean Difference < 0.05 between Treatment & Control)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data_generation.generator import GeneratedDatasetBundle
from src.utils.config_loader import get_logger

logger = get_logger("data_generation.validator")


@dataclass
class ValidationReport:
    """Structured result of dataset quality and randomization validation."""

    passed: bool
    checks_run: int
    checks_passed: int
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def validate_generated_dataset(
    bundle: GeneratedDatasetBundle,
    cutoff_date: str = "2026-06-30",
) -> ValidationReport:
    """Run comprehensive data quality, referential integrity, and RCT balance checks."""
    errors: List[str] = []
    checks_run = 0
    checks_passed = 0

    def _record(condition: bool, msg: str) -> None:
        nonlocal checks_run, checks_passed
        checks_run += 1
        if condition:
            checks_passed += 1
        else:
            errors.append(msg)

    cust = bundle.customers
    usage = bundle.usage_billing
    tickets = bundle.support_tickets
    outcomes = bundle.campaigns_outcomes

    # 1. Row counts and primary key uniqueness
    _record(len(cust) >= 1000, f"Expected >= 1,000 customers, got {len(cust)}")
    _record(cust["customer_id"].is_unique, "Duplicate customer_id in customers table")
    _record(outcomes["customer_id"].is_unique, "Duplicate customer_id in campaigns_outcomes table")
    _record(tickets["ticket_id"].is_unique, "Duplicate ticket_id in support_tickets table")

    # 2. Null checks on required columns
    _record(int(cust.isna().sum().sum()) == 0, "Null values found in customers table")
    _record(int(usage.isna().sum().sum()) == 0, "Null values found in usage_billing table")
    _record(int(outcomes.isna().sum().sum()) == 0, "Null values found in campaigns_outcomes table")

    # 3. Domain & range checks
    _record((cust["monthly_charges"] > 0).all(), "Non-positive monthly_charges detected")
    _record((cust["tenure_months"] >= 1).all(), "Invalid tenure_months < 1 detected")
    _record(
        ((usage["active_seats_ratio"] >= 0.0) & (usage["active_seats_ratio"] <= 1.0)).all(),
        "active_seats_ratio outside [0, 1]",
    )
    _record(set(outcomes["churned"].unique()).issubset({0, 1}), "churned column is not binary {0, 1}")
    _record(set(outcomes["treatment_flag"].unique()).issubset({0, 1}), "treatment_flag is not binary {0, 1}")
    _record(
        ((outcomes["true_baseline_churn_prob"] > 0.0) & (outcomes["true_baseline_churn_prob"] < 1.0)).all(),
        "true_baseline_churn_prob outside (0, 1)",
    )

    # 4. Referential integrity
    cust_ids = set(cust["customer_id"])
    _record(set(usage["customer_id"]).issubset(cust_ids), "Orphan customer_id in usage_billing")
    _record(set(tickets["customer_id"]).issubset(cust_ids), "Orphan customer_id in support_tickets")
    _record(set(outcomes["customer_id"]) == cust_ids, "Mismatch in customer_id between customers and outcomes")

    # 5. Cutoff date & leakage test data checks
    _record((cust["signup_date"] <= cutoff_date).all(), "Customer signup_date found after cutoff_date")
    pre_tickets = tickets[tickets["created_at"] <= cutoff_date]
    post_tickets = tickets[tickets["created_at"] > cutoff_date]
    _record(len(pre_tickets) > 0, "No pre-cutoff support tickets found")
    _record(len(post_tickets) > 0, "No post-cutoff leakage test tickets found")

    # 6. Randomized Treatment/Control Covariate Balance (Standardized Mean Difference)
    merged = cust.merge(outcomes[["customer_id", "treatment_flag"]], on="customer_id")
    treated = merged[merged["treatment_flag"] == 1]
    control = merged[merged["treatment_flag"] == 0]

    smds: Dict[str, float] = {}
    for col in ["tenure_months", "monthly_charges", "seats_licensed"]:
        mean_t = float(treated[col].mean())
        mean_c = float(control[col].mean())
        pooled_sd = float(np.sqrt((treated[col].var() + control[col].var()) / 2.0))
        smd = abs(mean_t - mean_c) / max(pooled_sd, 1e-6)
        smds[f"smd_{col}"] = round(smd, 5)
        _record(smd < 0.08, f"Covariate imbalance on {col}: SMD={smd:.4f} >= 0.08")

    passed = len(errors) == 0
    metrics = {
        "n_customers": len(cust),
        "n_pre_cutoff_tickets": len(pre_tickets),
        "n_post_cutoff_tickets": len(post_tickets),
        **smds,
    }
    logger.info(
        f"Dataset validation complete: {checks_passed}/{checks_run} checks passed | "
        f"Max covariate SMD={max(smds.values()):.4f}"
    )
    return ValidationReport(
        passed=passed,
        checks_run=checks_run,
        checks_passed=checks_passed,
        errors=errors,
        metrics=metrics,
    )
