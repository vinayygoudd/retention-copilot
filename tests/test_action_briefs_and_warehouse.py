"""Automated tests for AI Action Briefs, Claim Groundedness Validation, and Warehouse SQL."""

from __future__ import annotations

import json
import pandas as pd

from src.llm.action_brief_generator import (
    ActionBriefGenerator,
    build_customer_evidence_payload,
    validate_action_brief_claims,
)
from src.utils.config_loader import get_project_root


def test_action_brief_generation_and_claim_validation() -> None:
    root = get_project_root()
    decisions_df = pd.read_parquet(root / "data" / "processed" / "master_customer_decisions.parquet")

    gen = ActionBriefGenerator()
    # Test priority account brief
    row_top = decisions_df.sort_values("priority_rank").iloc[0]
    res_top = gen.generate_brief(row_top, prefer_live_api=False)
    assert res_top["validation"]["is_valid"] is True
    assert res_top["validation"]["checks_passed"] == 9
    assert len(res_top["validation"]["discrepancies"]) == 0

    # Test detection of fabricated/corrupted numerical claim
    evidence = build_customer_evidence_payload(row_top)
    corrupted_brief = res_top["brief_markdown"].replace(evidence["pred_churn_pct_str"], "99.9%")
    invalid_check = validate_action_brief_claims(corrupted_brief, evidence)
    assert invalid_check.is_valid is False
    assert len(invalid_check.discrepancies) >= 1


def test_warehouse_sql_validation_report() -> None:
    root = get_project_root()
    with open(root / "reports" / "warehouse_validation_report.json") as f:
        report = json.load(f)

    assert report["tables_loaded"]["CUSTOMERS"] == 20000
    assert report["tables_loaded"]["CUSTOMER_DECISIONS"] == 20000
    val_1 = report["validation_checks"][0]
    val_2 = report["validation_checks"][1]
    assert val_1["duplicate_customer_count"] == 0
    assert val_2["invalid_churn_probs"] == 0
    assert val_2["invalid_financial_values"] == 0
    assert val_2["unprofitable_budget_selections"] == 0
