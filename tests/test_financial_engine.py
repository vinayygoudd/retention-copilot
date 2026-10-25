"""Automated tests for the deterministic Financial Decision Engine (`src/decision_engine/`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine.financial_engine import (
    calculate_expected_incremental_profit,
    simulate_budget_constrained_policies,
)
from src.utils.config_loader import get_project_root


def test_deterministic_expected_incremental_profit_formula() -> None:
    # Exact formula check: Profit = Uplift * Value_if_retained - Cost
    profit_scalar = calculate_expected_incremental_profit(
        incremental_retention_prob=0.15,
        expected_contribution_value=1000.0,
        intervention_cost=45.0,
    )
    assert abs(float(profit_scalar) - 105.0) < 1e-9

    # Vectorized check
    uplifts = np.array([0.0, 0.10, 0.20, -0.05])
    values = np.array([500.0, 800.0, 1500.0, 1000.0])
    costs = np.array([28.0, 45.0, 85.0, 28.0])
    expected = np.array([-28.0, 35.0, 215.0, -78.0])
    actual = calculate_expected_incremental_profit(uplifts, values, costs)
    assert np.allclose(actual, expected)


def test_budget_constraint_and_policy_dominance() -> None:
    root = get_project_root()
    decisions_df = pd.read_parquet(root / "data" / "processed" / "master_customer_decisions.parquet")

    budget = 50000.0
    prioritized_df, policy_comp_df = simulate_budget_constrained_policies(decisions_df, budget=budget)

    selected = prioritized_df[prioritized_df["selected_under_budget"] == 1]
    total_spend = float(selected["recommended_intervention_cost"].sum())

    # Spend must strictly respect budget constraint
    assert total_spend <= budget
    # All selected accounts must have positive expected incremental profit
    assert (selected["expected_incremental_profit"] > 0).all()
    # Profit-optimized policy must beat naive churn-risk and random targeting on expected net profit
    opt_profit = float(policy_comp_df.iloc[0]["expected_net_incremental_profit"])
    naive_profit = float(policy_comp_df.iloc[2]["expected_net_incremental_profit"])
    rand_profit = float(policy_comp_df.iloc[4]["expected_net_incremental_profit"])
    assert opt_profit > naive_profit
    assert opt_profit > rand_profit
