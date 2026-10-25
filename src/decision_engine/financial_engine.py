"""Financial Decision Engine for Retention Copilot.

Implements deterministic, auditable financial calculations for customer retention:
    Expected Incremental Profit =
        Estimated Incremental Retention Probability * Expected Contribution Value if Retained
        - Intervention Cost

Features:
1. Exact unit-testable formula functions (`calculate_expected_incremental_profit`)
2. Multi-tier intervention matching (evaluating which campaign tier maximizes net expected
   incremental profit for each customer given their complaint theme and uplift profile)
3. Customer eligibility rules (minimum churn risk, minimum positive uplift, sleeping-dog
   protection, and positive net profit threshold)
4. Budget-constrained prioritization and policy simulation comparing:
   - **Profit-Optimized Uplift Policy** (Greedy knapsack by ROI / Expected Net Profit)
   - **Pure Uplift Policy** (Ranked by incremental retention probability)
   - **Churn-Risk Policy** (Ranked by predicted churn probability — common naive practice)
   - **Random Targeting Policy** (Uniform allocation up to budget)
   - **No Intervention Baseline**
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import get_logger, get_project_root, load_config

logger = get_logger("decision_engine.financial_engine")


def calculate_expected_incremental_profit(
    incremental_retention_prob: float | np.ndarray,
    expected_contribution_value: float | np.ndarray,
    intervention_cost: float | np.ndarray,
) -> float | np.ndarray:
    """Deterministic expected incremental profit formula required by specification:

    Expected incremental profit =
        Estimated incremental retention probability * Expected contribution value if retained
        - Intervention cost
    """
    return incremental_retention_prob * expected_contribution_value - intervention_cost


def evaluate_interventions_for_customers(
    df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate all available retention offers per customer and select the profit-optimal action."""
    if config is None:
        config = load_config()

    fin_cfg = config["financial_engine"]
    min_churn = float(fin_cfg.get("min_churn_risk_threshold", 0.15))
    min_uplift = float(fin_cfg.get("min_uplift_threshold", 0.01))
    min_profit = float(fin_cfg.get("min_expected_profit", 5.0))

    df_out = df.copy()
    base_uplift = df_out["pred_uplift_retention"].to_numpy(dtype=np.float64)
    churn_prob = df_out["pred_churn_prob"].to_numpy(dtype=np.float64)
    ltv = df_out["expected_retained_value"].to_numpy(dtype=np.float64)
    theme = df_out["complaint_theme_code"].astype(str).to_numpy()

    # Offer-specific uplift multipliers aligned with customer complaint theme
    mult_discount = np.where(theme == "billing_and_pricing", 1.15, 0.85)
    mult_support = np.where(
        np.isin(theme, ["support_responsiveness", "technical_outage_and_bugs"]), 1.20, 0.95
    )
    mult_exec = np.full(len(df_out), 1.28, dtype=np.float64)

    uplift_discount = np.clip(base_uplift * mult_discount, -0.08, churn_prob * 0.90)
    uplift_support = np.clip(base_uplift * mult_support, -0.08, churn_prob * 0.90)
    uplift_exec = np.clip(base_uplift * mult_exec, -0.08, churn_prob * 0.90)

    cost_discount = 28.0
    cost_support = 45.0
    cost_exec = 85.0

    profit_discount = calculate_expected_incremental_profit(uplift_discount, ltv, cost_discount)
    profit_support = calculate_expected_incremental_profit(uplift_support, ltv, cost_support)
    profit_exec = calculate_expected_incremental_profit(uplift_exec, ltv, cost_exec)

    # Stack candidate offers: 0=discount_15pct, 1=priority_support, 2=executive_bundle
    offer_codes = np.array(["discount_15pct", "priority_support", "executive_bundle"])
    offer_labels = np.array([
        "15% Promotional Discount (3 Months)",
        "Dedicated Priority Support & Onboarding",
        "Executive Account Review + Custom Credit",
    ])
    offer_costs = np.array([cost_discount, cost_support, cost_exec])

    profits_matrix = np.column_stack([profit_discount, profit_support, profit_exec])
    uplifts_matrix = np.column_stack([uplift_discount, uplift_support, uplift_exec])

    best_offer_idx = np.argmax(profits_matrix, axis=1)
    row_indices = np.arange(len(df_out))

    best_profit = profits_matrix[row_indices, best_offer_idx]
    best_uplift = uplifts_matrix[row_indices, best_offer_idx]
    best_cost = offer_costs[best_offer_idx]
    best_code = offer_codes[best_offer_idx]
    best_label = offer_labels[best_offer_idx]

    # Eligibility rules:
    # 1. Churn risk >= min_churn
    # 2. Recommended offer uplift >= min_uplift (protects Sleeping Dogs)
    # 3. Expected incremental profit >= min_profit
    is_eligible = (
        (churn_prob >= min_churn)
        & (best_uplift >= min_uplift)
        & (best_profit >= min_profit)
    )

    df_out["recommended_action_code"] = np.where(is_eligible, best_code, "no_intervention")
    df_out["recommended_action_label"] = np.where(
        is_eligible, best_label, "No Intervention (Monitor / Hold)"
    )
    df_out["recommended_offer_uplift"] = np.round(np.where(is_eligible, best_uplift, 0.0), 5)
    df_out["recommended_intervention_cost"] = np.round(np.where(is_eligible, best_cost, 0.0), 2)
    df_out["expected_incremental_revenue"] = np.round(
        np.where(is_eligible, best_uplift * ltv, 0.0), 2
    )
    df_out["expected_incremental_profit"] = np.round(
        np.where(is_eligible, best_profit, 0.0), 2
    )
    df_out["expected_roi_ratio"] = np.round(
        np.where(is_eligible & (best_cost > 0), best_profit / best_cost, 0.0), 4
    )
    df_out["is_eligible_for_intervention"] = is_eligible.astype(int)

    # Also compute deterministic profit at a fixed standard intervention ($45 Priority Support)
    # so policy comparisons under uniform intervention cost are apples-to-apples
    df_out["std_offer_cost"] = 45.0
    df_out["std_offer_expected_profit"] = np.round(
        calculate_expected_incremental_profit(base_uplift, ltv, 45.0), 2
    )
    df_out["std_offer_true_profit"] = np.round(
        calculate_expected_incremental_profit(
            df_out["true_uplift_retention"].to_numpy(dtype=np.float64), ltv, 45.0
        ),
        2,
    )

    return df_out


def simulate_budget_constrained_policies(
    df: pd.DataFrame,
    budget: float = 75000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare retention targeting strategies under a fixed campaign budget.

    Returns:
        prioritized_df: Customer table with `selected_under_budget` flag and priority rank.
        policy_comparison_df: Summary table comparing 5 targeting policies on expected
            and ground-truth realized financial outcomes.
    """
    df_out = df.copy()

    # 1. Rank eligible customers by Expected Incremental Profit for the primary recommendation
    eligible_mask = df_out["is_eligible_for_intervention"] == 1
    df_out["priority_score"] = np.where(eligible_mask, df_out["expected_incremental_profit"], -1e9)
    order_profit = np.argsort(-df_out["priority_score"].to_numpy())

    ranks = np.empty(len(df_out), dtype=int)
    ranks[order_profit] = np.arange(1, len(df_out) + 1)
    df_out["priority_rank"] = ranks

    # Greedy budget allocation on eligible customers
    sorted_costs = df_out["recommended_intervention_cost"].to_numpy()[order_profit]
    sorted_eligible = eligible_mask.to_numpy()[order_profit]
    cum_spend = np.cumsum(np.where(sorted_eligible, sorted_costs, 0.0))
    within_budget_sorted = sorted_eligible & (cum_spend <= budget)

    selected_flag = np.zeros(len(df_out), dtype=int)
    selected_flag[order_profit] = within_budget_sorted.astype(int)
    df_out["selected_under_budget"] = selected_flag

    # 2. Policy Comparison Simulation under `budget`
    rng = np.random.default_rng(42)
    random_order = rng.permutation(len(df_out))

    policy_definitions: List[Tuple[str, np.ndarray, str]] = [
        (
            "1. Profit-Optimized Uplift + Offer Matching (Recommended)",
            order_profit,
            "dynamic_offer",
        ),
        (
            "2. Pure Uplift Targeting (Top Incremental Retention Prob)",
            np.argsort(-df_out["pred_uplift_retention"].to_numpy()),
            "standard_offer",
        ),
        (
            "3. Naive Churn-Risk Targeting (Top Churn Probability)",
            np.argsort(-df_out["pred_churn_prob"].to_numpy()),
            "standard_offer",
        ),
        (
            "4. High-LTV Account Targeting (Top Customer Value)",
            np.argsort(-df_out["expected_retained_value"].to_numpy()),
            "standard_offer",
        ),
        (
            "5. Random Targeting (Untargeted Campaign Blast)",
            random_order,
            "standard_offer",
        ),
    ]

    policy_rows = []
    for pol_name, idx_order, mode in policy_definitions:
        sub = df_out.iloc[idx_order]
        if mode == "dynamic_offer":
            elig = sub["is_eligible_for_intervention"].to_numpy() == 1
            costs = np.where(elig, sub["recommended_intervention_cost"].to_numpy(), 0.0)
            c_spend = np.cumsum(costs)
            chosen = elig & (c_spend <= budget)
            targeted = sub[chosen]

            total_cost = float(targeted["recommended_intervention_cost"].sum())
            exp_saves = float(targeted["recommended_offer_uplift"].sum())
            true_saves = float(targeted["true_uplift_retention"].sum())
            exp_rev = float(targeted["expected_incremental_revenue"].sum())
            exp_profit = float(targeted["expected_incremental_profit"].sum())
            true_rev = float((targeted["true_uplift_retention"] * targeted["expected_retained_value"]).sum())
            true_profit = true_rev - total_cost
        else:
            # Standard $45 offer applied until budget is exhausted
            costs = np.full(len(sub), 45.0)
            c_spend = np.cumsum(costs)
            chosen = c_spend <= budget
            targeted = sub[chosen]

            total_cost = float(len(targeted) * 45.0)
            exp_saves = float(targeted["pred_uplift_retention"].sum())
            true_saves = float(targeted["true_uplift_retention"].sum())
            exp_rev = float((targeted["pred_uplift_retention"] * targeted["expected_retained_value"]).sum())
            exp_profit = exp_rev - total_cost
            true_rev = float((targeted["true_uplift_retention"] * targeted["expected_retained_value"]).sum())
            true_profit = true_rev - total_cost

        roi_pct = (100.0 * exp_profit / total_cost) if total_cost > 0 else 0.0
        true_roi_pct = (100.0 * true_profit / total_cost) if total_cost > 0 else 0.0

        policy_rows.append({
            "policy_name": pol_name,
            "budget_limit": round(budget, 2),
            "customers_targeted": int(len(targeted)),
            "total_intervention_spend": round(total_cost, 2),
            "expected_incremental_saves": round(exp_saves, 2),
            "ground_truth_incremental_saves": round(true_saves, 2),
            "expected_incremental_revenue": round(exp_rev, 2),
            "expected_net_incremental_profit": round(exp_profit, 2),
            "ground_truth_net_incremental_profit": round(true_profit, 2),
            "expected_roi_pct": round(roi_pct, 2),
            "ground_truth_roi_pct": round(true_roi_pct, 2),
        })

    policy_comparison_df = pd.DataFrame(policy_rows)
    return df_out, policy_comparison_df


def run_financial_decision_engine(
    df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    save_to_disk: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run full financial scoring, eligibility filtering, and budget policy simulation."""
    if config is None:
        config = load_config()

    budget = float(config["financial_engine"].get("default_budget", 75000.0))
    scored_df = evaluate_interventions_for_customers(df, config=config)
    prioritized_df, policy_comparison_df = simulate_budget_constrained_policies(
        scored_df, budget=budget
    )

    # Also evaluate policies across a grid of budgets ($15k to $200k) for dashboard sensitivity curve
    budget_grid = [15000.0, 30000.0, 50000.0, 75000.0, 100000.0, 150000.0, 200000.0]
    frontier_rows = []
    for b_val in budget_grid:
        _, comp_b = simulate_budget_constrained_policies(scored_df, budget=b_val)
        for _, r in comp_b.iterrows():
            frontier_rows.append(r.to_dict())
    budget_frontier_df = pd.DataFrame(frontier_rows)

    summary = {
        "default_budget": budget,
        "total_eligible_customers": int(prioritized_df["is_eligible_for_intervention"].sum()),
        "selected_customers_under_budget": int(prioritized_df["selected_under_budget"].sum()),
        "policy_comparison_at_default_budget": policy_comparison_df.to_dict(orient="records"),
    }

    if save_to_disk:
        processed_dir = get_project_root() / "data" / "processed"
        prioritized_df.to_parquet(processed_dir / "master_customer_decisions.parquet", index=False)
        policy_comparison_df.to_parquet(processed_dir / "policy_comparison.parquet", index=False)
        budget_frontier_df.to_parquet(processed_dir / "budget_efficiency_frontier.parquet", index=False)

        reports_dir = get_project_root() / "reports"
        with open(reports_dir / "financial_policy_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        best_row = policy_comparison_df.iloc[0]
        naive_row = policy_comparison_df.iloc[2]
        logger.info(
            f"Financial Engine complete (Budget=${budget:,.0f}): "
            f"Profit-Optimized Policy yields ${best_row['expected_net_incremental_profit']:,.2f} expected net profit "
            f"(GT=${best_row['ground_truth_net_incremental_profit']:,.2f}, ROI={best_row['expected_roi_pct']:.1f}%) "
            f"vs Naive Churn-Risk Policy ${naive_row['expected_net_incremental_profit']:,.2f} "
            f"(GT=${naive_row['ground_truth_net_incremental_profit']:,.2f})."
        )

    return prioritized_df, policy_comparison_df, summary
