-- ============================================================================
-- 03_analytical_queries.sql
-- Executive, Segment, Complaint, and Campaign ROI Analytical Queries
-- ============================================================================

-- Query 1: Executive Churn Risk & Expected Value at Risk by Segment and Plan Tier
SELECT
    segment,
    plan_tier,
    COUNT(*) AS total_customers,
    ROUND(AVG(monthly_charges), 2) AS avg_mrr,
    ROUND(AVG(pred_churn_prob), 4) AS avg_predicted_churn_prob,
    ROUND(AVG(churned), 4) AS observed_churn_rate,
    ROUND(SUM(pred_churn_prob * expected_retained_value), 2) AS expected_ltv_at_risk,
    SUM(selected_under_budget) AS targeted_under_budget_count,
    ROUND(SUM(CASE WHEN selected_under_budget = 1 THEN expected_incremental_profit ELSE 0 END), 2) AS expected_net_profit_saved
FROM CUSTOMER_DECISIONS
GROUP BY segment, plan_tier
ORDER BY expected_ltv_at_risk DESC;

-- Query 2: NLP Complaint Theme Impact on Churn & Uplift Persuadability
SELECT
    complaint_theme_label,
    COUNT(*) AS customer_count,
    ROUND(AVG(ticket_count_90d), 2) AS avg_tickets_90d,
    ROUND(AVG(pred_churn_prob), 4) AS avg_pred_churn_prob,
    ROUND(AVG(churned), 4) AS observed_churn_rate,
    ROUND(AVG(pred_uplift_retention), 4) AS avg_estimated_uplift,
    ROUND(SUM(expected_incremental_profit), 2) AS total_potential_net_profit
FROM CUSTOMER_DECISIONS
GROUP BY complaint_theme_label
ORDER BY observed_churn_rate DESC;

-- Query 3: Randomized Controlled Trial (RCT) Empirical Campaign Performance
SELECT
    assigned_campaign,
    treatment_flag,
    COUNT(*) AS cohort_size,
    ROUND(AVG(retained), 4) AS empirical_retention_rate,
    ROUND(AVG(churned), 4) AS empirical_churn_rate,
    ROUND(AVG(true_uplift_retention), 4) AS mean_ground_truth_uplift
FROM CUSTOMER_DECISIONS
GROUP BY assigned_campaign, treatment_flag
ORDER BY treatment_flag ASC, empirical_retention_rate DESC;

-- Query 4: Top 25 Priority Intervention Accounts under Budget Constraint
SELECT
    priority_rank,
    customer_id,
    segment,
    plan_tier,
    monthly_charges,
    expected_retained_value,
    ROUND(pred_churn_prob, 4) AS pred_churn_prob,
    ROUND(pred_uplift_retention, 4) AS pred_uplift_retention,
    complaint_theme_label,
    recommended_action_label,
    recommended_intervention_cost,
    expected_incremental_profit
FROM CUSTOMER_DECISIONS
WHERE selected_under_budget = 1
ORDER BY priority_rank ASC
LIMIT 25;
