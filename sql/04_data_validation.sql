-- ============================================================================
-- 04_data_validation.sql
-- Automated Warehouse Data Quality, Leakage, and Referential Integrity Checks
-- ============================================================================

-- Validation 1: Row counts and Primary Key uniqueness check on CUSTOMER_DECISIONS
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS distinct_customers,
    COUNT(*) - COUNT(DISTINCT customer_id) AS duplicate_customer_count
FROM CUSTOMER_DECISIONS;

-- Validation 2: Verify probability bounds and non-negative financial metrics
SELECT
    SUM(CASE WHEN pred_churn_prob < 0.0 OR pred_churn_prob > 1.0 THEN 1 ELSE 0 END) AS invalid_churn_probs,
    SUM(CASE WHEN monthly_charges <= 0 OR expected_retained_value <= 0 THEN 1 ELSE 0 END) AS invalid_financial_values,
    SUM(CASE WHEN selected_under_budget = 1 AND expected_incremental_profit <= 0 THEN 1 ELSE 0 END) AS unprofitable_budget_selections
FROM CUSTOMER_DECISIONS;

-- Validation 3: Verify Total Budget Spend Constraint ($75,000 limit)
SELECT
    SUM(CASE WHEN selected_under_budget = 1 THEN 1 ELSE 0 END) AS total_selected_accounts,
    ROUND(SUM(CASE WHEN selected_under_budget = 1 THEN recommended_intervention_cost ELSE 0 END), 2) AS total_budget_allocated,
    ROUND(SUM(CASE WHEN selected_under_budget = 1 THEN expected_incremental_profit ELSE 0 END), 2) AS total_expected_net_profit
FROM CUSTOMER_DECISIONS;
