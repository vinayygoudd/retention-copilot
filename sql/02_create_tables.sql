-- ============================================================================
-- 02_create_tables.sql
-- DDL Script for Raw and Curated Analytical Tables (Snowflake & SQLite Compatible)
-- ============================================================================

CREATE TABLE IF NOT EXISTS CUSTOMERS (
    customer_id VARCHAR(32) PRIMARY KEY,
    signup_date VARCHAR(16) NOT NULL,
    prediction_cutoff_date VARCHAR(16) NOT NULL,
    segment VARCHAR(32) NOT NULL,
    region VARCHAR(32) NOT NULL,
    industry VARCHAR(64) NOT NULL,
    plan_tier VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    payment_method VARCHAR(64) NOT NULL,
    auto_pay_enabled INTEGER NOT NULL,
    tenure_months INTEGER NOT NULL,
    seats_licensed INTEGER NOT NULL,
    monthly_charges FLOAT NOT NULL,
    gross_margin_pct FLOAT NOT NULL,
    expected_retained_value FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS USAGE_BILLING (
    customer_id VARCHAR(32) NOT NULL,
    snapshot_date VARCHAR(16) NOT NULL,
    active_seats_ratio FLOAT NOT NULL,
    active_days_month INTEGER NOT NULL,
    weekly_sessions INTEGER NOT NULL,
    feature_adoption_score FLOAT NOT NULL,
    api_calls_k FLOAT NOT NULL,
    billed_amount FLOAT NOT NULL,
    late_payment_flag INTEGER NOT NULL,
    payment_failures INTEGER NOT NULL,
    is_post_cutoff_record INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS SUPPORT_TICKETS (
    ticket_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    created_at VARCHAR(16) NOT NULL,
    ground_truth_theme VARCHAR(64) NOT NULL,
    severity INTEGER NOT NULL,
    resolution_time_hours FLOAT NOT NULL,
    is_unresolved INTEGER NOT NULL,
    csat_score INTEGER NOT NULL,
    complaint_text VARCHAR(2000) NOT NULL
);

CREATE TABLE IF NOT EXISTS CUSTOMER_DECISIONS (
    customer_id VARCHAR(32) PRIMARY KEY,
    dataset_split VARCHAR(16) NOT NULL,
    segment VARCHAR(32) NOT NULL,
    region VARCHAR(32) NOT NULL,
    industry VARCHAR(64) NOT NULL,
    plan_tier VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    tenure_months INTEGER NOT NULL,
    monthly_charges FLOAT NOT NULL,
    expected_retained_value FLOAT NOT NULL,
    active_seats_ratio_m0 FLOAT NOT NULL,
    usage_drop_pct_m2_to_m0 FLOAT NOT NULL,
    ticket_count_90d INTEGER NOT NULL,
    ticket_unresolved_count_90d INTEGER NOT NULL,
    complaint_theme_code VARCHAR(64) NOT NULL,
    complaint_theme_label VARCHAR(128) NOT NULL,
    assigned_campaign VARCHAR(64) NOT NULL,
    treatment_flag INTEGER NOT NULL,
    churned INTEGER NOT NULL,
    retained INTEGER NOT NULL,
    pred_churn_prob_logistic FLOAT NOT NULL,
    pred_churn_prob_xgb FLOAT NOT NULL,
    pred_churn_prob FLOAT NOT NULL,
    churn_risk_tier VARCHAR(32) NOT NULL,
    shap_top_driver_1 VARCHAR(128) NOT NULL,
    shap_top_driver_2 VARCHAR(128) NOT NULL,
    shap_top_driver_3 VARCHAR(128) NOT NULL,
    pred_uplift_retention FLOAT NOT NULL,
    true_uplift_retention FLOAT NOT NULL,
    uplift_quadrant VARCHAR(64) NOT NULL,
    recommended_action_code VARCHAR(64) NOT NULL,
    recommended_action_label VARCHAR(128) NOT NULL,
    recommended_offer_uplift FLOAT NOT NULL,
    recommended_intervention_cost FLOAT NOT NULL,
    expected_incremental_revenue FLOAT NOT NULL,
    expected_incremental_profit FLOAT NOT NULL,
    expected_roi_ratio FLOAT NOT NULL,
    is_eligible_for_intervention INTEGER NOT NULL,
    priority_rank INTEGER NOT NULL,
    selected_under_budget INTEGER NOT NULL
);
