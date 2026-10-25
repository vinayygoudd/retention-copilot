# System Architecture & Data Dictionary

**Project**: Retention Copilot — AI-Powered Customer Retention Platform  
**Prediction Cutoff Timestamp**: `2026-06-30` (All features strictly computed from <= `2026-06-30`; churn & campaign outcomes observed in 60-day post-cutoff window `2026-07-01` to `2026-08-29`).

## 1. End-to-End System Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                        1. SYNTHETIC DATA GENERATION & VALIDATION LAYER                            |
|  src/data_generation/generator.py  |  src/data_generation/validator.py                            |
|  - customers.parquet (N=20,000)    - usage_billing.parquet (N=80,000: 3 pre-cutoff + 1 post-cutoff)|
|  - support_tickets.parquet (22,556: 22,306 pre-cutoff + 250 post-cutoff leakage traps)            |
|  - campaigns_outcomes.parquet (N=20,000: Randomized Control + 3 Interventions, Ground-Truth CATE) |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                     2. PYSPARK / DATAFRAME ETL & TEMPORAL LEAKAGE FIREWALL                        |
|  src/etl/pyspark_pipeline.py       |  src/etl/feature_engineering.py                              |
|  - Enforces snapshot_date <= 2026-06-30 (drops 20,000 post-cutoff July usage rows)                |
|  - Enforces created_at <= 2026-06-30 (drops 250 post-cutoff exit/termination tickets)             |
|  - Computes 29 behavioral, billing, contract, and support velocity features                       |
|  - Deterministic stratified split: Train (13,999) | Validation (3,000) | Test (3,001)             |
+-----------------------+---------------------------------------------------+-----------------------+
                        |                                                   |
                        v                                                   v
+-----------------------------------------------+   +-----------------------------------------------+
|     3. NLP COMPLAINT INTELLIGENCE ENGINE      |   |       4. CHURN PREDICTION & EXPLAINABILITY    |
|  src/nlp/complaint_analyzer.py                |   |  src/models/churn_model.py                    |
|  - Sentence-Transformers / TF-IDF+SVD (16-d)  |-->|  - Logistic Regression vs. XGBoost / GBDT     |
|  - K-Means++ & Hungarian Bipartite Matching   |   |  - Ablation: Core Only vs. Core + 16 NLP Feats|
|  - 5 Distinct Themes + 16 Numeric NLP Features|   |  - Calibration Curves, Brier Score, ECE       |
+-----------------------------------------------+   |  src/explainability/shap_explainer.py         |
                                                    |  - Global Mean |SHAP| & Local Waterfall       |
                                                    +-----------------------+-----------------------+
                                                                            |
                                                                            v
+---------------------------------------------------------------------------------------------------+
|                   5. CAUSAL UPLIFT MODELING & STATISTICAL EVALUATION LAYER                        |
|  src/models/uplift_model.py                                                                       |
|  - Two-Model T-Learner: Control Estimator mu_0(x) vs. Treated Estimator mu_1(x)                   |
|  - Incremental Retention Probability: tau_hat(x) = mu_0(x) - mu_1(x)                              |
|  - SciPy / Statsmodels Two-Proportion Z-Tests, 95% CIs, Qini / AUUC, PEHE vs. Ground-Truth CATE   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                   6. DETERMINISTIC FINANCIAL DECISION & BUDGET OPTIMIZATION ENGINE                |
|  src/decision_engine/financial_engine.py                                                          |
|  - Formula: Expected Incremental Profit = Uplift * Expected Retained Contribution Value - Cost    |
|  - Multi-tier offer matching ($28 15% Discount, $45 Priority Support, $85 Executive Bundle)       |
|  - Eligibility guardrails & greedy knapsack budget allocation ($75,000 default budget)            |
+-----------------------+---------------------------------------------------+-----------------------+
                        |                                                   |
                        v                                                   v
+-----------------------------------------------+   +-----------------------------------------------+
|     7. SNOWFLAKE & SQL WAREHOUSE LAYER        |   |   8. AI ACTION BRIEFS & STREAMLIT DASHBOARD   |
|  src/warehouse/snowflake_client.py  |  sql/   |   |  src/llm/action_brief_generator.py            |
|  - Snowflake Cloud + Local SQLite Emulator    |   |  - OpenAI API + Deterministic Grounded Engine |
|  - 4 DDL Tables, Bulk Loader, 4 BI Queries    |   |  - Automated 9-Point Factual Claim Validator  |
|  - 3 Automated SQL Integrity & Budget Checks  |   |  app/streamlit_app.py (6 Interactive Pages)   |
+-----------------------------------------------+   +-----------------------------------------------+
```

## 2. Comprehensive Data Dictionary

### Table 1: `CUSTOMERS` (`data/raw/customers.parquet` — 20,000 rows)
- `customer_id` (`VARCHAR(32)` PK): Unique account ID (`CUST_000001` .. `CUST_020000`)
- `signup_date` (`VARCHAR(16)`): Subscription start date (<= `2026-06-30`)
- `prediction_cutoff_date` (`VARCHAR(16)`): Feature freeze date (`2026-06-30`)
- `segment` (`VARCHAR(32)`): `SMB`, `Mid_Market`, `Commercial`, `Enterprise`
- `region` (`VARCHAR(32)`): `North_America`, `EMEA`, `APAC`, `LATAM`
- `industry` (`VARCHAR(64)`): `SaaS_Tech`, `Financial_Services`, `Healthcare`, `Retail_Ecomm`, `Manufacturing`, `Professional_Services`
- `plan_tier` (`VARCHAR(32)`): `Basic` ($29 base), `Standard` ($69 base), `Enterprise` ($199 base)
- `contract_type` (`VARCHAR(32)`): `Month-to-Month`, `One-Year`, `Two-Year`
- `payment_method` (`VARCHAR(64)`): `Credit_Card_Auto`, `ACH_Bank_Transfer`, `Manual_Invoice`, `Digital_Wallet`
- `auto_pay_enabled` (`INTEGER`): `{0, 1}`
- `tenure_months` (`INTEGER`): `1` .. `72`
- `seats_licensed` (`INTEGER`): `1` .. `65`
- `monthly_charges` (`FLOAT`): Monthly Recurring Revenue (MRR) in USD
- `gross_margin_pct` (`FLOAT`): `0.72` (Basic), `0.78` (Standard), `0.84` (Enterprise)
- `expected_retained_value` (`FLOAT`): Expected contribution value if retained over post-retention horizon

### Table 2: `USAGE_BILLING` (`data/raw/usage_billing.parquet` — 80,000 rows)
- `customer_id`, `snapshot_date` (`2026-04-30`, `2026-05-31`, `2026-06-30`, plus post-cutoff `2026-07-31` dropped by ETL), `active_seats_ratio`, `active_days_month`, `weekly_sessions`, `feature_adoption_score`, `api_calls_k`, `billed_amount`, `late_payment_flag`, `payment_failures`, `is_post_cutoff_record`.

### Table 3: `SUPPORT_TICKETS` (`data/raw/support_tickets.parquet` — 22,556 rows)
- `ticket_id`, `customer_id`, `created_at` (22,306 pre-cutoff <= `2026-06-30` + 250 post-cutoff > `2026-06-30` dropped by ETL), `ground_truth_theme`, `severity` (`1..5`), `resolution_time_hours`, `is_unresolved` (`{0,1}`), `csat_score` (`1..5`), `complaint_text`.

### Table 4: `CUSTOMER_DECISIONS` (`data/processed/master_customer_decisions.parquet` — 20,000 rows)
- Contains all 29 core features, 16 NLP features, `dataset_split` (`train`/`val`/`test`), `pred_churn_prob_logistic`, `pred_churn_prob_xgb`, `pred_churn_prob`, `churn_risk_tier`, `shap_top_driver_1..3`, `complaint_theme_code`, `complaint_theme_label`, `pred_uplift_retention`, `true_uplift_retention`, `uplift_quadrant`, `recommended_action_code`, `recommended_action_label`, `recommended_offer_uplift`, `recommended_intervention_cost`, `expected_incremental_revenue`, `expected_incremental_profit`, `expected_roi_ratio`, `is_eligible_for_intervention`, `priority_rank`, `selected_under_budget`.
