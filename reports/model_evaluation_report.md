# Model Evaluation & Explainability Report

**Dataset**: Synthetic SaaS Portfolio (`N = 20,000` customers; `Train = 13,999`, `Validation = 3,000`, `Test = 3,001`, Cutoff `2026-06-30`)

## 1. Churn Model Comparison (With vs. Without NLP Complaint Features)

| Model Configuration | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1-Score | Test Accuracy | Test Brier Score | Test ECE (10-Bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Core Only, No NLP)** | 0.8025 | 0.8040 | 0.7775 | 0.7251 | 0.6527 | 0.6870 | 0.7438 | 0.1762 | 0.0261 |
| **Logistic Regression (Core + NLP Features)** | 0.8029 | 0.8033 | 0.7771 | 0.7236 | 0.6520 | 0.6859 | 0.7428 | 0.1764 | 0.0275 |
| **XGBoost / GBDT (Core Only, No NLP)** | 0.8134 | 0.8164 | 0.7977 | 0.7449 | 0.6551 | 0.6971 | 0.7547 | 0.1690 | 0.0188 |
| **XGBoost / GBDT (Core + NLP Features — Champion)** | **0.8135** | **0.8168** | **0.7971** | **0.7389** | **0.6543** | **0.6940** | **0.7514** | **0.1688** | **0.0196** |

## 2. Global SHAP Feature Importance (Top 10 Drivers)

| Rank | Feature Code | Business Feature Name | Mean |SHAP| (Log-Odds) | Relative Importance (%) | Risk Direction |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `is_month_to_month` | Month-to-Month Contract | 0.5715 | 28.43% | Increases Churn Risk (+) |
| 2 | `ticket_min_csat_90d` | Minimum Support CSAT Score (1-5) | 0.3129 | 15.56% | Reduces Churn Risk (-) |
| 3 | `usage_drop_pct_m2_to_m0` | 90-Day Usage Decline (%) | 0.2498 | 12.43% | Increases Churn Risk (+) |
| 4 | `active_seats_ratio_m0` | Current Seat Utilization Ratio | 0.2191 | 10.90% | Reduces Churn Risk (-) |
| 5 | `is_two_year` | Two-Year Multi-Year Contract | 0.1450 | 7.21% | Reduces Churn Risk (-) |
| 6 | `tenure_months` | Account Tenure (Months) | 0.1388 | 6.90% | Reduces Churn Risk (-) |
| 7 | `ticket_unresolved_count_90d` | Unresolved Support Tickets Count | 0.1118 | 5.56% | Increases Churn Risk (+) |
| 8 | `late_payment_count_3m` | Late Payments (Last 3 Months) | 0.0722 | 3.59% | Increases Churn Risk (+) |
| 9 | `ticket_max_severity_90d` | Max Support Ticket Severity (1-5) | 0.0566 | 2.82% | Increases Churn Risk (+) |
| 10 | `active_seats_ratio_mean_3m` | 3-Month Avg Seat Utilization | 0.0445 | 2.21% | Reduces Churn Risk (-) |

## 3. NLP Complaint Clusters (`N = 12,776` Complaining Customers + `7,224` No-Complaint Baseline)

| Cluster ID | Theme Code | Business Theme Label | Customer Count | Avg Tickets (90d) | Avg Max Severity | Unresolved Rate | Observed Churn Rate |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4 | `technical_outage_and_bugs` | Platform Outages, API Latency & Sync Bugs | 984 | 2.06 | 3.40 | 46.44% | **57.62%** |
| 2 | `missing_features_and_ux` | Missing Integrations, Reporting UX & Admin Controls | 3,080 | 1.71 | 3.27 | 40.91% | **55.36%** |
| 1 | `account_and_contract_flexibility` | Rigid Contract Lock-In & Seat Reduction Policies | 2,763 | 1.75 | 3.34 | 42.27% | **54.29%** |
| 0 | `billing_and_pricing` | Billing Surprises, Overage Fees & Renewal Pricing | 2,585 | 1.76 | 3.35 | 41.82% | **53.73%** |
| 3 | `support_responsiveness` | Slow Support SLA, Escalations & Unresolved Tickets | 3,364 | 1.67 | 3.31 | 41.11% | **53.18%** |
| -1 | `no_complaint` | No Pre-Cutoff Complaints (Baseline Cohort) | 7,224 | 0.00 | 0.00 | 0.00% | **23.16%** |
