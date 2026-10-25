# Uplift Modeling & Financial Campaign Policy Evaluation Report

**Dataset**: Synthetic Randomized Controlled Trial (`N = 20,000` accounts; Control `N = 6,746`, Treated `N = 13,254`; Held-Out Test Set `N = 3,001`)

## 1. Randomized Controlled Trial (RCT) Statistical Evaluation

| Campaign Intervention Tier | Unit Cost ($) | Treated Cohort (N) | Control Cohort (N) | Treated Retention Rate | Control Retention Rate | Empirical Retention Lift (ATE) | 95% Confidence Interval | Z-Statistic | Two-Sided P-Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Priority Support & Onboarding (`priority_support`)** | $45.00 | 4,397 | 6,746 | 60.93% | 51.32% | **+9.61%** | `[+7.74%, +11.48%]` | 10.063 | $< 10^{-15}$ |
| **Executive Account Review + Credit (`executive_bundle`)** | $85.00 | 4,365 | 6,746 | 59.43% | 51.32% | **+8.11%** | `[+6.23%, +9.99%]` | 8.441 | $< 10^{-15}$ |
| **15% Promotional Discount (`discount_15pct`)** | $28.00 | 4,492 | 6,746 | 58.82% | 51.32% | **+7.50%** | `[+5.63%, +9.37%]` | 7.860 | $3.78 \times 10^{-15}$ |
| **Held-Out Test Set Overall RCT (`test` split)** | Mixed | 1,989 | 1,012 | 59.73% | 51.38% | **+8.35%** | `[+4.59%, +12.10%]` | 4.351 | $1.35 \times 10^{-5}$ |

## 2. T-Learner Uplift Estimation Quality (Held-Out Test Split, `N = 3,001`)
- **PEHE (RMSE vs. Ground-Truth CATE)**: `0.0610`
- **Pearson Correlation with Ground-Truth Uplift**: `r = 0.6069` (`Spearman rho = 0.6600`)
- **AUUC (Expected Incremental Saves per 1,000 Accounts)**:
  - **Oracle True Uplift**: `64.587`
  - **T-Learner Uplift Targeting**: **`57.723`** (+17.5% over Churn-Risk Targeting, +29.3% over Random Targeting)
  - **Naive Churn-Risk Targeting**: `49.129`
  - **Random Targeting Baseline**: `44.642`

## 3. Financial Policy Comparison under `$75,000` Campaign Budget (`N = 20,000` Portfolio)

| Targeting Policy | Budget Limit ($) | Accounts Targeted | Total Spend ($) | Expected Incremental Saves | Ground-Truth Realized Saves | Expected Incremental Revenue ($) | Expected Net Incremental Profit ($) | Ground-Truth Realized Net Profit ($) | Expected ROI (%) | Ground-Truth Realized ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Profit-Optimized Uplift + Offer Matching (Recommended)** | $75,000 | **893** | $74,985 | 113.52 | 79.48 | $1,093,504.96 | **$1,018,519.96** | **$733,701.20** | **1,358.3%** | **978.5%** |
| **2. Pure Uplift Targeting (Top Incremental Retention Prob)** | $75,000 | 1,666 | $74,970 | **322.76** | **267.33** | $343,007.85 | $268,037.85 | $192,716.21 | 357.5% | 257.1% |
| **3. Naive Churn-Risk Targeting (Top Churn Probability)** | $75,000 | 1,666 | $74,970 | 53.38 | 44.30 | $289,202.69 | $214,232.69 | $167,383.43 | 285.8% | 223.3% |
| **4. High-LTV Account Targeting (Top Customer Value)** | $75,000 | 1,666 | $74,970 | 70.42 | 71.94 | $886,157.27 | $811,187.27 | $840,120.43 | 1,082.0% | 1,120.6% |
| **5. Random Targeting (Untargeted Campaign Blast)** | $75,000 | 1,666 | $74,970 | 129.10 | 153.30 | $217,337.81 | $142,367.81 | $158,363.27 | 189.9% | 211.2% |
