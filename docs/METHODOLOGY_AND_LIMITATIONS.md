# Methodology, Statistical Assumptions & Limitations

**Project**: Retention Copilot — AI-Powered Customer Retention Platform  

## 1. Temporal Leakage Prevention & Experimental Design
- **Explicit Cutoff Timestamp (`2026-06-30`)**: Every customer's feature vector is aggregated strictly from telemetry snapshots (`snapshot_date <= 2026-06-30`) and support tickets (`created_at <= 2026-06-30`).
- **Deliberate Leakage Traps & Automated Audits**: The synthetic generator injects 20,000 July 2026 usage records (`2026-07-31`) and 250 July 2026 exit tickets (`created_at > 2026-06-30`). The ETL pipeline filters out 100% of post-cutoff records, and `tests/test_etl_leakage.py` verifies that zero post-cutoff phrases or timestamps enter the feature store.

## 2. Predictive Churn Modeling & Calibration
- **L2-Regularized Logistic Regression vs. Second-Order Gradient Boosted Decision Trees (XGBoost)**: Evaluated on ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score, and 10-bin Expected Calibration Error (ECE).
- **Calibration for Financial Decisioning**: Predicted probabilities are well-calibrated (`ECE = 0.0196`, `Brier = 0.1688` for XGBoost), ensuring unbiased dollar expected-value calculations.

## 3. Explainable AI (SHAP): Predictive Associations vs. Causal Effects
- **Governance Principle**: SHAP values quantify how strongly each feature shifts predicted churn log-odds based on observational correlations. They do **not** prove that directly altering a feature will causally prevent churn. Causal campaign decisions are governed by the T-Learner Uplift model.

## 4. Causal Uplift Modeling (T-Learner) & Identification Assumptions
- **Formulation**: $\hat{\tau}(x) = \hat{P}(\text{Retained}=1 \mid X=x, T=1) - \hat{P}(\text{Retained}=1 \mid X=x, T=0) = \hat{\mu}_0(x) - \hat{\mu}_1(x)$.
- **Identification Assumptions**:
  1. **Conditional Ignorability (Unconfoundedness)**: $(Y(1), Y(0)) \perp T \mid X$ (verified via RCT covariate balance $\text{SMD} \le 0.0133$).
  2. **SUTVA (Stable Unit Treatment Value Assumption)**: No interference across customer accounts and well-defined intervention tiers.
  3. **Positivity (Common Support)**: $0 < P(T=1 \mid X=x) < 1$ across all segments.
- **T-Learner Statistical Limitations**: Independent estimation of $\hat{\mu}_0$ and $\hat{\mu}_1$ can compound variance when taking their difference; mitigated via leaf sample floors (`min_samples_leaf=30`), L2 regularization, and empirical Bayes shrinkage.

## 5. Financial Decision Engine & Synthetic Data Disclaimer
- **Deterministic Profit Equation**: $\text{Expected Incremental Profit}_i = \hat{\tau}_i(a) \times V_i - C(a)$.
- **Synthetic Data Labeling**: All 20,000 customer records and campaign outcomes are synthetically generated (`seed=42`) with known ground-truth parameters to enable direct verification of CATE accuracy (`PEHE = 0.0610`) and realized campaign ROI.
