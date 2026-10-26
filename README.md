# Retention Copilot — AI-Powered Customer Retention Platform










**End-to-End AI/ML Engineering Portfolio Project**  
*Predictive Churn Modeling, Probability Calibration, SHAP Explainability, Pre-Cutoff NLP Complaint Intelligence, Causal T-Learner Uplift Estimation, Deterministic Financial ROI Optimization, Snowflake Data Warehouse Integration, and Claim-Validated GenAI Action Briefs.*

## 1. Executive Summary & Business Problem

Enterprise customer retention programs frequently lose money because they conflate **predictive churn risk** with **causal persuadability**. Standard churn models rank customers by $\hat{P}(\text{Churn})$, inadvertently spending retention budgets on:
- **Lost Causes**: Customers with very high churn risk who will cancel regardless of intervention.
- **Sure Things**: Loyal customers who would have renewed without a costly promotional discount.
- **Sleeping Dogs**: Customers who react negatively when disturbed by aggressive renewal outreach.

**Retention Copilot** solves this end-to-end by combining **calibrated churn prediction**, **root-cause NLP complaint clustering**, **causal T-Learner uplift modeling**, and a **deterministic financial decision engine**:
$$\text{Expected Incremental Profit}_i(a) = \hat{\tau}_i(a) \times V_i - C(a)$$
where $\hat{\tau}_i(a)$ is the estimated incremental retention probability under intervention $a$, $V_i$ is the expected contribution margin LTV if retained, and $C(a)$ is the direct intervention cost.

Under a **$75,000** campaign budget across a 20,000-customer portfolio, **Profit-Optimized Uplift + Offer Matching** generates **$1,042,746.06 in expected net incremental profit ($731,933.69 ground-truth realized net profit, 1,391.1% expected ROI)**—outperforming **Naive Churn-Risk Targeting ($235,111.14 expected / $176,236.31 realized net profit)** by more than **4.1x**.

> **Synthetic Data Notice**: All 20,000 customer accounts, telemetry snapshots, support complaint tickets, and randomized campaign outcomes in this repository are **synthetically generated** (`seed=42`) with known ground-truth treatment effects to enable rigorous verification of causal uplift estimation (`PEHE = 0.0587`) and realized campaign ROI.

## 2. System Architecture

```text
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
|  - Sentence-Transformers (all-MiniLM-L6-v2)   |-->|  - Logistic Regression vs. XGBoost            |
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

## 3. Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Core Data Science & Statistics** | `Python 3.10+`, `NumPy`, `Pandas`, `SciPy`, `Statsmodels`, `PyArrow`, `PyYAML` |
| **Data Engineering & Warehousing** | `PySpark` (`SparkSession` + local DataFrame fallback), `Snowflake` (`snowflake-connector-python`), `SQLite3`, `SQL` |
| **Machine Learning & Uplift** | `scikit-learn`, `XGBoost`, Custom Two-Model `T-Learner` for Heterogeneous Treatment Effects (CATE) |
| **Explainable AI (XAI)** | `SHAP` (`TreeExplainer`, global feature importance, individual waterfall log-odds drivers, causal disclaimers) |
| **NLP Complaint Intelligence** | `Sentence-Transformers` (`all-MiniLM-L6-v2` embeddings + deterministic TF-IDF/SVD fallback), `K-Means++`, Hungarian assignment |
| **GenAI & Claim Verification** | `OpenAI API` (`gpt-4o-mini`), deterministic grounded fallback template, programmatic 9-point factual claim validator |
| **Dashboard & Visualization** | `Streamlit` (6-page interactive web application), `Plotly`, `Matplotlib`, `Seaborn` |
| **Testing & Quality Assurance** | `Pytest` (`10/10` automated unit and integration tests covering data validation, leakage, ML, uplift, finance, and SQL) |

## 4. Key Analytical Results

### A. Churn Prediction & Calibration (Held-Out Test Split, `N = 3,001`)

| Model Configuration | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1-Score | Test Brier Score | Test ECE (10-Bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Core Telemetry Only)** | 0.8025 | 0.8040 | 0.7775 | 0.7251 | 0.6527 | 0.6865 | 0.1762 | 0.0261 |
| **Logistic Regression (Core + NLP Features)** | 0.8029 | 0.8035 | 0.7769 | 0.7236 | 0.6520 | 0.6824 | 0.1764 | 0.0275 |
| **XGBoost (Core Telemetry Only)** | 0.8134 | 0.8161 | 0.8013 | 0.7449 | 0.6551 | 0.7026 | 0.1687 | 0.0188 |
| **XGBoost (Core + NLP Features — Champion)** | **0.8135** | **0.8162** | **0.8009** | **0.7389** | **0.6543** | **0.6989** | **0.1690** | **0.0196** |

- **Top Global SHAP Predictive Drivers**: `Month-to-Month Contract` (`mean |SHAP| = 0.5840`), `Minimum Support CSAT Score (1-5)`, `90-Day Usage Decline (%)`, `Current Seat Utilization Ratio`, `Two-Year Multi-Year Contract`, and `Account Tenure (Months)`.
- **Role of NLP Complaint Intelligence**: While structured ticket counters already capture most raw classification lift, semantic clustering (`all-MiniLM-L6-v2`) separates `12,776` complaining accounts (`53.2%–57.6%` churn) from `7,224` non-complaining accounts (`23.2%` churn) and powers **prescriptive offer matching** (routing billing complaints to promotional discounts and technical/SLA complaints to priority engineering support).

### B. T-Learner Uplift & Financial Policy Evaluation (`$75,000` Campaign Budget)

- **RCT Empirical Retention Lift**: Overall treated retention is **`59.72%`** vs. control retention **`51.32%`** (**ATE = `+8.40%`**, test split $z = 4.351, p = 1.35 \times 10^{-5}$).
- **T-Learner CATE Accuracy (`test` split)**: Achieves **Pearson $r = 0.6371$** against ground-truth individual treatment effects, **PEHE (RMSE) = `0.0587`**, and **AUUC = `58.05` expected saves per 1,000 customers** (vs. `49.02` for Churn-Risk Targeting and `44.64` for Random Targeting).

| Targeting Policy (`$75,000` Budget Limit) | Accounts Targeted | Total Spend ($) | ExpectedNet Profit ($) | Ground-Truth Realized Net Profit ($) | Expected ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Profit-Optimized Uplift + Offer Matching (Recommended)** | **893** | $74,985 | **$1,042,746.06** | **$731,933.69** | **1,391.1%** |
| **2. High-LTV Account Targeting (Top Customer Value)** | 1,666 | $74,970 | $811,187.27 | $840,120.43 | 1,082.0% |
| **3. Pure Uplift Targeting (Top Incremental Retention Prob)** | 1,666 | $74,970 | $268,037.85 | $192,716.21 | 357.5% |
| **4. Naive Churn-Risk Targeting (Top Churn Probability)** | 1,666 | $74,970 | $235,111.14 | $176,236.31 | 313.6% |
| **5. Random Targeting (Untargeted Campaign Blast)** | 1,666 | $74,970 | $142,367.81 | $158,363.27 | 189.9% |

## 5. Six-Page Interactive Streamlit & Plotly Dashboard

Launching `streamlit run app/streamlit_app.py` opens a six-page executive decision application:
1. **Executive Overview**: Portfolio KPIs, expected LTV at risk by segment/tier, global SHAP importance bar chart, model evaluation table, and 10-bin calibration curves.
2. **Customer Explorer**: Account-level lookup across all 20,000 customers, local SHAP waterfall chart, telemetry diagnostics, and predictive-vs-causal governance notices.
3. **Retention Prioritization**: Interactive budget slider (`$10k–$200k`), churn/uplift eligibility thresholds, Uplift Quadrant scatter map (*Persuadables*, *Sure Things*, *Lost Causes*, *Sleeping Dogs*), and prioritized intervention queue.
4. **Complaint Intelligence**: Semantic complaint cluster breakdown, churn rate & severity by theme, SLA resolution metrics, and NLP ablation comparison.
5. **AI Action Brief**: Real-time OpenAI / Deterministic Evidence-Grounded Action Brief generator paired with a 9-point automated numerical claim verifier.
6. **Campaign Evaluation**: T-Learner uplift curves, budget efficiency frontier, RCT hypothesis test table ($z$-statistics, 95% CIs, $p$-values), and causal identification assumptions.

## 6. Quickstart & Local Execution Guide

### Windows (`cmd.exe` / PowerShell)

```cmd
:: 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 2. Install dependencies and initialize environment file
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env

:: 3. Run the end-to-end pipeline
python scripts\run_data_generation.py
python scripts\run_etl.py
python scripts\run_training_and_eval.py
mkdir data\warehouse
python scripts\run_warehouse_load.py

:: 4. Run the automated test suite (10/10 passing)
pytest

:: 5. Launch the 6-page interactive Streamlit dashboard
streamlit run app\streamlit_app.py
```

### Linux / macOS (`bash` / `zsh`)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies and initialize environment file
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env

# 3. Run the end-to-end pipeline
python3 scripts/run_data_generation.py
python3 scripts/run_etl.py
python3 scripts/run_training_and_eval.py
python3 scripts/run_warehouse_load.py

# 4. Run the automated test suite (10/10 passing)
pytest

# 5. Launch the 6-page interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

## 7. Repository Structure

```text
retention_copilot/
├── README.md                                     # Project documentation, architecture, metrics, and setup
├── requirements.txt                              # Pinned Python dependencies across the stack
├── pyproject.toml                                # Build configuration and pytest settings
├── .gitignore                                    # Excludes .env, credentials, virtualenvs, and local binaries
├── .env.example                                  # Template for OpenAI API and Snowflake credentials
├── configs/
│   └── config.yaml                               # Single source of truth for pipeline parameters
├── src/
│   ├── utils/config_loader.py                    # YAML configuration loader and structured logger
│   ├── data_generation/
│   │   ├── generator.py                          # 20,000-customer synthetic RCT & telemetry generator
│   │   └── validator.py                          # 22-check schema, range, leakage, and SMD validator
│   ├── etl/
│   │   ├── feature_engineering.py                # Cutoff filtering and 29 behavioral/billing features
│   │   └── pyspark_pipeline.py                   # PySpark SQL / DataFrame ETL & stratified splitting
│   ├── nlp/
│   │   └── complaint_analyzer.py                 # Sentence-Transformers embeddings & 5 complaint clusters
│   ├── models/
│   │   ├── churn_model.py                        # Logistic Regression, XGBoost, & calibration curves
│   │   └── uplift_model.py                       # T-Learner CATE estimator, AUUC, & RCT z-tests
│   ├── explainability/
│   │   └── shap_explainer.py                     # Global & individual SHAP explanations + disclaimers
│   ├── decision_engine/
│   │   └── financial_engine.py                   # Deterministic profit formula & budget optimization
│   ├── llm/
│   │   └── action_brief_generator.py             # OpenAI API + deterministic fallback & 9-point validator
│   └── warehouse/
│       └── snowflake_client.py                   # Snowflake cloud connector & SQLite local warehouse
├── app/
│   └── streamlit_app.py                          # 6-Page interactive Streamlit & Plotly dashboard
├── sql/
│   ├── 01_setup_warehouse.sql                    # Snowflake warehouse, database, and schema DDL
│   ├── 02_create_tables.sql                      # Table DDL (CUSTOMERS, USAGE_BILLING, TICKETS, DECISIONS)
│   ├── 03_analytical_queries.sql                 # 4 executive, complaint, RCT, and ROI SQL queries
│   └── 04_data_validation.sql                    # 3 automated warehouse integrity & budget checks
├── notebooks/
│   ├── 01_eda_and_data_validation.ipynb          # EDA, covariate balance (SMD), and leakage audit
│   ├── 02_churn_modeling_and_calibration.ipynb   # Logistic vs XGBoost & 10-bin calibration curves
│   ├── 03_shap_explainability.ipynb              # Global mean |SHAP| and customer waterfall examples
│   ├── 04_nlp_complaint_intelligence.ipynb       # Pre-cutoff complaint clusters and ablation study
│   └── 05_uplift_and_financial_evaluation.ipynb  # T-Learner CATE, RCT z-tests, and $75k budget ROI
├── tests/
│   ├── test_data_generation.py                   # Schema, bounds, and RCT balance tests
│   ├── test_etl_leakage.py                       # Post-cutoff leakage firewall and NLP cluster tests
│   ├── test_churn_and_uplift.py                  # ROC-AUC, Brier, SHAP, and T-Learner AUUC tests
│   ├── test_financial_engine.py                  # Exact profit equation and budget knapsack tests
│   └── test_action_briefs_and_warehouse.py       # LLM claim verification and SQL warehouse tests
├── scripts/
│   ├── run_data_generation.py                    # Step 1 CLI: Generate & validate synthetic data
│   ├── run_etl.py                                # Step 2 CLI: Run PySpark/DataFrame ETL & NLP clustering
│   ├── run_training_and_eval.py                  # Step 3 CLI: Train Churn, SHAP, Uplift, & Financial Engine
│   ├── run_warehouse_load.py                     # Step 4 CLI: Load warehouse & run SQL validations
│   ├── generate_notebooks.py                     # Utility to regenerate executed Jupyter notebooks
│   └── run_tests.py                              # Pytest-compatible automated test suite runner
├── reports/
│   ├── model_evaluation_report.md                # Full churn, calibration, SHAP, and NLP report
│   ├── campaign_comparison_report.md             # Full RCT, T-Learner uplift, and financial ROI report
│   └── *.json                                    # Machine-readable pipeline metrics and audit logs
└── docs/
    ├── ARCHITECTURE_AND_DATA_DICTIONARY.md       # System diagram and table schemas
    ├── METHODOLOGY_AND_LIMITATIONS.md            # Causal assumptions, leakage firewall, and limitations
    ├── SETUP_AND_TESTING.md                      # Step-by-step installation, CLI commands, and testing
    ├── SNOWFLAKE_SETUP_GUIDE.md                  # Snowflake cloud setup, RBAC, FinOps, and troubleshooting
    └── PROJECT_COMPLETION_CHECKLIST.md           # Deliverable verification matrix
```

