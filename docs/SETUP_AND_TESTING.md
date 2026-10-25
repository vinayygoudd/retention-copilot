# Local Setup, Execution & Automated Testing Guide

## 1. Environment Installation
```bash
cd retention_copilot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Exact Execution Commands
```bash
# Step 1: Generate and validate 20,000-customer synthetic dataset
python3 scripts/run_data_generation.py

# Step 2: Run PySpark/DataFrame ETL (with temporal leakage guard) and NLP Complaint Intelligence
python3 scripts/run_etl.py

# Step 3: Train & evaluate Churn Models, SHAP Explainability, T-Learner Uplift, and Financial Engine
python3 scripts/run_training_and_eval.py

# Step 4: Load tables into Snowflake / Local SQLite Warehouse and run SQL validation & analytical queries
python3 scripts/run_warehouse_load.py

# Step 5: Generate and validate sample AI Action Briefs
python3 -c "from src.llm.action_brief_generator import generate_batch_sample_briefs; generate_batch_sample_briefs(8)"

# Step 6: Launch the 6-Page Interactive Streamlit Dashboard
streamlit run app/streamlit_app.py
```

## 3. Running the Automated Test Suite (`10/10` Passing)
```bash
pytest
# Or directly:
python3 scripts/run_tests.py
```
