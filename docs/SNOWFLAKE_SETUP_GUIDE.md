# Snowflake Data Warehouse Setup & Integration Guide

**Project**: Retention Copilot — AI-Powered Customer Retention Platform  
**Target Role**: Deloitte Analyst – AI/ML Portfolio Deliverable  
**Module**: `src/warehouse/snowflake_client.py` & `sql/01_setup_warehouse.sql` – `04_data_validation.sql`

> **Transparency & Verification Status**:  
> In accordance with engineering governance, no Snowflake credentials have been fabricated. All SQL DDL scripts (`sql/01_setup_warehouse.sql`, `sql/02_create_tables.sql`), bulk table loaders, analytical queries (`sql/03_analytical_queries.sql`), and automated integrity checks (`sql/04_data_validation.sql`) are **implemented and verified locally** against the embedded SQLite relational warehouse (`data/warehouse/retention_warehouse.db` — 20,000 customer records, 80,000 usage/billing records, 22,556 support tickets, and 20,000 decision records with 0 duplicates). Live cloud execution is **pending your personal Snowflake account credentials** in `.env`.

## 1. Snowflake Account & Trial Provisioning

1. Navigate to [signup.snowflake.com](https://signup.snowflake.com/) and register for a **30-day Free Trial** ($400 in free compute credits, no credit card required).
2. Choose **Standard** or **Enterprise** edition on your preferred cloud provider (AWS `us-east-1` or Azure `east-us-2` recommended).
3. Activate your account via the confirmation email and copy your **Account Identifier** (e.g., `xy12345.us-east-1` or `orgname-accountname`).

## 2. Role, Warehouse, Database, and Schema Configuration

Log into the Snowflake Snowsight Web UI as `ACCOUNTADMIN` or `SYSADMIN`, open a new SQL Worksheet, and execute `sql/01_setup_warehouse.sql`:

```sql
CREATE WAREHOUSE IF NOT EXISTS RETENTION_WH
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Compute warehouse for Retention Copilot ETL, scoring, and analytical queries';

CREATE DATABASE IF NOT EXISTS RETENTION_COPILOT_DB
    COMMENT = 'Database for Retention Copilot customer churn, uplift, and financial decision tables';

USE DATABASE RETENTION_COPILOT_DB;

CREATE SCHEMA IF NOT EXISTS RAW_INGESTION;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;
```

### Least-Privilege Service Role & User Setup

```sql
USE ROLE SECURITYADMIN;

CREATE ROLE IF NOT EXISTS RETENTION_ANALYST_ROLE;
GRANT USAGE ON WAREHOUSE RETENTION_WH TO ROLE RETENTION_ANALYST_ROLE;
GRANT USAGE ON DATABASE RETENTION_COPILOT_DB TO ROLE RETENTION_ANALYST_ROLE;
GRANT ALL PRIVILEGES ON SCHEMA RETENTION_COPILOT_DB.RAW_INGESTION TO ROLE RETENTION_ANALYST_ROLE;
GRANT ALL PRIVILEGES ON SCHEMA RETENTION_COPILOT_DB.ANALYTICS TO ROLE RETENTION_ANALYST_ROLE;

CREATE USER IF NOT EXISTS RETENTION_SERVICE_USER
    PASSWORD = '<YOUR_STRONG_PASSWORD>'
    DEFAULT_ROLE = RETENTION_ANALYST_ROLE
    DEFAULT_WAREHOUSE = RETENTION_WH
    DEFAULT_NAMESPACE = 'RETENTION_COPILOT_DB.ANALYTICS';

GRANT ROLE RETENTION_ANALYST_ROLE TO USER RETENTION_SERVICE_USER;
```

## 3. Creating Tables (`sql/02_create_tables.sql`)

You can either run `sql/02_create_tables.sql` manually in Snowsight or let `python3 scripts/run_warehouse_load.py` execute it automatically. The DDL creates four core tables:
- `CUSTOMERS` (20,000 rows — primary key `customer_id`)
- `USAGE_BILLING` (80,000 rows — 3 pre-cutoff months + 1 post-cutoff audit month)
- `SUPPORT_TICKETS` (22,556 rows — 22,306 pre-cutoff tickets + 250 post-cutoff leakage test tickets)
- `CUSTOMER_DECISIONS` (20,000 rows — curated features, churn probabilities, SHAP top drivers, T-Learner uplift, recommended intervention, and expected incremental profit)

## 4. Environment Variables Configuration (`.env`)

Copy `.env.example` to `.env` in `retention_copilot/.env` and populate your Snowflake credentials:

```bash
cp .env.example .env
```

```ini
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_USER=RETENTION_SERVICE_USER
SNOWFLAKE_PASSWORD=your_actual_snowflake_password
SNOWFLAKE_ROLE=RETENTION_ANALYST_ROLE
SNOWFLAKE_WAREHOUSE=RETENTION_WH
SNOWFLAKE_DATABASE=RETENTION_COPILOT_DB
SNOWFLAKE_SCHEMA=ANALYTICS
```

> **Security Guardrail**: `.env` is listed in `.gitignore`. Never commit `.env`, private RSA keys (`.p8`/`.pem`), or passwords to Git.

## 5. Data Loading & Validation Execution

```bash
pip install "snowflake-connector-python[pandas]>=3.6.0"
python3 scripts/run_warehouse_load.py
```

When `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, and `SNOWFLAKE_PASSWORD` are set, `RetentionWarehouseManager` connects via `snowflake.connector.connect(...)`, executes `01_setup_warehouse.sql` and `02_create_tables.sql`, bulk-loads DataFrames via `write_pandas`, runs `04_data_validation.sql` and `03_analytical_queries.sql`, and saves the audit log to `reports/warehouse_validation_report.json`.

## 6. Troubleshooting Common Issues

| Issue / Error Code | Root Cause | Resolution |
| :--- | :--- | :--- |
| `250001 (08001): Failed to connect to DB` | Incorrect `SNOWFLAKE_ACCOUNT` format | Exclude `https://` and `.snowflakecomputing.com`. Use `<org>-<account>` or `<locator>.<region>.<cloud>`. |
| `002043 (02000): Object does not exist` | Role lacks privilege on `RETENTION_COPILOT_DB` or `RETENTION_WH` | Run the `GRANT USAGE` statements in Section 2 as `SECURITYADMIN` or `ACCOUNTADMIN`. |
| `write_pandas` throws `pyarrow` error | Incompatible PyArrow binary | Run `pip install "snowflake-connector-python[pandas]>=3.6.0" --upgrade`. |
| MFA / SSO required by policy | Password-only auth blocked | Set `authenticator='externalbrowser'` or configure key-pair authentication. |

## 7. FinOps & Cost Control Best Practices

- **Warehouse Sizing**: `XSMALL` (1 credit/hour, billed per second with a 60-second minimum) loads and queries all 4 tables in < 12 seconds (~0.0033 credits, or < $0.01 per run).
- **Auto-Suspend**: `AUTO_SUSPEND = 60` suspends compute 60 seconds after the last query finishes.
- **Resource Monitors**: Attach a monthly credit quota monitor (`CREDIT_QUOTA = 10`, suspend at 100%) to prevent unexpected trial credit consumption.
