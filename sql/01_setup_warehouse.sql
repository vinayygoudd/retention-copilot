-- ============================================================================
-- 01_setup_warehouse.sql
-- Snowflake Role, Warehouse, Database, and Schema Provisioning Script
-- Project: Retention Copilot — AI-Powered Customer Retention Platform
-- ============================================================================

-- 1. Create dedicated compute warehouse with cost-saving auto-suspend (60 seconds)
CREATE WAREHOUSE IF NOT EXISTS RETENTION_WH
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Compute warehouse for Retention Copilot ETL, scoring, and analytical queries';

-- 2. Create database and schemas
CREATE DATABASE IF NOT EXISTS RETENTION_COPILOT_DB
    COMMENT = 'Database for Retention Copilot customer churn, uplift, and financial decision tables';

USE DATABASE RETENTION_COPILOT_DB;

CREATE SCHEMA IF NOT EXISTS RAW_INGESTION
    COMMENT = 'Landing zone for raw customer subscriptions, telemetry, tickets, and RCT outcomes';

CREATE SCHEMA IF NOT EXISTS ANALYTICS
    COMMENT = 'Curated feature store, model predictions, SHAP attributions, and financial recommendations';
