"""Automated tests for ETL pipeline, feature engineering, and strict temporal leakage prevention."""

from __future__ import annotations

import pandas as pd

from src.data_generation.generator import generate_synthetic_dataset
from src.etl.feature_engineering import NUMERIC_CORE_FEATURES, engineer_customer_features
from src.nlp.complaint_analyzer import run_complaint_intelligence
from src.utils.config_loader import get_project_root, load_config


def test_temporal_cutoff_leakage_prevention() -> None:
    config = load_config()
    bundle = generate_synthetic_dataset(config=config, n_customers_override=1000, save_to_disk=False)

    features_df, audit = engineer_customer_features(
        customers_df=bundle.customers,
        usage_billing_df=bundle.usage_billing,
        support_tickets_df=bundle.support_tickets,
        campaigns_outcomes_df=bundle.campaigns_outcomes,
        cutoff_date="2026-06-30",
    )

    # Verify all 1,000 post-cutoff July usage rows and all post-cutoff tickets were dropped
    assert audit["dropped_post_cutoff_usage_rows"] == 1000
    assert audit["dropped_post_cutoff_ticket_rows"] > 0
    assert len(features_df) == 1000

    # Verify zero post-cutoff leakage phrases appear in combined_complaint_text
    leakage_phrases = ["Confirmed account termination", "Post-cutoff exit interview", "Final retention save offer"]
    for phrase in leakage_phrases:
        assert not features_df["combined_complaint_text"].str.contains(phrase).any(), (
            f"Target leakage detected: post-cutoff phrase '{phrase}' leaked into features!"
        )

    for col in NUMERIC_CORE_FEATURES:
        assert col in features_df.columns, f"Missing engineered feature column: {col}"
        assert features_df[col].isna().sum() == 0, f"NaN values found in feature column: {col}"


def test_nlp_complaint_clustering_and_features() -> None:
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "customer_features_with_nlp.parquet")
    clusters_df = pd.read_parquet(root / "data" / "processed" / "complaint_cluster_summary.parquet")

    assert len(df) == 20000
    assert "complaint_theme_label" in df.columns
    # 5 complaint clusters + 1 no_complaint baseline cohort = 6 rows
    assert len(clusters_df) == 6
    complaining_rate = float(df[df["complaint_cluster_id"] >= 0]["churned"].mean())
    no_complaint_rate = float(df[df["complaint_cluster_id"] == -1]["churned"].mean())
    assert complaining_rate > no_complaint_rate + 0.15
