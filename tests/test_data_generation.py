"""Automated tests for synthetic data generation and validation (`src/data_generation/`)."""

from __future__ import annotations

from src.data_generation.generator import generate_synthetic_dataset
from src.data_generation.validator import validate_generated_dataset
from src.utils.config_loader import load_config


def test_synthetic_dataset_generation_and_validation() -> None:
    config = load_config()
    bundle = generate_synthetic_dataset(config=config, n_customers_override=2500, save_to_disk=False)

    assert len(bundle.customers) == 2500
    assert len(bundle.usage_billing) == 2500 * 4
    assert len(bundle.campaigns_outcomes) == 2500
    assert bundle.customers["customer_id"].is_unique
    assert 0.25 < bundle.metadata["overall_churn_rate"] < 0.65
    assert bundle.metadata["empirical_ate_retention"] > 0.02

    report = validate_generated_dataset(bundle, cutoff_date="2026-06-30")
    assert report.passed, f"Validation failed: {report.errors}"
    assert report.checks_passed == report.checks_run
    assert report.metrics["n_post_cutoff_tickets"] > 0
