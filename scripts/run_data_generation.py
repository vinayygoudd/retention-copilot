"""CLI Script to generate and validate the 20,000-customer synthetic dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.generator import generate_synthetic_dataset
from src.data_generation.validator import validate_generated_dataset
from src.utils.config_loader import get_logger, get_project_root, load_config

logger = get_logger("scripts.run_data_generation")


def main() -> None:
    config = load_config()
    bundle = generate_synthetic_dataset(config=config, save_to_disk=True)
    report = validate_generated_dataset(
        bundle=bundle,
        cutoff_date=config["project"]["prediction_cutoff_date"],
    )
    if not report.passed:
        logger.error(f"Validation failed with errors: {report.errors}")
        sys.exit(1)

    summary_path = get_project_root() / "reports" / "data_generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": bundle.metadata,
                "validation": {
                    "passed": report.passed,
                    "checks_run": report.checks_run,
                    "checks_passed": report.checks_passed,
                    "metrics": report.metrics,
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote data generation and validation summary to {summary_path}")


if __name__ == "__main__":
    main()
