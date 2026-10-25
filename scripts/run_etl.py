"""CLI Script to execute the PySpark/DataFrame ETL pipeline and NLP feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.pyspark_pipeline import RetentionSparkETLPipeline
from src.nlp.complaint_analyzer import run_complaint_intelligence
from src.utils.config_loader import get_logger, load_config

logger = get_logger("scripts.run_etl")


def main() -> None:
    config = load_config()
    pipeline = RetentionSparkETLPipeline(config=config)
    features_df, _ = pipeline.run(save_to_disk=True)
    run_complaint_intelligence(features_df=features_df, config=config, save_to_disk=True)
    logger.info("ETL and NLP feature enrichment pipeline finished successfully.")


if __name__ == "__main__":
    main()
