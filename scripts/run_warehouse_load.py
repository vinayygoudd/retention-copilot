"""CLI Script to execute Snowflake / Local Warehouse DDL, Data Loading, and SQL Validation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import get_logger, load_config
from src.warehouse.snowflake_client import RetentionWarehouseManager

logger = get_logger("scripts.run_warehouse_load")


def main() -> None:
    config = load_config()
    manager = RetentionWarehouseManager(config=config)
    report = manager.run_warehouse_pipeline()
    logger.info(f"Warehouse pipeline finished: {report['snowflake_status']}")


if __name__ == "__main__":
    main()
