"""Automated Test Runner for Retention Copilot.

Executes `pytest` when installed in the environment, or discovers and runs all
`tests/test_*.py` test functions directly with detailed pass/fail reporting.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_MODULES = [
    "tests.test_data_generation",
    "tests.test_etl_leakage",
    "tests.test_churn_and_uplift",
    "tests.test_financial_engine",
    "tests.test_action_briefs_and_warehouse",
]


def main() -> int:
    try:
        import pytest  # type: ignore

        return int(pytest.main(["-v", "tests"]))
    except ImportError:
        pass

    print("======================================================================")
    print("Retention Copilot — Automated Test Suite Runner (pytest-compatible)")
    print("======================================================================")
    total = 0
    passed = 0
    failed = 0
    t0 = time.time()

    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_"):
                total += 1
                try:
                    t_start = time.time()
                    func()
                    elapsed = time.time() - t_start
                    passed += 1
                    print(f"PASSED  {mod_name}::{name} ({elapsed:.2f}s)")
                except Exception as exc:
                    failed += 1
                    print(f"FAILED  {mod_name}::{name} -> {exc}")
                    traceback.print_exc()

    total_elapsed = time.time() - t0
    print("----------------------------------------------------------------------")
    print(f"Summary: {passed}/{total} tests passed, {failed} failed in {total_elapsed:.2f}s")
    print("======================================================================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
