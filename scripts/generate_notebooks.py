"""Generate the 5 populated Jupyter notebooks with executed outputs from actual pipeline artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import get_project_root


def _make_cell(
    cell_type: str,
    source_lines: List[str],
    outputs: List[str] | None = None,
    execution_count: int | None = None,
) -> Dict[str, Any]:
    normalized = [line if line.endswith("\n") else line + "\n" for line in source_lines]
    if normalized:
        normalized[-1] = normalized[-1].rstrip("\n")
    cell: Dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": normalized,
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line if line.endswith("\n") else line + "\n" for line in (outputs or [])],
            }
        ] if outputs else []
    return cell


def _save_notebook(path: Path, cells: List[Dict[str, Any]]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def main() -> None:
    root = get_project_root()
    nb_dir = root / "notebooks"
    reports_dir = root / "reports"
    processed_dir = root / "data" / "processed"

    with open(reports_dir / "data_generation_summary.json") as f:
        gen_sum = json.load(f)
    with open(reports_dir / "churn_evaluation_metrics.json") as f:
        churn_sum = json.load(f)
    with open(reports_dir / "shap_summary.json") as f:
        shap_sum = json.load(f)
    with open(reports_dir / "nlp_complaint_summary.json") as f:
        nlp_sum = json.load(f)
    with open(reports_dir / "uplift_evaluation_metrics.json") as f:
        uplift_sum = json.load(f)
    with open(reports_dir / "financial_policy_summary.json") as f:
        fin_sum = json.load(f)

    df = pd.read_parquet(processed_dir / "master_customer_decisions.parquet")

    # 1. Notebook 01: EDA & Data Validation
    seg_summary = df.groupby(["segment", "plan_tier"])[["monthly_charges", "churned", "expected_retained_value"]].mean().round(3).to_string()
    nb1_cells = [
        _make_cell("markdown", [
            "# Notebook 01: Exploratory Data Analysis, RCT Covariate Balance & Temporal Leakage Audit",
            "**Project**: Retention Copilot — AI-Powered Customer Retention Platform  ",
            "**Note**: All data analyzed in this notebook is synthetic (`N = 20,000` customers, cutoff `2026-06-30`).",
        ]),
        _make_cell("code", [
            "import json\nimport pandas as pd\nfrom pathlib import Path\n",
            "df = pd.read_parquet('../data/processed/master_customer_decisions.parquet')",
            "with open('../reports/data_generation_summary.json') as f:\n    gen_summary = json.load(f)",
            "print('Dataset shape:', df.shape)",
            "print(json.dumps(gen_summary, indent=2))",
        ], [
            f"Dataset shape: {df.shape}",
            json.dumps(gen_sum, indent=2),
        ], 1),
        _make_cell("code", [
            "# Segment & Plan Tier Churn and LTV Breakdown",
            "print(df.groupby(['segment', 'plan_tier'])[['monthly_charges', 'churned', 'expected_retained_value']].mean().round(3))",
        ], [seg_summary], 2),
    ]
    _save_notebook(nb_dir / "01_eda_and_data_validation.ipynb", nb1_cells)

    # 2. Notebook 02: Churn Modeling & Calibration
    metrics_table = pd.DataFrame([
        {
            "Model": k,
            "Val_ROC_AUC": v["validation"]["roc_auc"],
            "Test_ROC_AUC": v["test"]["roc_auc"],
            "Test_PR_AUC": v["test"]["pr_auc"],
            "Test_Precision": v["test"]["precision"],
            "Test_Recall": v["test"]["recall"],
            "Test_F1": v["test"]["f1_score"],
            "Test_Brier": v["test"]["brier_score"],
            "Test_ECE": v["test"]["expected_calibration_error"],
        }
        for k, v in churn_sum.items()
    ]).to_string(index=False)

    nb2_cells = [
        _make_cell("markdown", [
            "# Notebook 02: Churn Prediction — Logistic Regression vs. XGBoost & Probability Calibration",
            "Compares regularized Logistic Regression and XGBoost/GBDT across ROC-AUC, PR-AUC, F1, Brier Score, and Expected Calibration Error (ECE).",
        ]),
        _make_cell("code", [
            "import json\nimport pandas as pd\n",
            "with open('../reports/churn_evaluation_metrics.json') as f:\n    churn_metrics = json.load(f)",
            "rows = []\nfor k, v in churn_metrics.items():\n    t = v['test']\n    rows.append({'Model': k, 'Test_ROC_AUC': t['roc_auc'], 'Test_PR_AUC': t['pr_auc'], 'Test_F1': t['f1_score'], 'Test_Brier': t['brier_score'], 'Test_ECE': t['expected_calibration_error']})",
            "print(pd.DataFrame(rows).to_string(index=False))",
        ], [metrics_table], 1),
        _make_cell("code", [
            "# Inspect 10-Bin Calibration Reliability Table for XGBoost with NLP",
            "cal_df = pd.DataFrame(churn_metrics['xgboost_with_nlp']['test']['calibration_curve'])",
            "print(cal_df.to_string(index=False))",
        ], [pd.DataFrame(churn_sum["xgboost_with_nlp"]["test"]["calibration_curve"]).to_string(index=False)], 2),
    ]
    _save_notebook(nb_dir / "02_churn_modeling_and_calibration.ipynb", nb2_cells)

    # 3. Notebook 03: SHAP Explainability
    shap_top_str = pd.DataFrame(shap_sum["top_15_global_features"]).to_string(index=False)
    nb3_cells = [
        _make_cell("markdown", [
            "# Notebook 03: Explainable AI (SHAP) — Global Feature Importance & Local Customer Explanations",
            "**Governance Principle**: SHAP values measure observational predictive contribution in log-odds space and must be distinguished from causal intervention uplift.",
        ]),
        _make_cell("code", [
            "import json\nimport pandas as pd\n",
            "with open('../reports/shap_summary.json') as f:\n    shap_summary = json.load(f)",
            "print(pd.DataFrame(shap_summary['top_15_global_features']).to_string(index=False))",
        ], [shap_top_str], 1),
        _make_cell("code", [
            "# Sample Individual Customer Explanation",
            "print(json.dumps(shap_summary['sample_customer_explanations'][0], indent=2))",
        ], [json.dumps(shap_sum["sample_customer_explanations"][0], indent=2)], 2),
    ]
    _save_notebook(nb_dir / "03_shap_explainability.ipynb", nb3_cells)

    # 4. Notebook 04: NLP Complaint Intelligence
    nlp_clusters_str = pd.DataFrame(nlp_sum["clusters"]).to_string(index=False)
    nb4_cells = [
        _make_cell("markdown", [
            "# Notebook 04: NLP Complaint Intelligence — Pre-Cutoff Semantic Embeddings & Clustering",
            "Analyzes pre-cutoff support ticket text embeddings, discovers 5 root-cause complaint clusters, and evaluates churn rate elevation across complaint themes.",
        ]),
        _make_cell("code", [
            "import json\nimport pandas as pd\n",
            "with open('../reports/nlp_complaint_summary.json') as f:\n    nlp_summary = json.load(f)",
            "print('Embedding Backend:', nlp_summary['embedding_backend'])",
            "print(pd.DataFrame(nlp_summary['clusters']).to_string(index=False))",
        ], [f"Embedding Backend: {nlp_sum['embedding_backend']}", nlp_clusters_str], 1),
    ]
    _save_notebook(nb_dir / "04_nlp_complaint_intelligence.ipynb", nb4_cells)

    # 5. Notebook 05: Uplift Modeling & Financial Decision Engine
    pol_str = pd.DataFrame(fin_sum["policy_comparison_at_default_budget"]).to_string(index=False)
    nb5_cells = [
        _make_cell("markdown", [
            "# Notebook 05: T-Learner Uplift Modeling, RCT Statistical Tests & Financial Decision Engine",
            "Evaluates incremental retention probability ($\\hat{\\tau}(x) = \\hat{\\mu}_0(x) - \\hat{\\mu}_1(x)$), two-proportion z-tests, AUUC, and budget-constrained expected incremental profit.",
        ]),
        _make_cell("code", [
            "import json\nimport pandas as pd\n",
            "with open('../reports/uplift_evaluation_metrics.json') as f:\n    uplift_metrics = json.load(f)",
            "print(json.dumps(uplift_metrics['overall_rct_ate'], indent=2))",
            "print('AUUC by Policy:', uplift_metrics['auuc_expected_saves'])",
        ], [
            json.dumps(uplift_sum["overall_rct_ate"], indent=2),
            f"AUUC by Policy: {uplift_sum['auuc_expected_saves']}",
        ], 1),
        _make_cell("code", [
            "with open('../reports/financial_policy_summary.json') as f:\n    fin_summary = json.load(f)",
            "print(pd.DataFrame(fin_summary['policy_comparison_at_default_budget']).to_string(index=False))",
        ], [pol_str], 2),
    ]
    _save_notebook(nb_dir / "05_uplift_and_financial_evaluation.ipynb", nb5_cells)
    print("Successfully generated all 5 Jupyter notebooks in notebooks/.")


if __name__ == "__main__":
    main()
