"""Churn Prediction, Evaluation, and Calibration Module for Retention Copilot.

Implements:
1. Standardized preprocessing pipeline (z-score scaling, missing value imputation)
2. Regularized Logistic Regression (`scikit-learn` or `scipy.optimize.minimize` L-BFGS-B)
3. Gradient Boosted Decision Trees / XGBoost (`xgboost.XGBClassifier` when installed,
   or a native vectorized second-order Newton Gradient Boosted Decision Tree ensemble
   with exact tree leaf weights and regularization)
4. Full evaluation suite: ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Brier Score,
   Expected Calibration Error (ECE), and 10-bin calibration curves
5. Ablation comparison: Models trained WITHOUT vs WITH NLP complaint features
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.etl.feature_engineering import NUMERIC_CORE_FEATURES
from src.utils.config_loader import get_logger, get_project_root, load_config

logger = get_logger("models.churn_model")

try:
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SklearnLogisticRegression = None  # type: ignore
    SklearnStandardScaler = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore
    XGBOOST_AVAILABLE = False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class StandardPreprocessor:
    """Zero-mean, unit-variance feature scaler with median/zero imputation."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "StandardPreprocessor":
        X = df[self.feature_names].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        self.means_ = X.mean(axis=0)
        stds = X.std(axis=0)
        self.stds_ = np.where(stds > 1e-8, stds, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("StandardPreprocessor must be fitted before calling transform().")
        X = df[self.feature_names].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        return (X - self.means_) / self.stds_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class LogisticChurnClassifier:
    """L2-Regularized Logistic Regression supporting scikit-learn and SciPy L-BFGS-B."""

    def __init__(self, C: float = 1.0, max_iter: int = 300) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._sklearn_model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticChurnClassifier":
        y_float = y.astype(np.float64)
        if SKLEARN_AVAILABLE and SklearnLogisticRegression is not None:
            model = SklearnLogisticRegression(C=self.C, max_iter=self.max_iter, random_state=42)
            model.fit(X, y_float)
            self._sklearn_model = model
            self.coef_ = model.coef_.ravel().copy()
            self.intercept_ = float(model.intercept_[0])
            return self

        n_samples, n_features = X.shape
        reg_lambda = 1.0 / max(self.C * n_samples, 1e-6)

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            w = params[:-1]
            b = params[-1]
            logits = X @ w + b
            probs = _sigmoid(logits)
            eps = 1e-12
            nll = -np.mean(y_float * np.log(probs + eps) + (1.0 - y_float) * np.log(1.0 - probs + eps))
            loss = float(nll + 0.5 * reg_lambda * np.sum(w ** 2))
            diff = (probs - y_float) / n_samples
            grad_w = X.T @ diff + reg_lambda * w
            grad_b = float(np.sum(diff))
            return loss, np.append(grad_w, grad_b)

        init_params = np.zeros(n_features + 1, dtype=np.float64)
        res = minimize(
            loss_and_grad,
            init_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coef_ = res.x[:-1].copy()
        self.intercept_ = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticChurnClassifier is not fitted.")
        logits = X @ self.coef_ + self.intercept_
        p1 = _sigmoid(logits)
        return np.column_stack([1.0 - p1, p1])


@dataclass
class _TreeNode:
    is_leaf: bool
    value: float = 0.0
    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    gain: float = 0.0


class XGBoostStyleChurnClassifier:
    """Second-Order Newton Gradient Boosted Decision Trees (XGBoost algorithm).

    Uses `xgboost.XGBClassifier` when the `xgboost` library is installed, and
    provides an exact, vectorized 2nd-order Taylor expansion (gradient `g` and
    hessian `h`) histogram tree booster in NumPy/SciPy when executing in
    environments without `xgboost` compiled binaries.
    """

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 4,
        learning_rate: float = 0.12,
        subsample: float = 0.85,
        reg_lambda: float = 2.0,
        min_samples_leaf: int = 25,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.reg_lambda = float(reg_lambda)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_seed = int(random_seed)
        self.base_logit_: float = 0.0
        self.trees_: List[_TreeNode] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self._xgb_model: Any = None

    def _build_newton_tree(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int,
        thresholds_per_feat: List[np.ndarray],
        gain_Importance: np.ndarray,
    ) -> _TreeNode:
        G = float(np.sum(g))
        H = float(np.sum(h))
        leaf_weight = -G / (H + self.reg_lambda)

        if depth >= self.max_depth or len(g) < 2 * self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=leaf_weight)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        parent_score = (G ** 2) / (H + self.reg_lambda)

        n_features = X.shape[1]
        for f_idx in range(n_features):
            col = X[:, f_idx]
            cand_thresholds = thresholds_per_feat[f_idx]
            if len(cand_thresholds) == 0:
                continue
            # Vectorized comparison: (n_samples, n_thresholds)
            mask_left = col[:, None] <= cand_thresholds[None, :]
            n_left = np.sum(mask_left, axis=0)
            n_right = len(col) - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            G_left = g @ mask_left
            H_left = h @ mask_left
            G_right = G - G_left
            H_right = H - H_left

            gains = 0.5 * (
                (G_left ** 2) / (H_left + self.reg_lambda)
                + (G_right ** 2) / (H_right + self.reg_lambda)
                - parent_score
            )
            gains = np.where(valid, gains, -1.0)
            max_idx = int(np.argmax(gains))
            if gains[max_idx] > best_gain:
                best_gain = float(gains[max_idx])
                best_feat = f_idx
                best_thresh = float(cand_thresholds[max_idx])

        if best_feat == -1 or best_gain <= 1e-5:
            return _TreeNode(is_leaf=True, value=leaf_weight)

        gain_Importance[best_feat] += best_gain
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_newton_tree(
            X[left_mask], g[left_mask], h[left_mask], depth + 1, thresholds_per_feat, gain_Importance
        )
        right_child = self._build_newton_tree(
            X[right_mask], g[right_mask], h[right_mask], depth + 1, thresholds_per_feat, gain_Importance
        )
        return _TreeNode(
            is_leaf=False,
            value=leaf_weight,
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            gain=best_gain,
        )

    @staticmethod
    def _predict_tree(node: _TreeNode, X: np.ndarray) -> np.ndarray:
        if node.is_leaf or node.left is None or node.right is None:
            return np.full(X.shape[0], node.value, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        left_mask = X[:, node.feature_idx] <= node.threshold
        if np.any(left_mask):
            out[left_mask] = XGBoostStyleChurnClassifier._predict_tree(node.left, X[left_mask])
        if np.any(~left_mask):
            out[~left_mask] = XGBoostStyleChurnClassifier._predict_tree(node.right, X[~left_mask])
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostStyleChurnClassifier":
        y_float = y.astype(np.float64)
        n_samples, n_features = X.shape

        if XGBOOST_AVAILABLE and xgb is not None:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                reg_lambda=self.reg_lambda,
                random_state=self.random_seed,
                eval_metric="logloss",
            )
            model.fit(X, y_float)
            self._xgb_model = model
            self.feature_importances_ = model.feature_importances_.copy()
            return self

        pos_rate = float(np.clip(np.mean(y_float), 0.01, 0.99))
        self.base_logit_ = float(np.log(pos_rate / (1.0 - pos_rate)))
        raw_logits = np.full(n_samples, self.base_logit_, dtype=np.float64)

        # Precompute candidate quantile thresholds per feature (fast histogram splits)
        thresholds_per_feat: List[np.ndarray] = []
        quantiles = np.linspace(0.08, 0.92, 11)
        for f_idx in range(n_features):
            uniq = np.unique(np.quantile(X[:, f_idx], quantiles))
            thresholds_per_feat.append(uniq)

        rng = np.random.default_rng(self.random_seed)
        gain_importance = np.zeros(n_features, dtype=np.float64)
        self.trees_ = []

        for _ in range(self.n_estimators):
            probs = _sigmoid(raw_logits)
            g = probs - y_float
            h = np.maximum(probs * (1.0 - probs), 1e-6)

            if self.subsample < 1.0:
                sub_size = int(n_samples * self.subsample)
                sub_idx = rng.choice(n_samples, size=sub_size, replace=False)
                tree = self._build_newton_tree(
                    X[sub_idx], g[sub_idx], h[sub_idx], 0, thresholds_per_feat, gain_importance
                )
            else:
                tree = self._build_newton_tree(X, g, h, 0, thresholds_per_feat, gain_importance)

            update = self._predict_tree(tree, X)
            raw_logits += self.learning_rate * update
            self.trees_.append(tree)

        total_gain = float(np.sum(gain_importance))
        self.feature_importances_ = (
            gain_importance / total_gain if total_gain > 0 else np.ones(n_features) / n_features
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._xgb_model is not None:
            return self._xgb_model.predict_proba(X)
        logits = np.full(X.shape[0], self.base_logit_, dtype=np.float64)
        for tree in self.trees_:
            logits += self.learning_rate * self._predict_tree(tree, X)
        p1 = _sigmoid(logits)
        return np.column_stack([1.0 - p1, p1])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score, ECE, and calibration curve."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # 1. ROC-AUC via Mann-Whitney U / rank statistic (exact, O(N log N))
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos > 0 and n_neg > 0:
        ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
        # Handle tied probabilities by averaging ranks
        _, inv, counts = np.unique(y_prob[order], return_inverse=True, return_counts=True)
        cum_counts = np.cumsum(counts)
        start_ranks = cum_counts - counts + 1
        avg_ranks = 0.5 * (start_ranks + cum_counts)
        tied_ranks = avg_ranks[inv]
        sum_pos_ranks = float(np.sum(tied_ranks[y_sorted == 1]))
        roc_auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        roc_auc = 0.5

    # 2. PR-AUC (Average Precision) via descending probability sort
    desc_order = np.argsort(-y_prob)
    y_desc = y_true[desc_order]
    tp_cum = np.cumsum(y_desc == 1).astype(np.float64)
    fp_cum = np.cumsum(y_desc == 0).astype(np.float64)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    recalls = tp_cum / max(n_pos, 1)
    # Integrate precision-recall curve at positive hits
    pos_indices = np.where(y_desc == 1)[0]
    pr_auc = float(np.mean(precisions[pos_indices])) if len(pos_indices) > 0 else 0.0

    # 3. Thresholded metrics (Precision, Recall, F1, Accuracy)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)

    # 4. Brier Score & Calibration Curve
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    calibration_bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            mean_obs = float(np.mean(y_true[mask]))
            ece += (cnt / len(y_true)) * abs(mean_pred - mean_obs)
            calibration_bins.append({
                "bin_index": b,
                "bin_lower": round(float(bin_edges[b]), 2),
                "bin_upper": round(float(bin_edges[b + 1]), 2),
                "count": cnt,
                "mean_predicted_prob": round(mean_pred, 4),
                "observed_churn_rate": round(mean_obs, 4),
            })

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "brier_score": round(float(brier_score), 4),
        "expected_calibration_error": round(float(ece), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "calibration_curve": calibration_bins,
    }


def train_and_evaluate_churn_models(
    df: pd.DataFrame,
    nlp_feature_columns: List[str],
    config: Dict[str, Any] | None = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train Logistic Regression and XGBoost (with and without NLP features) and compare."""
    if config is None:
        config = load_config()

    churn_cfg = config["models"]["churn"]
    base_features = list(NUMERIC_CORE_FEATURES)
    full_features = base_features + list(nlp_feature_columns)

    train_df = df[df["dataset_split"] == "train"].copy()
    val_df = df[df["dataset_split"] == "val"].copy()
    test_df = df[df["dataset_split"] == "test"].copy()

    y_train = train_df["churned"].to_numpy(dtype=np.int64)
    y_val = val_df["churned"].to_numpy(dtype=np.int64)
    y_test = test_df["churned"].to_numpy(dtype=np.int64)

    # 1. Preprocessors for Core-Only vs Core+NLP
    prep_core = StandardPreprocessor(base_features).fit(train_df)
    prep_full = StandardPreprocessor(full_features).fit(train_df)

    X_train_core = prep_core.transform(train_df)
    X_val_core = prep_core.transform(val_df)
    X_test_core = prep_core.transform(test_df)

    X_train_full = prep_full.transform(train_df)
    X_val_full = prep_full.transform(val_df)
    X_test_full = prep_full.transform(test_df)

    # 2. Train 4 Model Variants (Logistic & XGBoost x Core vs Full+NLP)
    logger.info("Training Logistic Regression (Core features vs Core+NLP features)...")
    lr_core = LogisticChurnClassifier(**churn_cfg["logistic_regression"]).fit(X_train_core, y_train)
    lr_full = LogisticChurnClassifier(**churn_cfg["logistic_regression"]).fit(X_train_full, y_train)

    logger.info("Training XGBoost / GBDT Classifier (Core features vs Core+NLP features)...")
    xgb_core = XGBoostStyleChurnClassifier(**churn_cfg["xgboost"]).fit(X_train_core, y_train)
    xgb_full = XGBoostStyleChurnClassifier(**churn_cfg["xgboost"]).fit(X_train_full, y_train)

    models_dict = {
        "logistic_regression_no_nlp": (lr_core, prep_core, X_val_core, X_test_core),
        "logistic_regression_with_nlp": (lr_full, prep_full, X_val_full, X_test_full),
        "xgboost_no_nlp": (xgb_core, prep_core, X_val_core, X_test_core),
        "xgboost_with_nlp": (xgb_full, prep_full, X_val_full, X_test_full),
    }

    evaluation_results: Dict[str, Any] = {}
    for name, (model, _, X_v, X_t) in models_dict.items():
        val_probs = model.predict_proba(X_v)[:, 1]
        test_probs = model.predict_proba(X_t)[:, 1]
        evaluation_results[name] = {
            "validation": compute_classification_metrics(y_val, val_probs),
            "test": compute_classification_metrics(y_test, test_probs),
        }
        logger.info(
            f"Model [{name}] Test ROC-AUC={evaluation_results[name]['test']['roc_auc']:.4f} | "
            f"PR-AUC={evaluation_results[name]['test']['pr_auc']:.4f} | "
            f"F1={evaluation_results[name]['test']['f1_score']:.4f} | "
            f"Brier={evaluation_results[name]['test']['brier_score']:.4f}"
        )

    # Generate full-dataset customer churn probabilities from the champion models
    X_all_full = prep_full.transform(df)
    df_out = df.copy()
    df_out["pred_churn_prob_logistic"] = np.round(lr_full.predict_proba(X_all_full)[:, 1], 5)
    df_out["pred_churn_prob_xgb"] = np.round(xgb_full.predict_proba(X_all_full)[:, 1], 5)
    # Primary champion churn score
    df_out["pred_churn_prob"] = df_out["pred_churn_prob_xgb"]
    df_out["churn_risk_tier"] = np.where(
        df_out["pred_churn_prob"] >= 0.60,
        "High Risk",
        np.where(df_out["pred_churn_prob"] >= 0.30, "Medium Risk", "Low Risk"),
    )

    if save_artifacts:
        artifacts_dir = get_project_root() / "models" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "churn_models_bundle.pkl", "wb") as f:
            pickle.dump(
                {
                    "preprocessor_core": prep_core,
                    "preprocessor_full": prep_full,
                    "logistic_core": lr_core,
                    "logistic_full": lr_full,
                    "xgboost_core": xgb_core,
                    "xgboost_full": xgb_full,
                    "base_features": base_features,
                    "full_features": full_features,
                },
                f,
            )

        reports_dir = get_project_root() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "churn_evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2)

    return {
        "scored_df": df_out,
        "metrics": evaluation_results,
        "models": {
            "preprocessor_core": prep_core,
            "preprocessor_full": prep_full,
            "logistic_core": lr_core,
            "logistic_full": lr_full,
            "xgboost_core": xgb_core,
            "xgboost_full": xgb_full,
            "base_features": base_features,
            "full_features": full_features,
        },
    }
