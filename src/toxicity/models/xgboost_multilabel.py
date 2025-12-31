"""
XGBoost baseline module for multi-label toxicity prediction.

This module implements a strong and interpretable baseline using XGBoost
trained on molecular fingerprints. The toxicity task is treated as a
multi-label classification problem, where each biological assay is modeled
independently using a Binary Relevance strategy (one binary classifier per assay).

Key characteristics:
- Handles missing assay labels (NaN) naturally
- Accounts for strong class imbalance via per-task class weighting
- Fast to train and easy to interpret
- Provides a reliable benchmark for comparison with transformer-based models


"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import pickle
import warnings

# XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

# Metrics
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

# Tox21 assay names
TOX21_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


class MultiLabelXGBoost:
    def __init__(
        self,
        task_names: List[str],
        xgb_params: Optional[Dict] = None,
        use_gpu: bool = False,
        verbose: int = 1,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install with: pip install xgboost")

        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.verbose = verbose

        if xgb_params is None:
            xgb_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }

            if use_gpu:
                xgb_params["tree_method"] = "gpu_hist"
                xgb_params["predictor"] = "gpu_predictor"

        self.xgb_params = xgb_params
        self.models: Dict[str, XGBClassifier] = {}
        self.class_weights: Dict[str, float] = {}

        if self.verbose:
            print(f" MultiLabelXGBoost initialized ({self.n_tasks} tasks)")

    @staticmethod
    def _compute_class_weight(y: np.ndarray) -> float:
        y_valid = y[~np.isnan(y)]
        if len(y_valid) == 0:
            return 1.0

        n_pos = (y_valid == 1).sum()
        n_neg = (y_valid == 0).sum()

        if n_pos == 0:
            return 1.0

        return float(n_neg / n_pos)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,  
    ):
        for task_idx, task_name in enumerate(self.task_names):
            y_task = y[:, task_idx]
            valid_mask = ~np.isnan(y_task)

            if valid_mask.sum() < 10:
                if self.verbose:
                    print(f"  Skipping {task_name} (too few labels)")
                continue

            X_task = X[valid_mask]
            y_task_clean = y_task[valid_mask]

            class_weight = self._compute_class_weight(y_task)
            self.class_weights[task_name] = class_weight

            task_params = self.xgb_params.copy()
            task_params["scale_pos_weight"] = class_weight

            clf = XGBClassifier(**task_params)
            clf.fit(X_task, y_task_clean)

            self.models[task_name] = clf

            if self.verbose:
                print(f" Trained {task_name}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        probas = np.full((n_samples, self.n_tasks), np.nan)

        for i, task_name in enumerate(self.task_names):
            if task_name in self.models:
                probas[:, i] = self.models[task_name].predict_proba(X)[:, 1]

        return probas

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        preds = np.full(probas.shape, np.nan)
        mask = ~np.isnan(probas)
        preds[mask] = (probas[mask] >= threshold).astype(int)
        return preds
    
    

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for task, model in self.models.items():
            with open(path / f"{task}.pkl", "wb") as f:
                pickle.dump(model, f)

        metadata = {
            "task_names": self.task_names,
            "xgb_params": self.xgb_params,
            "class_weights": self.class_weights,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MultiLabelXGBoost":
        path = Path(path)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        model = cls(
            task_names=metadata["task_names"],
            xgb_params=metadata["xgb_params"],
            verbose=0,
        )

        model.class_weights = metadata["class_weights"]

        for task in metadata["task_names"]:
            model_path = path / f"{task}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model.models[task] = pickle.load(f)

        return model


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task_names: List[str],
) -> Dict:
    results = {"per_task": {}, "aggregate": {}}

    for i, task in enumerate(task_names):
        y_t = y_true[:, i]
        mask = ~np.isnan(y_t)

        if mask.sum() == 0:
            continue

        y_t = y_t[mask]
        y_p = y_pred[:, i][mask]
        y_pr = y_proba[:, i][mask]

        metrics = {
            "n_samples": len(y_t),
            "f1": f1_score(y_t, y_p, zero_division=0),
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
        }

        if len(np.unique(y_t)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_t, y_pr)
            metrics["pr_auc"] = average_precision_score(y_t, y_pr)
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan

        results["per_task"][task] = metrics

    results["aggregate"] = {
        "macro_f1": np.mean([m["f1"] for m in results["per_task"].values()]),
        "macro_precision": np.mean(
            [m["precision"] for m in results["per_task"].values()]
        ),
        "macro_recall": np.mean([m["recall"] for m in results["per_task"].values()]),
    }

    return results
