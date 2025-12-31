# Multi-label MLP baseline for Tox21 toxicity prediction
# ------------------------------------------------------
# - Binary relevance: one MLP per assay
# - Uses molecular fingerprints (e.g. 2048-bit Morgan)
# - Handles missing labels (NaNs)
# - Global feature scaling (important for neural nets)
# - Neural baseline between XGBoost and transformers


import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Tox21 assay/task names (12 tasks)
TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# Multi-label classifier using Binary Relevance with MLPs
# Each assay is trained as an independent binary neural network
class MultiLabelMLP:
    # Initialize model configuration and hyperparameters
    # No training happens here
    def __init__(
        self,
        task_names: List[str],
        hidden_layers: Tuple[int, ...] = (512, 256),
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int = 42,
        verbose: int = 1,
        min_samples_per_task: int = 50,
    ):
        self.task_names = task_names
        self.n_tasks = len(task_names)

        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.verbose = verbose
        self.min_samples_per_task = min_samples_per_task

        # Per-task models (trained in fit)
        self.models: Dict[str, MLPClassifier] = {}

        # One global scaler fitted in fit()
        self.scaler: Optional[StandardScaler] = None

        # Optional: stored class weights for inspection
        self.class_weights: Dict[str, Dict[int, float]] = {}

        if self.verbose:
            arch = " → ".join(map(str, (2048,) + self.hidden_layers + (1,)))
            print(" MultiLabelMLP initialized")
            print(f"   Tasks: {self.n_tasks}")
            print(f"   Architecture (typical): {arch}")

    # Compute balanced class weights for a single assay
    # multiply the loss with a bigger scaller when its a false negative 
    @staticmethod
    def _compute_class_weight_dict(y_task_clean: np.ndarray) -> Dict[int, float]:
        n = int(len(y_task_clean))
        if n == 0:
            return {0: 1.0, 1: 1.0}

        pos = int((y_task_clean == 1).sum())
        neg = int((y_task_clean == 0).sum())

        if pos == 0 or neg == 0:
            return {0: 1.0, 1: 1.0}

        w_pos = n / (2.0 * pos)
        w_neg = n / (2.0 * neg)
        return {0: float(w_neg), 1: float(w_pos)}

    # Safely (does not crash) compute ROC-AUC (returns NaN if only one class is present)
    @staticmethod
    def _safe_metric_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    # Safely compute PR-AUC (robust to undefined cases)
    @staticmethod
    def _safe_metric_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    # Train one MLPClassifier per assay (Binary Relevance)
    # - Fits a single global StandardScaler
    # - Ignores samples with missing labels
    # - Applies class imbalance weighting when supported
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLabelMLP":
        if self.verbose:
            print("\n" + "=" * 80)
            print("TRAINING MULTI-LABEL MLP")
            print("=" * 80)

        # Fit global scaler and transform all features once
        self.scaler = StandardScaler()
        X_scaled_all = self.scaler.fit_transform(X)

        for task_idx, task_name in enumerate(self.task_names):
            y_task = y[:, task_idx]
            valid_mask = ~np.isnan(y_task)
            n_valid = int(valid_mask.sum())

            if self.verbose:
                print(f"\n[{task_idx + 1}/{self.n_tasks}] {task_name}")
                print("-" * 60)
                print(f"   Labeled samples: {n_valid:,} / {len(y_task):,}")

            if n_valid < self.min_samples_per_task:
                if self.verbose:
                    print(f"     Skipping (need ≥ {self.min_samples_per_task} labeled samples)")
                continue

            X_task = X_scaled_all[valid_mask]
            y_task_clean = y_task[valid_mask].astype(int)

            pos = int((y_task_clean == 1).sum())
            neg = int((y_task_clean == 0).sum())
            if self.verbose:
                rate = (pos / (pos + neg) * 100) if (pos + neg) > 0 else 0.0
                print(f"   Positive: {pos:,} ({rate:.1f}%) | Negative: {neg:,}")

            # Compute imbalance weights
            cw = self._compute_class_weight_dict(y_task_clean)
            self.class_weights[task_name] = cw
            sample_weight = np.array([cw[int(lbl)] for lbl in y_task_clean], dtype=float)

            # Initialize per-task MLP
            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                learning_rate="adaptive",
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
                verbose=False,
            )

            # Train (use sample_weight if supported by sklearn version)
            try:
                clf.fit(X_task, y_task_clean, sample_weight=sample_weight)
            except TypeError:
                if self.verbose:
                    print("     sample_weight not supported; training without it.")
                clf.fit(X_task, y_task_clean)

            if self.verbose:
                print(f"    Trained: n_iter={clf.n_iter_} | loss={clf.loss_:.4f}")

            self.models[task_name] = clf

        if self.verbose:
            print("\n" + "=" * 80)
            print(f" TRAINING COMPLETE ({len(self.models)}/{self.n_tasks} assays trained)")
            print("=" * 80)

        return self

    # Predict toxicity probabilities for all assays
    # Output shape: (n_samples, n_tasks)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Model not fitted: scaler missing. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = int(X.shape[0])
        probas = np.full((n_samples, self.n_tasks), np.nan, dtype=float)

        for task_idx, task_name in enumerate(self.task_names):
            clf = self.models.get(task_name)
            if clf is None:
                continue
            probas[:, task_idx] = clf.predict_proba(X_scaled)[:, 1]

        return probas

    # Convert probabilities into binary predictions using a threshold
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        preds = np.full_like(probas, np.nan)
        mask = ~np.isnan(probas)
        preds[mask] = (probas[mask] >= threshold).astype(int)
        return preds

    # Evaluate model performance per assay and compute macro averages
    # Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        probas = self.predict_proba(X)
        preds = self.predict(X, threshold=threshold)

        per_task: Dict[str, Dict[str, float]] = {}

        for t, name in enumerate(self.task_names):
            y_true = y[:, t]
            valid = ~np.isnan(y_true)

            # No labels or model not trained
            if valid.sum() == 0 or np.all(np.isnan(probas[:, t])):
                per_task[name] = {
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "n": int(valid.sum()),
                }
                continue

            yt = y_true[valid].astype(int)
            ys = probas[valid, t]
            yp = preds[valid, t].astype(int)

            per_task[name] = {
                "roc_auc": self._safe_metric_roc_auc(yt, ys),
                "pr_auc": self._safe_metric_pr_auc(yt, ys),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "n": int(len(yt)),
            }

        # Macro averages ignoring NaNs
        def _nanmean(key: str) -> float:
            vals = np.array([per_task[k][key] for k in per_task.keys()], dtype=float)
            return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")

        summary = {
            "macro_roc_auc": _nanmean("roc_auc"),
            "macro_pr_auc": _nanmean("pr_auc"),
            "macro_precision": _nanmean("precision"),
            "macro_recall": _nanmean("recall"),
            "macro_f1": _nanmean("f1"),
            "n_tasks_trained": float(len(self.models)),
        }

        return {"per_task": per_task, "summary": summary}

    # Save scaler, trained models, and metadata to disk
    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        with open(p / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(p / "models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        meta = {
            "task_names": self.task_names,
            "hidden_layers": list(self.hidden_layers),
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "random_state": self.random_state,
            "min_samples_per_task": self.min_samples_per_task,
            "class_weights": self.class_weights,
        }

        with open(p / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f" Saved to: {p}")

    # Load a previously saved MultiLabelMLP model
    @classmethod
    def load(cls, path: str, verbose: int = 1) -> "MultiLabelMLP":
        p = Path(path)

        with open(p / "metadata.json", "r") as f:
            meta = json.load(f)

        model = cls(
            task_names=meta["task_names"],
            hidden_layers=tuple(meta["hidden_layers"]),
            alpha=meta["alpha"],
            learning_rate_init=meta["learning_rate_init"],
            max_iter=meta["max_iter"],
            early_stopping=meta["early_stopping"],
            validation_fraction=meta["validation_fraction"],
            n_iter_no_change=meta["n_iter_no_change"],
            random_state=meta["random_state"],
            verbose=verbose,
            min_samples_per_task=int(meta.get("min_samples_per_task", 50)),
        )
        model.class_weights = meta.get("class_weights", {})

        with open(p / "scaler.pkl", "rb") as f:
            model.scaler = pickle.load(f)

        with open(p / "models.pkl", "rb") as f:
            model.models = pickle.load(f)

        if verbose:
            print(f" Loaded from: {p} | assays trained: {len(model.models)}")

        return model


# sanity check with random data 
if __name__ == "__main__":
    np.random.seed(42)

    n_train = 1000
    n_test = 200
    n_features = 2048
    n_tasks = 12

    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, 2, size=(n_train, n_tasks)).astype(float)

    # Inject missing labels (NaNs)
    y_train[np.random.rand(*y_train.shape) < 0.10] = np.nan

    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, 2, size=(n_test, n_tasks)).astype(float)
    y_test[np.random.rand(*y_test.shape) < 0.10] = np.nan

    model = MultiLabelMLP(task_names=TOX21_ASSAYS, verbose=1)
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)
    preds = model.predict(X_test, threshold=0.5)
    report = model.evaluate(X_test, y_test, threshold=0.5)

    print("\n Test complete")
    print(f"   probas shape: {probas.shape}")
    print(f"   preds  shape: {preds.shape}")
    print("   summary:", report["summary"])
