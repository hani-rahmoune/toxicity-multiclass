
# ChemBERTa training code (not the model)
# Runs: train epochs + validate + early stopping + save best checkpoint
# Handles: missing labels via mask, class imbalance (inside model), AMP, scheduler, metrics

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# threshold can be one float (global) or a list/array (per task)
ThresholdType = Union[float, List[float], np.ndarray]


class ChemBERTaTrainer:
    # Trainer = wrapper around the whole training process
    # Keeps model, optimizer, scheduler, history, and best checkpoint logic
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        output_dir: str = "models/chemberta",
        threshold: float = 0.5,
        use_scheduler: bool = True,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        use_amp: bool = True,
        show_progress: bool = True,
    ):
        # pick GPU if available, else CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # move model to device
        self.model = model.to(self.device)

        # store loaders (they yield batches)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training knobs
        self.max_grad_norm = float(max_grad_norm)  # gradient clipping strength
        self.threshold = float(threshold)          # default threshold for metrics
        self.show_progress = bool(show_progress)   # tqdm progress bars

        # where to save checkpoints + history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AMP: faster training on GPU, lower memory
        # scaler prevents float16 gradient underflow
        self.use_amp = bool(use_amp) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # build optimizer with 2 param groups:
        # - encoder gets smaller LR (safer)
        # - classifier head can use larger LR (learn faster)
        head_params = []
        encoder_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(p)
            else:
                encoder_params.append(p)

        # transformer encoders usually need small LR (avoid wrecking pretrained weights)
        encoder_lr = min(float(learning_rate), 3e-5)
        head_lr = float(learning_rate)

        # AdamW = standard for transformers (decoupled weight decay)
        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=float(weight_decay),
            eps=1e-8,
        )

        # scheduler lowers LR when val macro F1 stops improving
        self.use_scheduler = bool(use_scheduler)
        self.scheduler = None
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",                    # we want to maximize F1
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
            )

        # tracking best model + early stopping + history
        self.current_epoch = 0
        self.best_val_f1 = -float("inf")
        self.epochs_without_improvement = 0
        self.training_history: List[Dict] = []

    def train_epoch(self) -> Dict[str, float]:
        # one full pass over the training loader
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        # tqdm for visuals (optional)
        iterator = self.train_loader
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Epoch {self.current_epoch + 1} [Train]", leave=False)

        # clear gradients before starting epoch
        self.optimizer.zero_grad(set_to_none=True)

        for batch in iterator:
            # batch tensors from Dataset/DataLoader
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_mask = batch["label_mask"].to(self.device)

            # forward pass (autocast uses float16/bfloat16 when enabled)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_mask=label_mask,
                )
                loss = outputs["loss"]  # model computes masked loss internally

            # if model didn't compute loss, training cannot continue
            if loss is None:
                raise ValueError("Model returned loss=None during training. Provide labels/label_mask.")

            # backward with scaling (AMP)
            self.scaler.scale(loss).backward()

            # unscale then clip gradients (stability)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # optimizer step (AMP-safe)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # reset grads for next batch
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            n_batches += 1

        # average loss over all batches
        avg_loss = total_loss / n_batches if n_batches else 0.0
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        task_names: Optional[List[str]] = None,
        threshold: Optional[ThresholdType] = None,
    ) -> Dict:
        # evaluation loop: no gradients, model in eval mode
        self.model.eval()

        # collect outputs across all batches
        all_logits = []
        all_labels = []
        all_masks = []

        total_loss = 0.0
        n_batches = 0

        # use provided threshold or default trainer threshold
        if threshold is None:
            threshold = self.threshold

        iterator = dataloader
        if self.show_progress:
            iterator = tqdm(iterator, desc="Evaluating", leave=False)

        for batch in iterator:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_mask = batch["label_mask"].to(self.device)

            # forward pass (still can use autocast for speed)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_mask=label_mask,
                )

            # store everything on CPU for sklearn metrics
            all_logits.append(outputs["logits"].detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(label_mask.detach().cpu())

            # average eval loss (optional)
            if outputs["loss"] is not None:
                total_loss += float(outputs["loss"].item())
                n_batches += 1

        # stack all batches into full arrays
        logits = torch.cat(all_logits, dim=0)
        labels_np = torch.cat(all_labels, dim=0).numpy()
        mask_np = torch.cat(all_masks, dim=0).numpy().astype(bool)

        # sigmoid converts logits → probabilities
        probs = torch.sigmoid(logits).numpy()

        # threshold converts probabilities → 0/1 predictions
        preds = self._apply_threshold(probs, threshold)

        # compute masked metrics (ignore missing labels)
        metrics = self._compute_metrics(
            labels=labels_np,
            predictions=preds,
            probabilities=probs,
            label_masks=mask_np,
            task_names=task_names,
        )

        # attach extra info
        metrics["loss"] = total_loss / n_batches if n_batches else 0.0
        metrics["threshold"] = self._threshold_to_serializable(threshold)

        return metrics

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 3,
        task_names: Optional[List[str]] = None,
    ):
        # full training loop over epochs
        num_epochs = int(num_epochs)
        early_stopping_patience = int(early_stopping_patience)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # train then validate
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, task_names=task_names)

            # macro F1 is the main selection metric
            val_f1 = float(val_metrics["aggregate"]["macro_f1"])

            # scheduler updates based on validation F1
            if self.use_scheduler and self.scheduler is not None:
                self.scheduler.step(val_f1)

            # log current learning rates (encoder/head)
            lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
            if len(lrs) >= 2:
                lr_str = f"{lrs[0]:.2e}/{lrs[1]:.2e}"
            else:
                lr_str = f"{lrs[0]:.2e}" if lrs else "0.00e+00"

            # store metrics for later plotting/reporting
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_metrics["train_loss"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_macro_f1": float(val_f1),
                    "val_macro_roc_auc": float(val_metrics["aggregate"]["macro_roc_auc"]),
                    "learning_rate": float(lrs[0]) if lrs else 0.0,
                    "threshold": val_metrics["threshold"],
                }
            )

            # print one summary line per epoch
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_macro_f1={val_f1:.4f} | "
                f"lr(enc/head)={lr_str}"
            )

            # save best model by macro F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
                self.save_checkpoint(metrics=val_metrics, filename="best_model.pt")
            else:
                self.epochs_without_improvement += 1

            # stop if no improvement for N epochs
            if self.epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping: no improvement for {early_stopping_patience} epochs. "
                    f"Best val macro F1 = {self.best_val_f1:.4f}"
                )
                break

        # save full history at the end
        self.save_training_history()

    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        label_masks: np.ndarray,
        task_names: Optional[List[str]] = None,
    ) -> Dict:
        # compute per-task metrics using only valid labels (mask)
        n_tasks = labels.shape[1]
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(n_tasks)]

        per_task = {}

        for i, name in enumerate(task_names):
            valid = label_masks[:, i]
            if valid.sum() == 0:
                continue

            y_true = labels[valid, i]
            y_pred = predictions[valid, i]
            y_prob = probabilities[valid, i]

            # f1/precision/recall depend on thresholded preds
            task_metrics = {
                "n_samples": int(valid.sum()),
                "n_positive": int(y_true.sum()),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }

            # ROC-AUC needs both classes present
            if len(np.unique(y_true)) > 1:
                task_metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                task_metrics["roc_auc"] = float("nan")

            per_task[name] = task_metrics

        # macro averages across tasks
        f1s = [m["f1"] for m in per_task.values()]
        precs = [m["precision"] for m in per_task.values()]
        recs = [m["recall"] for m in per_task.values()]
        aucs = [m["roc_auc"] for m in per_task.values() if not np.isnan(m["roc_auc"])]

        aggregate = {
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "macro_precision": float(np.mean(precs)) if precs else 0.0,
            "macro_recall": float(np.mean(recs)) if recs else 0.0,
            "macro_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        }

        return {"per_task": per_task, "aggregate": aggregate}

    @staticmethod
    def _apply_threshold(probs: np.ndarray, threshold: ThresholdType) -> np.ndarray:
        # supports either one threshold or one per task
        if isinstance(threshold, (list, tuple, np.ndarray)):
            thr = np.asarray(threshold, dtype=float).reshape(1, -1)
            return (probs >= thr).astype(int)
        return (probs >= float(threshold)).astype(int)

    @staticmethod
    def _threshold_to_serializable(threshold: ThresholdType):
        # JSON-friendly threshold storage
        if isinstance(threshold, (list, tuple, np.ndarray)):
            return [float(x) for x in np.asarray(threshold).ravel().tolist()]
        return float(threshold)

    def save_checkpoint(self, metrics: Dict, filename: str = "checkpoint.pt"):
        # save model + optimizer (+ scheduler) + metrics snapshot
        path = self.output_dir / filename
        payload = {
            "epoch": int(self.current_epoch),
            "best_val_f1": float(self.best_val_f1),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.use_scheduler and self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str = "best_model.pt"):
        # restore model + optimizer (+ scheduler) state
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.use_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_f1 = float(checkpoint.get("best_val_f1", -float("inf")))
        print(f"Loaded checkpoint: {path}")

    def save_training_history(self, filename: str = "training_history.json"):
        # save epoch-by-epoch metrics for plotting/reporting
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Saved training history: {path}")


# small helpers for reporting how many params are frozen/trainable
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": int(total), "trainable": int(trainable), "frozen": int(frozen)}


def print_model_info(model: nn.Module):
    params = count_parameters(model)
    pct = (params["trainable"] / params["total"] * 100.0) if params["total"] else 0.0
    print("MODEL INFORMATION")
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,} ({pct:.1f}%)")
    print(f"Frozen parameters:    {params['frozen']:,}")


if __name__ == "__main__":
    # just a message if you run this file directly
    print("ChemBERTa Trainer Module (Epoch logging)")
    print("Import ChemBERTaTrainer and use it with your model and DataLoaders.")
