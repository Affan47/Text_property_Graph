"""
Training & Evaluation — GNN-based EPSS Prediction
====================================================
Trains the EPSSGraphClassifier on CVE TPG graphs with:
    - Weighted BCE loss (handles ~5% class imbalance)
    - PR-AUC as primary metric (consistent with EPSS v1/v2/v3 papers)
    - F1, Brier score, calibration evaluation
    - Early stopping on validation PR-AUC

Usage:
    trainer = Trainer(dataset, model, device="cuda")
    trainer.train(epochs=100)
    results = trainer.evaluate(split="test")
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from epss.gnn_model import EPSSGraphClassifier, HybridEPSSClassifier

logger = logging.getLogger(__name__)


class Trainer:
    """Training and evaluation for the EPSS GNN model.

    Args:
        dataset: CVEGraphDataset instance.
        model: EPSSGraphClassifier instance.
        device: 'cuda' or 'cpu'.
        output_dir: Directory to save checkpoints and results.
        lr: Learning rate.
        weight_decay: L2 regularization.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without improvement).
    """

    def __init__(
        self,
        dataset,
        model: nn.Module,
        device: str = "cpu",
        output_dir: str = "output/epss",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        patience: int = 15,
    ):
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.patience = patience

        # Class weights for imbalanced data
        class_weights = dataset.get_class_weights()
        self.pos_weight = torch.tensor(
            [class_weights[1] / class_weights[0]], device=device
        )
        logger.info("Class weights: neg=%.3f, pos=%.3f (pos_weight=%.3f)",
                     class_weights[0], class_weights[1], self.pos_weight.item())

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )

        # Data splits
        splits = dataset.get_split_indices()
        self.train_loader = DataLoader(
            dataset[splits["train"]], batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset[splits["val"]], batch_size=batch_size
        )
        self.test_loader = DataLoader(
            dataset[splits["test"]], batch_size=batch_size
        )

        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )

    def train(self, epochs: int = 100) -> Dict[str, list]:
        """Train the model with early stopping on validation PR-AUC.

        Returns:
            history dict with train_loss, val_loss, val_prauc per epoch.
        """
        history = {"train_loss": [], "val_loss": [], "val_prauc": [], "val_f1": []}
        best_prauc = 0.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_epoch()

            # Validate
            val_metrics = self._evaluate(self.val_loader)

            # LR scheduling
            self.scheduler.step(val_metrics["pr_auc"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_prauc"].append(val_metrics["pr_auc"])
            history["val_f1"].append(val_metrics["f1"])

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
                "val_PR-AUC=%.4f | val_F1=%.4f | lr=%.2e",
                epoch, train_loss, val_metrics["loss"],
                val_metrics["pr_auc"], val_metrics["f1"], lr,
            )

            # Early stopping
            if val_metrics["pr_auc"] > best_prauc:
                best_prauc = val_metrics["pr_auc"]
                best_epoch = epoch
                no_improve = 0
                self._save_checkpoint("best_model.pt", epoch, val_metrics)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (best PR-AUC=%.4f at epoch %d)",
                        epoch, best_prauc, best_epoch,
                    )
                    break

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def _train_epoch(self) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch).squeeze(-1)
            labels = batch.y.float().squeeze(-1)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader.

        Works for both EPSSGraphClassifier and HybridEPSSClassifier.
        The tabular tensor is automatically batched by PyG's DataLoader
        when present on Data objects (concatenated along dim=0).
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch).squeeze(-1)
                labels = batch.y.float().squeeze(-1)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels, dtype=int)

        metrics = compute_metrics(all_labels, all_probs)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def evaluate_test(self) -> Dict[str, float]:
        """Final evaluation on the held-out test set.

        Loads the best checkpoint and evaluates.
        """
        ckpt_path = self.output_dir / "best_model.pt"
        if ckpt_path.exists():
            self._load_checkpoint(ckpt_path)

        metrics = self._evaluate(self.test_loader)

        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info("PR-AUC:      %.4f", metrics["pr_auc"])
        logger.info("ROC-AUC:     %.4f", metrics["roc_auc"])
        logger.info("F1:          %.4f", metrics["f1"])
        logger.info("Precision:   %.4f", metrics["precision"])
        logger.info("Recall:      %.4f", metrics["recall"])
        logger.info("Brier Score: %.4f", metrics["brier"])
        logger.info("=" * 60)

        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict):
        path = self.output_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", "?"))


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all evaluation metrics consistent with EPSS papers.

    Metrics:
        - PR-AUC (primary metric — used by EPSS v1/v2/v3)
        - ROC-AUC
        - F1 score (at given threshold)
        - Precision (EPSS "efficiency")
        - Recall (EPSS "coverage")
        - Brier score (calibration quality)

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities (0.0 to 1.0).
        threshold: Decision threshold for binary predictions.

    Returns:
        dict of metric name -> value.
    """
    from sklearn.metrics import (
        precision_recall_curve, auc,
        roc_auc_score, f1_score,
        precision_score, recall_score,
        brier_score_loss,
    )

    # PR-AUC (primary metric)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    # Binary predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }
