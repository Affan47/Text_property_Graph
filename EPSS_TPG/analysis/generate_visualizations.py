"""
Generate full visualization suite for a trained EPSS-GNN checkpoint.

Handles the tabular dimension mismatch: loads the checkpoint, detects the
trained tabular_dim from the saved weights, slices the dataset's tabular
features to match, then generates all plots.

Usage:
    python generate_visualizations.py --ckpt-dir output/epss_multiview_hybrid
    python generate_visualizations.py --ckpt-dir output/epss_multiview_hybrid --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def get_tabular_dim_from_checkpoint(ckpt_path: str) -> int:
    """Detect tabular_dim from checkpoint weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    key = "tabular_encoder.0.weight"
    if key in state:
        return state[key].shape[1]
    return 0


class TabularSliceWrapper(nn.Module):
    """Wraps HybridEPSSClassifier and slices tabular features to target_dim."""

    def __init__(self, model, target_dim: int):
        super().__init__()
        self.model = model
        self.target_dim = target_dim

    def forward(self, data):
        if data.tabular is not None and data.tabular.shape[-1] > self.target_dim:
            data.tabular = data.tabular[..., : self.target_dim]
        return self.model(data)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


def run_inference(model, loader, device, criterion, target_tabular_dim):
    """Run model inference on a data loader, collecting predictions."""
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Slice tabular features to match checkpoint's trained dim
            if hasattr(batch, "tabular") and batch.tabular is not None:
                if batch.tabular.shape[-1] > target_tabular_dim:
                    batch.tabular = batch.tabular[..., :target_tabular_dim]

            logits = model(batch).squeeze(-1)
            labels = batch.y.float().squeeze(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if hasattr(batch, "cve_id"):
                ids = batch.cve_id
                if isinstance(ids, (list, tuple)):
                    all_ids.extend(ids)
                else:
                    all_ids.append(str(ids))

    return (
        np.array(all_probs),
        np.array(all_labels, dtype=int),
        all_ids,
        total_loss / max(n_batches, 1),
    )


def run_for_experiment(ckpt_dir: str, device: str = "cpu"):
    """Generate full visualization suite for a single experiment directory."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best_model.pt"
    config_path = ckpt_dir / "experiment_config.json"
    history_path = ckpt_dir / "training_history.json"

    if not ckpt_path.exists():
        logger.error("No checkpoint at %s", ckpt_path)
        return

    logger.info("=" * 60)
    logger.info("Processing: %s", ckpt_dir.name)
    logger.info("=" * 60)

    # Load experiment config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    args_cfg = config.get("args", {})

    backbone = args_cfg.get("backbone", "gat")
    hidden = args_cfg.get("hidden", 128)
    layers = args_cfg.get("layers", 3)
    heads = args_cfg.get("heads", 4)
    dropout = args_cfg.get("dropout", 0.3)
    in_channels = config.get("in_channels", 781)
    num_edge_types = config.get("num_edge_types", 13)
    label_mode = args_cfg.get("label_mode", "binary")
    embedding_dim = args_cfg.get("embedding_dim", 768)

    labeled_file = args_cfg.get("labeled_file") or "data/epss/labeled_cves_balanced_v2.json"
    data_dir = args_cfg.get("data_dir", "data/epss")

    # Detect tabular dim from checkpoint
    ckpt_tabular_dim = get_tabular_dim_from_checkpoint(str(ckpt_path))
    is_hybrid = ckpt_tabular_dim > 0
    logger.info("Backbone: %s | tabular_dim from checkpoint: %d | hybrid: %s",
                backbone, ckpt_tabular_dim, is_hybrid)

    # Load edge type vocab
    edge_type_vocab = None
    vocab_path = Path(data_dir) / "pyg_dataset" / "processed" / "edge_type_vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            edge_type_vocab = json.load(f)

    # Load training history
    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # Build model with checkpoint's tabular_dim
    from epss.gnn_model import build_model
    model = build_model(
        in_channels=in_channels,
        backbone=backbone,
        hidden_channels=hidden,
        num_layers=layers,
        dropout=dropout,
        num_heads=heads,
        tabular_dim=ckpt_tabular_dim,
        num_edge_types=num_edge_types,
        edge_type_vocab=edge_type_vocab,
    )

    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Loaded checkpoint (epoch %d)", ckpt.get("epoch", "?"))

    # Load dataset — use the labeled file from config
    from epss.cve_dataset import CVEGraphDataset
    dataset = CVEGraphDataset(
        root=str(Path(data_dir) / "pyg_dataset"),
        labeled_cves_path=labeled_file,
        label_mode=label_mode,
        embedding_dim=embedding_dim,
        use_hybrid=True,
        include_tabular=is_hybrid,
    )
    logger.info("Dataset: %d graphs", len(dataset))

    # Get test split (deterministic — same seed as training)
    splits = dataset.get_split_indices(seed=42)
    test_loader = DataLoader(dataset[splits["test"]], batch_size=32)
    logger.info("Test split: %d graphs", len(splits["test"]))

    # Set up loss (pos_weight from dataset class weights)
    class_weights = dataset.get_class_weights()
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Run inference
    y_prob, y_true, cve_ids, test_loss = run_inference(
        model, test_loader, device, criterion, ckpt_tabular_dim
    )
    logger.info("Test inference complete: %d samples, %d positive",
                len(y_true), int(y_true.sum()))

    # Compute metrics
    from epss.train import compute_metrics
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = test_loss

    logger.info("PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f | Brier: %.4f",
                metrics["pr_auc"], metrics["roc_auc"], metrics["f1"], metrics["brier"])

    # Save updated test results
    with open(ckpt_dir / "test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate all visualizations
    from epss.visualize import generate_all
    generate_all(
        cve_ids=cve_ids,
        y_true=y_true,
        y_prob=y_prob,
        history=history,
        metrics=metrics,
        output_dir=ckpt_dir,
        backbone=backbone,
        threshold=metrics.get("threshold", 0.5),
        results_root=ckpt_dir.parent,
    )

    logger.info("Done: %s", ckpt_dir.name)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate EPSS-GNN visualizations")
    parser.add_argument("--ckpt-dir", default="output/epss_multiview_hybrid",
                        help="Path to experiment output dir (contains best_model.pt)")
    parser.add_argument("--all", action="store_true",
                        help="Process all experiment dirs in output/")
    parser.add_argument("--output-root", default="output",
                        help="Root output dir when --all is used")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    if args.all:
        output_root = Path(args.output_root)
        exp_dirs = sorted([
            d for d in output_root.iterdir()
            if d.is_dir() and (d / "best_model.pt").exists()
        ])
        logger.info("Found %d experiment dirs", len(exp_dirs))
        all_metrics = {}
        for exp_dir in exp_dirs:
            try:
                metrics = run_for_experiment(str(exp_dir), device)
                if metrics:
                    all_metrics[exp_dir.name] = metrics
            except Exception as e:
                logger.error("Failed for %s: %s", exp_dir.name, e, exc_info=True)

        logger.info("\n=== Summary ===")
        for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["pr_auc"]):
            logger.info("%-35s  PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
                        name, m["pr_auc"], m["roc_auc"], m["f1"])
    else:
        run_for_experiment(args.ckpt_dir, device)


if __name__ == "__main__":
    main()
