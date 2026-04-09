"""
EPSS-GNN Temporal Inference & Ground-Truth Verification
=========================================================
Score unseen CVEs with a trained EPSS-GNN checkpoint and verify predictions
against real-world ground truth (CISA KEV + FIRST EPSS API).

Three usage modes:

  1. POST-DATASET  — CVEs published after the training set ends (2025-06-01).
     These are completely unseen by the model; KEV status by today (2026-04-09)
     serves as exploit ground truth.

       python -m epss.infer --mode post-dataset \\
           --after-date 2025-07-01 --before-date 2025-09-30 \\
           --checkpoint output/epss_sec4ai/best_model.pt \\
           --config    output/epss_sec4ai/experiment_config.json

  2. PRE-DATASET   — CVEs published before the earliest training record (2021-11-01).
     These are historically old; KEV catalogue is fully settled.

       python -m epss.infer --mode pre-dataset \\
           --after-date 2019-01-01 --before-date 2021-10-31 \\
           --checkpoint output/epss_sec4ai/best_model.pt \\
           --config    output/epss_sec4ai/experiment_config.json \\
           --max-cves 500

  3. CUSTOM LIST   — Arbitrary CVE IDs (comma-separated or from a file).

       python -m epss.infer --mode custom \\
           --cve-ids CVE-2025-31200,CVE-2025-30065 \\
           --checkpoint output/epss_sec4ai/best_model.pt \\
           --config    output/epss_sec4ai/experiment_config.json

Output:
  --output-dir   (default: output/infer/<mode>/)
  predictions_infer.csv     — one row per CVE with risk tier + ground truth
  verification_summary.txt  — aggregated precision/recall vs KEV
"""

import argparse
import json
import logging
import sys
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from epss.data_collector import DataCollector
from epss.cve_dataset import CVEGraphDataset
from epss.gnn_model import build_model

logger = logging.getLogger(__name__)

# ── Risk tier thresholds (mirrors visualize.py) ────────────────────────────────
TIERS = [
    (0.90, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MEDIUM"),
    (0.30, "LOW"),
    (0.00, "MINIMAL"),
]

# ── Dataset date range constants ───────────────────────────────────────────────
DATASET_START = "2021-11-23"   # earliest CVE in training data
DATASET_END   = "2025-06-01"   # latest  CVE in training data


def risk_tier(prob: float) -> str:
    for thresh, label in TIERS:
        if prob >= thresh:
            return label
    return "MINIMAL"


# ── NVD date-range fetch ───────────────────────────────────────────────────────

def fetch_nvd_by_date(
    after: str,
    before: str,
    data_dir: Path,
    max_cves: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Dict[str, dict]:
    """Fetch NVD CVE records published in [after, before] (YYYY-MM-DD).

    Returns labeled_cves.json-compatible dict: CVE-ID → record.
    Uses DataCollector._fetch_single_cve under the hood so all parsing
    is identical to training-time data collection.
    """
    import requests

    NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    session  = requests.Session()
    if api_key:
        session.headers["apiKey"] = api_key

    results: Dict[str, dict] = {}
    start_idx = 0
    page_size = 2000

    logger.info("Fetching NVD CVEs published %s → %s", after, before)

    # One DataCollector instance is enough — reuse _parse_nvd_record
    collector = DataCollector(output_dir=str(data_dir), nvd_api_key=api_key)
    delay = 0.6 if api_key else 6.0

    # NVD API 2.0 max range is 120 days — split long windows into monthly chunks
    from datetime import date, timedelta

    def _date_chunks(start_str: str, end_str: str, chunk_days: int = 90):
        """Yield (start, end) pairs of at most chunk_days each."""
        cur = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        while cur <= end:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
            yield str(cur), str(chunk_end)
            cur = chunk_end + timedelta(days=1)

    for chunk_start, chunk_end in _date_chunks(after, before):
        chunk_start_idx = 0
        logger.info("  Chunk: %s → %s", chunk_start, chunk_end)

        while True:
            params = {
                "pubStartDate": f"{chunk_start}T00:00:00.000",
                "pubEndDate":   f"{chunk_end}T23:59:59.999",
                "startIndex":   chunk_start_idx,
                "resultsPerPage": page_size,
            }
            try:
                resp = session.get(NVD_API, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning("NVD page offset=%d failed: %s — retrying in 10s",
                               chunk_start_idx, e)
                time.sleep(10)
                continue

            vulns = data.get("vulnerabilities", [])
            total = data.get("totalResults", 0)

            for item in vulns:
                cve_obj = item.get("cve", {})
                # CVE ID lives at cve.id, NOT inside _parse_nvd_record output
                cve_id = cve_obj.get("id", "").strip()
                if not cve_id:
                    continue
                record = collector._parse_nvd_record(cve_obj)
                if not record or not record.get("description", "").strip():
                    continue
                # Add cve_id + inference defaults (fields needed by tabular encoder)
                record["cve_id"] = cve_id
                record.setdefault("binary_label", 0)
                record.setdefault("epss_score", 0.0)
                record.setdefault("epss_percentile", 0.0)
                record.setdefault("in_kev", False)
                record.setdefault("has_public_exploit", False)
                record.setdefault("num_exploits", 0)
                record.setdefault("social_source_count", 0)
                results[cve_id] = record

            chunk_start_idx += len(vulns)
            logger.info("  offset=%d  page=%d  chunk_total=%d  collected=%d",
                        chunk_start_idx, len(vulns), total, len(results))

            if not vulns or chunk_start_idx >= total:
                break
            if max_cves and len(results) >= max_cves:
                logger.info("  Reached max-cves limit (%d)", max_cves)
                break

            time.sleep(delay)

        if max_cves and len(results) >= max_cves:
            break

    logger.info("Collected %d NVD records in range [%s, %s]", len(results), after, before)
    return results


def fetch_nvd_by_ids(
    cve_ids: List[str],
    data_dir: Path,
    api_key: Optional[str] = None,
) -> Dict[str, dict]:
    """Fetch specific CVE IDs from NVD one at a time."""
    collector = DataCollector(output_dir=str(data_dir), nvd_api_key=api_key)
    results: Dict[str, dict] = {}
    delay = 0.6 if api_key else 6.0

    for i, cve_id in enumerate(cve_ids, 1):
        logger.info("[%d/%d] Fetching %s from NVD", i, len(cve_ids), cve_id)
        record = collector._fetch_single_cve(cve_id)
        if record:
            # _fetch_single_cve calls _parse_nvd_record which does NOT include cve_id
            record["cve_id"] = cve_id
            record.setdefault("binary_label", 0)
            record.setdefault("epss_score", 0.0)
            record.setdefault("epss_percentile", 0.0)
            record.setdefault("in_kev", False)
            record.setdefault("has_public_exploit", False)
            record.setdefault("num_exploits", 0)
            record.setdefault("social_source_count", 0)
            results[cve_id] = record
        else:
            logger.warning("  Could not fetch %s from NVD", cve_id)
        time.sleep(delay)

    return results


# ── Ground truth lookup ────────────────────────────────────────────────────────

def fetch_ground_truth(
    cve_ids: List[str],
    data_dir: Path,
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, float]]:
    """Fetch KEV catalog and current EPSS scores.

    Returns:
        kev      : CVE-ID → {dateAdded, vendor, product, ransomware}
        epss_now : CVE-ID → current EPSS probability
        pct_now  : CVE-ID → current EPSS percentile
    """
    collector = DataCollector(output_dir=str(data_dir), nvd_api_key=api_key)

    # CISA KEV
    kev = collector.fetch_kev()
    logger.info("Loaded CISA KEV: %d exploited CVEs", len(kev))

    # Current EPSS from FIRST API for our specific CVEs
    epss_now: Dict[str, float] = {}
    pct_now:  Dict[str, float] = {}
    import requests

    batch_size = 100
    session = requests.Session()
    for i in range(0, len(cve_ids), batch_size):
        batch = cve_ids[i:i + batch_size]
        try:
            resp = session.get(
                "https://api.first.org/data/v1/epss",
                params={"cve": ",".join(batch)},
                timeout=30,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                cid = item["cve"]
                epss_now[cid] = float(item.get("epss", 0.0))
                pct_now[cid]  = float(item.get("percentile", 0.0))
        except Exception as e:
            logger.warning("EPSS batch %d failed: %s", i // batch_size, e)
        time.sleep(0.4)

    logger.info("Fetched current EPSS for %d/%d CVEs", len(epss_now), len(cve_ids))
    return kev, epss_now, pct_now


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, config_path: Path, device: str):
    """Load trained model from checkpoint using saved experiment config."""
    with open(config_path) as f:
        cfg = json.load(f)

    args = cfg["args"]

    # Reconstruct edge_type_vocab if needed
    edge_type_vocab = None
    # Try loading from the original processed dir
    orig_data_dir = Path(args.get("data_dir", "data/epss_sec4ai"))
    vocab_path = orig_data_dir / "pyg_dataset" / "processed" / "edge_type_vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            edge_type_vocab = json.load(f)
        logger.info("Loaded edge_type_vocab: %d types", len(edge_type_vocab))

    model = build_model(
        in_channels=cfg["in_channels"],
        backbone=args["backbone"],
        hidden_channels=args.get("hidden", 128),
        num_layers=args.get("layers", 3),
        dropout=args.get("dropout", 0.3),
        num_heads=args.get("heads", 4),
        tabular_dim=cfg["tabular_dim"],
        num_edge_types=cfg["num_edge_types"],
        edge_type_vocab=edge_type_vocab,
    )

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint (epoch %d, val_PR-AUC=%.4f) — %d params",
        ckpt.get("epoch", "?"),
        ckpt.get("metrics", {}).get("pr_auc", float("nan")),
        cfg["model_params"],
    )
    return model, cfg


# ── Graph building for unseen CVEs ─────────────────────────────────────────────

def build_inference_dataset(
    labeled: Dict[str, dict],
    work_dir: Path,
    cfg: dict,
    label_mode: str = "soft",
) -> CVEGraphDataset:
    """Build a CVEGraphDataset for inference-only CVEs.

    Uses the same preprocessing pipeline (SecBERT embedding, TPG construction)
    as training so graph structure and node feature dimensions match exactly.

    The training-time CWE vocabulary (from tabular_vocab.json) is injected into
    the CVEGraphDataset so the 57-dim tabular features use identical feature indices.

    Args:
        labeled   : labeled_cves.json-format dict for unseen CVEs.
        work_dir  : scratch directory for graph cache (.pt files).
        cfg       : experiment_config.json dict (for feature dimensions).
        label_mode: 'soft' or 'binary'.

    Returns:
        CVEGraphDataset ready for DataLoader.
    """
    # Patch CVEGraphDataset.process to inject training CWE vocab
    # This ensures the 57-dim tabular space is identical to training.
    orig_data_dir = Path(cfg["args"].get("data_dir", "data/epss_sec4ai"))
    tab_vocab_path = orig_data_dir / "pyg_dataset" / "processed" / "tabular_vocab.json"
    training_cwe_vocab: dict = {}
    if tab_vocab_path.exists():
        with open(tab_vocab_path) as f:
            training_cwe_vocab = json.load(f).get("cwe_to_idx", {})
        logger.info("Loaded training CWE vocab: %d CWEs", len(training_cwe_vocab))
    else:
        logger.warning(
            "tabular_vocab.json not found at %s — tabular CWE features may differ "
            "from training. Run the training pipeline first to generate this file.",
            tab_vocab_path,
        )

    # Monkey-patch process() to inject training CWE vocab after fit()
    from epss import cve_dataset as _cds
    _orig_process = _cds.CVEGraphDataset.process

    def _patched_process(self_ds):
        # Call original process but with vocab injection (via monkeypatched fit)
        if training_cwe_vocab is not None and self_ds.include_tabular:
            from epss.tabular_features import TabularFeatureExtractor
            _orig_fit = TabularFeatureExtractor.fit

            def _fit_with_vocab(self_te, labeled_cves):
                _orig_fit(self_te, labeled_cves)
                # Override with training vocab so feature indices are identical
                self_te.cwe_to_idx = dict(training_cwe_vocab)
                logger.info(
                    "Injected training CWE vocab (%d entries) into inference extractor",
                    len(training_cwe_vocab),
                )
                return self_te

            TabularFeatureExtractor.fit = _fit_with_vocab
            try:
                _orig_process(self_ds)
            finally:
                TabularFeatureExtractor.fit = _orig_fit
        else:
            _orig_process(self_ds)

    _cds.CVEGraphDataset.process = _patched_process

    lc_path = work_dir / "labeled_cves_infer.json"
    with open(lc_path, "w") as f:
        json.dump(labeled, f, indent=2)
    logger.info("Wrote %d CVEs to %s", len(labeled), lc_path)

    args = cfg["args"]
    dataset = CVEGraphDataset(
        root=str(work_dir / "pyg_infer"),
        labeled_cves_path=str(lc_path),
        label_mode=label_mode,
        embedding_dim=args.get("embedding_dim", 768),
        use_hybrid=not args.get("no_hybrid", False),
        include_tabular=args.get("hybrid", True),
        include_epss_feature=not args.get("no_epss_feature", False),
    )

    # Restore original process
    _cds.CVEGraphDataset.process = _orig_process

    logger.info("Built inference dataset: %d graphs", len(dataset))
    return dataset


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: CVEGraphDataset,
    device: str,
    batch_size: int = 32,
) -> Tuple[List[str], List[float]]:
    """Run model forward pass on all graphs.

    Returns:
        cve_ids : list of CVE IDs (in dataset order)
        probs   : predicted exploitation probabilities
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    all_ids: List[str] = []
    all_probs: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

        # Recover CVE IDs from batch
        if hasattr(batch, "cve_id"):
            ids = batch.cve_id
            if isinstance(ids, (list, tuple)):
                all_ids.extend(ids)
            elif isinstance(ids, str):
                all_ids.append(ids)
            else:
                all_ids.extend([str(ids)] * len(probs))
        else:
            all_ids.extend(["unknown"] * len(probs))

        all_probs.extend(probs)

    logger.info("Inference complete: %d predictions", len(all_probs))
    return all_ids, all_probs


# ── Output ─────────────────────────────────────────────────────────────────────

def write_predictions(
    cve_ids: List[str],
    probs: List[float],
    labeled: Dict[str, dict],
    kev: Dict[str, dict],
    epss_now: Dict[str, float],
    pct_now:  Dict[str, float],
    output_dir: Path,
    threshold: float = 0.5,
) -> Path:
    """Write predictions_infer.csv and verification_summary.txt.

    Ground truth: is_in_kev (CISA KEV — most authoritative).
    """
    import csv

    rows = []
    for cve_id, prob in zip(cve_ids, probs):
        # Map inference ID back to base CVE ID (strip training suffix like -3)
        base_id = cve_id
        rec = labeled.get(cve_id) or {}
        tier = risk_tier(prob)
        pred_label = 1 if prob >= threshold else 0

        # Ground truth from CISA KEV
        kev_entry    = kev.get(base_id, {})
        is_in_kev    = bool(kev_entry)
        kev_added    = kev_entry.get("dateAdded", "")
        kev_vendor   = kev_entry.get("vendor", "")
        kev_product  = kev_entry.get("product", "")
        kev_ransomware = kev_entry.get("knownRansomwareCampaignUse", "")

        # Current EPSS from FIRST API
        curr_epss = epss_now.get(base_id, float("nan"))
        curr_pct  = pct_now.get(base_id,  float("nan"))

        # Prediction correctness
        correct_vs_kev  = int(pred_label == int(is_in_kev))
        epss_positive   = int(curr_epss >= 0.1) if not np.isnan(curr_epss) else -1
        correct_vs_epss = int(pred_label == epss_positive) if epss_positive >= 0 else -1

        rows.append({
            "cve_id":             base_id,
            "published":          rec.get("published", ""),
            "cvss3_score":        rec.get("cvss3_score", ""),
            "description":        rec.get("description", "")[:120].replace("\n", " "),
            "predicted_prob":     f"{prob:.6f}",
            "predicted_label":    pred_label,
            "risk_tier":          tier,
            # CISA KEV ground truth
            "is_in_kev":          int(is_in_kev),
            "kev_date_added":     kev_added,
            "kev_vendor":         kev_vendor,
            "kev_product":        kev_product,
            "kev_ransomware":     kev_ransomware,
            "correct_vs_kev":     correct_vs_kev,
            # FIRST EPSS ground truth
            "current_epss_score": f"{curr_epss:.6f}" if not np.isnan(curr_epss) else "",
            "current_epss_pct":   f"{curr_pct:.4f}"  if not np.isnan(curr_pct)  else "",
            "correct_vs_epss":    correct_vs_epss if correct_vs_epss >= 0 else "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions_infer.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d predictions → %s", len(rows), csv_path)

    # ── Verification summary ──────────────────────────────────────────────────
    n = len(rows)
    kev_positives = [r for r in rows if r["is_in_kev"] == 1]
    kev_negatives = [r for r in rows if r["is_in_kev"] == 0]
    pred_positives = [r for r in rows if r["predicted_label"] == 1]

    tp = sum(1 for r in rows if r["predicted_label"] == 1 and r["is_in_kev"] == 1)
    fp = sum(1 for r in rows if r["predicted_label"] == 1 and r["is_in_kev"] == 0)
    fn = sum(1 for r in rows if r["predicted_label"] == 0 and r["is_in_kev"] == 1)
    tn = sum(1 for r in rows if r["predicted_label"] == 0 and r["is_in_kev"] == 0)

    precision_kev = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_kev    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_kev        = (2 * precision_kev * recall_kev / (precision_kev + recall_kev)
                     if (precision_kev + recall_kev) > 0 else 0.0)

    # Tier distribution
    tier_counts = {}
    for tier_name in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
        tier_counts[tier_name] = sum(1 for r in rows if r["risk_tier"] == tier_name)

    summary_path = output_dir / "verification_summary.txt"
    lines = [
        "=" * 70,
        "EPSS-GNN Inference — Verification Summary",
        f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Total CVEs scored:      {n}",
        f"Predicted positive:     {len(pred_positives)} ({100*len(pred_positives)/max(n,1):.1f}%)",
        f"Actually in KEV:        {len(kev_positives)} ({100*len(kev_positives)/max(n,1):.1f}%)",
        "",
        "── Ground Truth: CISA KEV ─────────────────────────────────",
        f"  TP (predicted+, KEV+): {tp}",
        f"  FP (predicted+, KEV-): {fp}",
        f"  FN (predicted-, KEV+): {fn}",
        f"  TN (predicted-, KEV-): {tn}",
        f"  Precision vs KEV:      {precision_kev:.4f}",
        f"  Recall    vs KEV:      {recall_kev:.4f}",
        f"  F1        vs KEV:      {f1_kev:.4f}",
        "",
        "── Risk Tier Distribution ──────────────────────────────────",
    ]
    for tier_name, count in tier_counts.items():
        # How many of this tier are in KEV?
        tier_kev = sum(1 for r in rows if r["risk_tier"] == tier_name and r["is_in_kev"] == 1)
        pct = 100 * count / max(n, 1)
        lines.append(f"  {tier_name:<10}: {count:4d} ({pct:5.1f}%)   KEV: {tier_kev}")
    lines += [
        "",
        "── Top-20 Highest Risk (sorted by predicted probability) ───",
        f"  {'CVE-ID':<22} {'Prob':>6}  {'Tier':<10} {'KEV':>4}  {'KEV_date':<12}  {'Curr_EPSS':>10}",
        f"  {'-'*22} {'------':>6}  {'-'*10} {'----':>4}  {'-'*12}  {'-'*10}",
    ]
    top20 = sorted(rows, key=lambda r: float(r["predicted_prob"]), reverse=True)[:20]
    for r in top20:
        kev_flag = "YES" if r["is_in_kev"] == 1 else " no"
        epss_str = r.get("current_epss_score", "n/a") or "n/a"
        lines.append(
            f"  {r['cve_id']:<22} {float(r['predicted_prob']):>6.4f}  "
            f"{r['risk_tier']:<10} {kev_flag:>4}  {r['kev_date_added']:<12}  {epss_str:>10}"
        )

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)
    logger.info("Verification summary → %s", summary_path)
    return csv_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["post-dataset", "pre-dataset", "custom"],
        default="post-dataset",
        help=(
            "post-dataset : CVEs after training set end (2025-06-01) — default; "
            "pre-dataset  : CVEs before training set start (2021-11-01); "
            "custom       : explicit --cve-ids or --cve-file"
        ),
    )

    # Date range (used for post-dataset / pre-dataset modes)
    parser.add_argument("--after-date",  default=None,
                        help="Fetch CVEs published on or after this date (YYYY-MM-DD)")
    parser.add_argument("--before-date", default=None,
                        help="Fetch CVEs published on or before this date (YYYY-MM-DD)")

    # Custom CVE list
    parser.add_argument("--cve-ids", default=None,
                        help="Comma-separated CVE IDs, e.g. CVE-2025-1234,CVE-2025-5678")
    parser.add_argument("--cve-file", default=None,
                        help="Text file with one CVE ID per line")

    # Model
    parser.add_argument("--checkpoint",
                        default="output/epss_sec4ai/best_model.pt",
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--config",
                        default="output/epss_sec4ai/experiment_config.json",
                        help="Path to experiment_config.json (for model architecture)")

    # Options
    parser.add_argument("--max-cves", type=int, default=500,
                        help="Max CVEs to fetch from NVD (API rate-limit friendly)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for predicted_label (default 0.5)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write predictions_infer.csv (auto if not set)")
    parser.add_argument("--work-dir", default=None,
                        help="Scratch dir for graph cache (default: system temp)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nvd-api-key", default=None,
                        help="NVD API key (10x higher rate limit; optional)")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Do not delete the graph cache scratch dir after inference")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip NVD fetch; use --labeled-json directly")
    parser.add_argument("--labeled-json", default=None,
                        help="Pre-built labeled_cves.json to use (skips NVD fetch)")
    parser.add_argument("--no-epss-prefetch", action="store_true",
                        help=(
                            "Skip pre-fetching current EPSS scores from FIRST API. "
                            "WARNING: the model was trained with epss_score as a tabular "
                            "feature; without real EPSS values all predictions collapse to "
                            "near-zero (pure graph/CVSS mode). Only use this to study the "
                            "graph-only signal contribution."
                        ))

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    cfg_path  = Path(args.config)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    # Default output dir
    if args.output_dir is None:
        args.output_dir = f"output/infer/{args.mode}"
    output_dir = Path(args.output_dir)

    # Work dir for graph scratch
    _tmp = None
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        _tmp = tempfile.mkdtemp(prefix="epss_infer_")
        work_dir = Path(_tmp)
    logger.info("Graph cache dir: %s", work_dir)

    data_dir = work_dir / "data"
    data_dir.mkdir(exist_ok=True)

    try:
        # ── Step 1: Build labeled dict for unseen CVEs ─────────────────────────
        if args.labeled_json:
            logger.info("Loading pre-built labeled_cves.json: %s", args.labeled_json)
            with open(args.labeled_json) as f:
                labeled = json.load(f)
        elif args.mode == "custom":
            cve_ids: List[str] = []
            if args.cve_ids:
                cve_ids = [c.strip() for c in args.cve_ids.split(",") if c.strip()]
            elif args.cve_file:
                cve_ids = [l.strip() for l in open(args.cve_file) if l.strip()]
            else:
                logger.error("--mode custom requires --cve-ids or --cve-file")
                sys.exit(1)
            logger.info("Custom mode: %d CVE IDs", len(cve_ids))
            labeled = fetch_nvd_by_ids(cve_ids, data_dir, api_key=args.nvd_api_key)
        else:
            # post-dataset or pre-dataset: default date windows
            if args.mode == "post-dataset":
                after  = args.after_date  or "2025-07-01"
                before = args.before_date or "2025-09-30"
            else:  # pre-dataset
                after  = args.after_date  or "2019-01-01"
                before = args.before_date or "2021-10-31"

            logger.info("Mode: %s  |  window: %s → %s", args.mode, after, before)
            labeled = fetch_nvd_by_date(
                after=after,
                before=before,
                data_dir=data_dir,
                max_cves=args.max_cves,
                api_key=args.nvd_api_key,
            )

        if not labeled:
            logger.error("No CVE records found — check date range or CVE IDs")
            sys.exit(1)

        # Limit to max_cves for custom/pre-dataset modes too
        if args.max_cves and len(labeled) > args.max_cves:
            keys = list(labeled.keys())[:args.max_cves]
            labeled = {k: labeled[k] for k in keys}

        logger.info("Scoring %d unseen CVEs", len(labeled))
        prefetch_epss: dict = {}   # populated in Step 1b if --no-epss-prefetch not set
        prefetch_pct:  dict = {}

        # ── Step 1b: Pre-fetch EPSS scores and inject into labeled dict ────────
        # CRITICAL: The model was trained with `epss_score` as a tabular input
        # feature (alongside CVSS, CWE, age, etc.).  For NVD-fetched CVEs we have
        # no pre-existing EPSS score, so we must pull current scores from the
        # FIRST API BEFORE building graphs.  Without this, every CVE gets
        # epss_score=0.0, which the model associates with "low risk" and returns
        # near-zero predictions for all CVEs — regardless of true risk.
        #
        # Note: if you want a pure graph-only prediction (no EPSS signal), pass
        # --no-epss-prefetch and the model uses only CVE text + CVSS + CWE.
        if not getattr(args, "no_epss_prefetch", False):
            logger.info("Pre-fetching current EPSS scores to populate tabular features...")
            import requests as _req
            _session = _req.Session()
            cve_id_list = list(labeled.keys())
            prefetch_epss: dict = {}
            prefetch_pct:  dict = {}
            for _i in range(0, len(cve_id_list), 100):
                _batch = cve_id_list[_i:_i + 100]
                try:
                    _r = _session.get(
                        "https://api.first.org/data/v1/epss",
                        params={"cve": ",".join(_batch)},
                        timeout=30,
                    )
                    _r.raise_for_status()
                    for _item in _r.json().get("data", []):
                        prefetch_epss[_item["cve"]] = float(_item.get("epss", 0.0))
                        prefetch_pct[_item["cve"]]  = float(_item.get("percentile", 0.0))
                except Exception as _e:
                    logger.warning("EPSS pre-fetch batch %d failed: %s", _i // 100, _e)
                time.sleep(0.4)

            n_filled = 0
            for cve_id in labeled:
                if cve_id in prefetch_epss:
                    labeled[cve_id]["epss_score"]      = prefetch_epss[cve_id]
                    labeled[cve_id]["epss_percentile"] = prefetch_pct.get(cve_id, 0.0)
                    n_filled += 1
            logger.info(
                "EPSS pre-fill: %d / %d CVEs got real EPSS scores "
                "(remainder stay at 0.0 — may not yet be in EPSS database)",
                n_filled, len(labeled),
            )
            # Log distribution for sanity check
            _epss_vals = [labeled[c]["epss_score"] for c in labeled]
            _pos_epss = sum(1 for v in _epss_vals if v >= 0.1)
            logger.info(
                "EPSS distribution: min=%.4f  max=%.4f  mean=%.4f  n_pos(>=0.1)=%d",
                min(_epss_vals), max(_epss_vals), sum(_epss_vals)/max(len(_epss_vals),1), _pos_epss,
            )
        else:
            logger.warning(
                "--no-epss-prefetch active: all epss_score tabular features = 0.0. "
                "Predictions will reflect graph-structure + CVSS only, NOT EPSS signal. "
                "Expect lower predicted probabilities across the board."
            )

        # ── Step 2: Load model ─────────────────────────────────────────────────
        model, cfg = load_model(ckpt_path, cfg_path, device)

        # ── Step 3: Build graphs (same pipeline as training) ───────────────────
        logger.info("Building TPG graphs (this may take a few minutes)...")
        dataset = build_inference_dataset(labeled, work_dir, cfg, label_mode="soft")

        if len(dataset) == 0:
            logger.error("Graph construction produced 0 graphs. Check descriptions.")
            sys.exit(1)

        # Sanity: check feature dims match
        sample = dataset[0]
        if sample.x.shape[1] != cfg["in_channels"]:
            logger.error(
                "Feature dimension mismatch! Graph has %d, model expects %d. "
                "Make sure inference uses the same pipeline as training.",
                sample.x.shape[1], cfg["in_channels"]
            )
            sys.exit(1)
        logger.info(
            "Graph dims OK: in_channels=%d  tabular=%s",
            sample.x.shape[1],
            sample.tabular.shape if hasattr(sample, "tabular") and sample.tabular is not None else "None",
        )

        # ── Step 4: Run inference ──────────────────────────────────────────────
        cve_ids_out, probs = run_inference(model, dataset, device, args.batch_size)

        # Map back to base CVE IDs (dataset may produce same IDs without suffix)
        logger.info("Got %d predictions for %d CVEs", len(probs), len(labeled))

        # ── Step 5: Fetch ground truth ─────────────────────────────────────────
        # Reuse the pre-fetched EPSS scores (already in prefetch_epss/prefetch_pct)
        # Only need to fetch KEV now; EPSS was already collected in Step 1b.
        logger.info("Fetching CISA KEV ground truth...")
        all_ids = list(set(cve_ids_out))
        kev, epss_now, pct_now = fetch_ground_truth(all_ids, data_dir, args.nvd_api_key)
        # Merge pre-fetched EPSS (avoids a second round-trip to FIRST API)
        if not getattr(args, "no_epss_prefetch", False):
            for _cid in all_ids:
                if _cid not in epss_now and _cid in prefetch_epss:
                    epss_now[_cid] = prefetch_epss[_cid]
                    pct_now[_cid]  = prefetch_pct.get(_cid, 0.0)

        # ── Step 6: Write output ───────────────────────────────────────────────
        write_predictions(
            cve_ids=cve_ids_out,
            probs=probs,
            labeled=labeled,
            kev=kev,
            epss_now=epss_now,
            pct_now=pct_now,
            output_dir=output_dir,
            threshold=args.threshold,
        )

    finally:
        if _tmp and not args.keep_work_dir:
            shutil.rmtree(_tmp, ignore_errors=True)
            logger.info("Cleaned up temp dir: %s", _tmp)
        elif args.keep_work_dir:
            logger.info("Graph cache kept at: %s", work_dir)


if __name__ == "__main__":
    main()
