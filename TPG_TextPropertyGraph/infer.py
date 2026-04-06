#!/usr/bin/env python3
"""
EPSS-GNN Inference Script
==========================
Predict exploitation probability for new or existing CVEs.

Mirrors EPSS operational usage:
  - Train model on CVEs published before a cutoff date
  - Predict exploitation probability for CVEs published after the cutoff
  - Evaluate retrospectively against CISA KEV (known ground truth)

Modes:
  1. Specific CVE IDs:       python infer.py --cve-ids CVE-2024-1234 CVE-2024-5678
  2. Text file of IDs:       python infer.py --cve-file ids.txt
  3. Recent N days:          python infer.py --recent-days 30
  4. Date range:             python infer.py --date-range 2024-01-01 2024-01-31
  5. Temporal evaluation:    python infer.py --temporal-eval --train-cutoff 2024-01-01 --eval-days 30

Output (CSV, sorted by prob descending):
  cve_id, prob, tier, binary_pred, cvss_score, published, in_kev, description

Tiers:
  CRITICAL  prob >= 0.70
  HIGH      0.40 <= prob < 0.70
  MEDIUM    0.10 <= prob < 0.40
  LOW       prob < 0.10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch

# Add project root to path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infer")

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CKPT_DIR = "output/epss_full_5pct_multiview_hybrid"
DEFAULT_LABELED_FILE = "data/epss_full/labeled_cves_5pct.json"
DEFAULT_KEV_FILE = "data/epss_full/cisa_kev.json"
DEFAULT_THRESHOLD = 0.448   # optimal F1 threshold from 5% stratified run
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

TIERS = [
    ("CRITICAL", 0.70),
    ("HIGH",     0.40),
    ("MEDIUM",   0.10),
    ("LOW",      0.00),
]


def prob_to_tier(p: float) -> str:
    for name, cutoff in TIERS:
        if p >= cutoff:
            return name
    return "LOW"


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: str, device: torch.device):
    """Load model, config, and tabular extractor from a checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best_model.pt"
    config_path = ckpt_dir / "experiment_config.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")

    config = json.loads(config_path.read_text())
    args = config["args"]

    # Detect tabular dim from checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    tabular_dim = 0
    for key, tensor in state.items():
        if "tabular_encoder.0.weight" in key:
            tabular_dim = tensor.shape[1]
            break

    logger.info("Checkpoint: epoch=%d, tabular_dim=%d", ckpt.get("epoch", "?"), tabular_dim)

    # Load edge type vocab for multiview
    pyg_dir = Path(args.get("data_dir", "data/epss")) / "pyg_dataset" / "processed"
    edge_vocab_path = pyg_dir / "edge_type_vocab.json"
    edge_type_vocab = {}
    if edge_vocab_path.exists():
        edge_type_vocab = json.loads(edge_vocab_path.read_text())

    # Build model
    from epss.gnn_model import build_model
    model = build_model(
        in_channels=config.get("in_channels", 781),
        backbone=args.get("backbone", "multiview"),
        hidden_channels=args.get("hidden", 128),
        num_layers=args.get("layers", 3),
        num_heads=args.get("heads", 4),
        dropout=args.get("dropout", 0.3),
        tabular_dim=tabular_dim,
        num_edge_types=config.get("num_edge_types", 13),
        edge_type_vocab=edge_type_vocab,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info("Model: %s | params=%s | device=%s",
                config.get("model_type", "?"),
                f"{sum(p.numel() for p in model.parameters()):,}",
                device)

    return model, config, tabular_dim


def load_tabular_extractor(ckpt_dir: str, labeled_file: str, top_k_cwes: int = 25):
    """Load or reconstruct the tabular feature extractor.

    Tries to load saved CWE vocab from {ckpt_dir}/cwe_vocab.json.
    Falls back to re-fitting from labeled_file.
    """
    from epss.tabular_features import TabularFeatureExtractor

    ckpt_dir = Path(ckpt_dir)
    cwe_vocab_path = ckpt_dir / "cwe_vocab.json"

    tab = TabularFeatureExtractor(top_k_cwes=top_k_cwes)

    if cwe_vocab_path.exists():
        cwe_to_idx = json.loads(cwe_vocab_path.read_text())
        tab.cwe_to_idx = cwe_to_idx
        tab._fitted = True
        logger.info("Loaded CWE vocab from %s (%d CWEs)", cwe_vocab_path, len(cwe_to_idx))
    else:
        # Re-fit from labeled file
        labeled_path = Path(labeled_file)
        if not labeled_path.exists():
            raise FileNotFoundError(
                f"Cannot fit tabular extractor: {labeled_path} not found.\n"
                f"Either provide {cwe_vocab_path} or a valid --labeled-file."
            )
        logger.info("Fitting tabular extractor from %s ...", labeled_path)
        labeled = json.loads(labeled_path.read_text())
        tab.fit(labeled)
        # Save for next time
        with open(cwe_vocab_path, "w") as f:
            json.dump(tab.cwe_to_idx, f, indent=2)
        logger.info("Saved CWE vocab to %s", cwe_vocab_path)

    return tab


# ─── NVD Fetching ─────────────────────────────────────────────────────────────

def fetch_single_cve(cve_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Fetch one CVE from NVD API 2.0."""
    import requests
    headers = {"apiKey": api_key} if api_key else {}
    try:
        r = requests.get(NVD_API, params={"cveId": cve_id}, headers=headers, timeout=30)
        r.raise_for_status()
        vulns = r.json().get("vulnerabilities", [])
        if vulns:
            return _parse_nvd_record(vulns[0].get("cve", {}))
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", cve_id, e)
    return None


def fetch_date_range(
    start: str, end: str, api_key: Optional[str] = None
) -> Dict[str, dict]:
    """Fetch all CVEs published in [start, end] from NVD API (paginated)."""
    import requests
    headers = {"apiKey": api_key} if api_key else {}
    delay = 0.6 if api_key else 6.0
    results = {}
    start_idx = 0
    per_page = 2000

    logger.info("Fetching NVD CVEs from %s to %s ...", start, end)

    while True:
        params = {
            "pubStartDate": f"{start}T00:00:00",
            "pubEndDate": f"{end}T23:59:59",
            "startIndex": start_idx,
            "resultsPerPage": per_page,
        }
        try:
            r = requests.get(NVD_API, params=params, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error("NVD API error at offset %d: %s", start_idx, e)
            break

        for item in data.get("vulnerabilities", []):
            cve_raw = item.get("cve", {})
            cid = cve_raw.get("id", "")
            if cid:
                results[cid] = _parse_nvd_record(cve_raw)

        total = data.get("totalResults", 0)
        start_idx += per_page
        logger.info("  Fetched %d / %d CVEs", min(start_idx, total), total)
        if start_idx >= total:
            break
        time.sleep(delay)

    logger.info("Fetched %d CVEs from NVD", len(results))
    return results


def _parse_nvd_record(cve: dict) -> dict:
    """Parse NVD JSON into a flat record compatible with tabular extractor."""
    description = ""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            description = d.get("value", "")
            break

    cvss3_score, cvss3_vector = None, ""
    for key in ["cvssMetricV31", "cvssMetricV30"]:
        ml = cve.get("metrics", {}).get(key, [])
        if ml:
            cd = ml[0].get("cvssData", {})
            cvss3_score = cd.get("baseScore")
            cvss3_vector = cd.get("vectorString", "")
            break

    cwe_ids = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            v = d.get("value", "")
            if v.startswith("CWE-"):
                cwe_ids.append(v)

    refs = [r.get("url", "") for r in cve.get("references", [])[:10]]

    return {
        "description": description,
        "published": cve.get("published", ""),
        "lastModified": cve.get("lastModified", ""),
        "cvss3_score": cvss3_score,
        "cvss3_vector": cvss3_vector,
        "cwe_ids": cwe_ids,
        "references": refs,
        # Inference-time: no labels, no EPSS scores initially
        "binary_label": -1,
        "epss_score": 0.0,
        "epss_percentile": 0.0,
        "has_public_exploit": False,
        "num_exploits": 0,
    }


def enrich_with_epss(records: Dict[str, dict], api_key: Optional[str] = None):
    """Fetch current EPSS scores for CVEs and add to records in-place."""
    import requests
    cve_ids = list(records.keys())
    logger.info("Fetching EPSS scores for %d CVEs ...", len(cve_ids))

    # EPSS API accepts up to 100 CVEs per request
    for i in range(0, len(cve_ids), 100):
        batch = cve_ids[i:i+100]
        params = [("cve", cid) for cid in batch]
        try:
            r = requests.get(
                "https://api.first.org/data/v1/epss",
                params=params, timeout=30
            )
            r.raise_for_status()
            for item in r.json().get("data", []):
                cid = item.get("cve")
                if cid in records:
                    records[cid]["epss_score"] = float(item.get("epss", 0.0))
                    records[cid]["epss_percentile"] = float(item.get("percentile", 0.0))
        except Exception as e:
            logger.warning("EPSS API batch %d failed: %s", i // 100, e)
        time.sleep(0.5)


# ─── Graph Building ───────────────────────────────────────────────────────────

def build_pipeline():
    """Initialize the TPG pipeline (HybridSecurityPipeline with SecBERT)."""
    try:
        from tpg.pipeline import HybridSecurityPipeline
        logger.info("Loading HybridSecurityPipeline (SecBERT + NLP tools) ...")
        return HybridSecurityPipeline()
    except Exception as e:
        logger.error("Failed to load HybridSecurityPipeline: %s", e)
        raise


def cve_to_pyg(
    pipeline,
    tab_extractor,
    cve_id: str,
    record: dict,
    embedding_dim: int = 768,
) -> Optional[Data]:
    """Convert a single CVE record to a PyG Data object."""
    description = record.get("description", "")
    if not description or len(description.strip()) < 10:
        logger.warning("Skipping %s: empty description", cve_id)
        return None

    try:
        graph = pipeline.run(description, doc_id=cve_id)
        if graph.num_nodes < 3:
            logger.warning("Skipping %s: trivially small graph (%d nodes)", cve_id, graph.num_nodes)
            return None

        pyg_dict = pipeline.export_pyg(graph, embedding_dim=embedding_dim)

        x = torch.tensor(pyg_dict["x"], dtype=torch.float)
        edge_index = torch.tensor(pyg_dict["edge_index"], dtype=torch.long)
        edge_type = torch.tensor(pyg_dict["edge_type"], dtype=torch.long)
        edge_attr = torch.tensor(pyg_dict["edge_attr"], dtype=torch.float)

        # Unknown label at inference time
        label = record.get("binary_label", -1)
        y = torch.tensor([max(label, 0)], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            y=y,
            num_nodes=pyg_dict["num_nodes"],
        )

        # Tabular features
        if tab_extractor is not None:
            tab_vec = tab_extractor.encode(record)
            data.tabular = torch.tensor(tab_vec, dtype=torch.float).unsqueeze(0)

        data.cve_id = cve_id
        return data

    except Exception as e:
        logger.warning("Failed to build graph for %s: %s", cve_id, e)
        return None


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    graphs: List[Data],
    device: torch.device,
    target_tabular_dim: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Run batched inference on a list of graphs. Returns prob array."""
    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs = []

    for batch in loader:
        batch = batch.to(device)

        # Slice tabular to match checkpoint dim (handles dim mismatch)
        if hasattr(batch, "tabular") and batch.tabular is not None and target_tabular_dim > 0:
            batch.tabular = batch.tabular[:, :target_tabular_dim]

        out = model(batch)

        if out.shape[-1] == 2:
            prob = torch.softmax(out, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(out).squeeze(-1)

        all_probs.extend(prob.cpu().numpy().tolist())

    return np.array(all_probs)


# ─── Output ───────────────────────────────────────────────────────────────────

def build_results(
    cve_ids: List[str],
    records: Dict[str, dict],
    probs: np.ndarray,
    kev_set: set,
    threshold: float,
) -> List[dict]:
    """Assemble results list sorted by probability descending."""
    rows = []
    for cve_id, prob in zip(cve_ids, probs):
        rec = records.get(cve_id, {})
        rows.append({
            "cve_id": cve_id,
            "prob": float(prob),
            "tier": prob_to_tier(prob),
            "predicted_exploited": int(prob >= threshold),
            "cvss_score": rec.get("cvss3_score") or "",
            "published": rec.get("published", "")[:10],
            "in_kev": int(cve_id in kev_set),
            "epss_score": rec.get("epss_score", ""),
            "description": (rec.get("description", "")[:120] + "...") if rec.get("description") else "",
        })
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return rows


def save_csv(rows: List[dict], path: str):
    import csv
    fieldnames = ["cve_id", "prob", "tier", "predicted_exploited", "cvss_score",
                  "published", "in_kev", "epss_score", "description"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %d predictions → %s", len(rows), path)


def print_summary(rows: List[dict], threshold: float, kev_set: set):
    """Print a ranked summary table to stdout."""
    n = len(rows)
    flagged = [r for r in rows if r["prob"] >= threshold]
    kev_flagged = [r for r in flagged if r["in_kev"]]

    print(f"\n{'='*72}")
    print(f"  EPSS-GNN Exploitation Probability Predictions  (threshold={threshold:.3f})")
    print(f"{'='*72}")
    print(f"  CVEs scored: {n}   |   Flagged (≥threshold): {len(flagged)}   |   KEV hits: {len(kev_flagged)}")
    if flagged:
        precision = len(kev_flagged) / len(flagged)
        print(f"  Precision (of flagged): {precision:.3f}")
    print(f"{'='*72}")
    print(f"  {'CVE-ID':<20} {'Prob':>6}  {'Tier':<10} {'CVSS':>5}  {'Published':<12} {'KEV':>4}  {'EPSS':>6}")
    print(f"  {'-'*70}")

    for r in rows[:50]:  # top 50
        kev_mark = "✓" if r["in_kev"] else "✗"
        epss = f"{r['epss_score']:.3f}" if r["epss_score"] != "" else "  N/A"
        cvss = f"{r['cvss_score']:.1f}" if r["cvss_score"] != "" else " N/A"
        print(f"  {r['cve_id']:<20} {r['prob']:>6.4f}  {r['tier']:<10} {cvss:>5}  {r['published']:<12} {kev_mark:>4}  {epss:>6}")

    if n > 50:
        print(f"  ... ({n - 50} more rows in output file)")
    print(f"{'='*72}\n")


def compute_eval_metrics(rows: List[dict], threshold: float):
    """Compute PR-AUC, F1, etc. when ground-truth KEV labels are available."""
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

    y_true = np.array([r["in_kev"] for r in rows])
    y_prob = np.array([r["prob"] for r in rows])

    if y_true.sum() == 0:
        logger.warning("No positive (KEV) samples in this set — cannot compute metrics")
        return

    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    n_pos = int(y_true.sum())
    n_total = len(y_true)

    print(f"\n{'='*50}")
    print(f"  Temporal Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  Dataset:   {n_total} CVEs ({n_pos} KEV positives, {100*n_pos/n_total:.2f}%)")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"{'='*50}\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="EPSS-GNN: Predict CVE exploitation probability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input modes (mutually exclusive)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--cve-ids", nargs="+", metavar="CVE-ID",
                     help="One or more CVE IDs to score")
    inp.add_argument("--cve-file", metavar="PATH",
                     help="Text file with one CVE-ID per line")
    inp.add_argument("--recent-days", type=int, metavar="N",
                     help="Score all CVEs published in the last N days")
    inp.add_argument("--date-range", nargs=2, metavar=("START", "END"),
                     help="Score CVEs published between START and END (YYYY-MM-DD)")
    inp.add_argument("--temporal-eval", action="store_true",
                     help="Train-cutoff temporal evaluation mode")

    # Temporal eval options
    p.add_argument("--train-cutoff", metavar="YYYY-MM-DD",
                   help="Date used as train/test split (with --temporal-eval)")
    p.add_argument("--eval-days", type=int, default=30,
                   help="Number of days after cutoff to evaluate (default: 30)")

    # Model options
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                   help=f"Checkpoint directory (default: {DEFAULT_CKPT_DIR})")
    p.add_argument("--labeled-file", default=DEFAULT_LABELED_FILE,
                   help="Training labeled_cves.json (for CWE vocab fallback)")
    p.add_argument("--kev-file", default=DEFAULT_KEV_FILE,
                   help="CISA KEV JSON file for ground-truth labels")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Decision threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Inference batch size (default: 16)")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: auto-detect)")
    p.add_argument("--nvd-api-key", default=None,
                   help="NVD API key (10× faster rate limit)")

    # Output
    p.add_argument("--output", default="predictions.csv",
                   help="Output CSV path (default: predictions.csv)")
    p.add_argument("--no-epss", action="store_true",
                   help="Skip EPSS score enrichment from FIRST API")
    p.add_argument("--eval", action="store_true",
                   help="Compute PR-AUC / F1 metrics (requires KEV labels)")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    model, config, tabular_dim = load_checkpoint(args.ckpt_dir, device)

    # Load tabular extractor
    tab_extractor = None
    if tabular_dim > 0:
        tab_extractor = load_tabular_extractor(
            args.ckpt_dir, args.labeled_file,
            top_k_cwes=config["args"].get("top_k_cwes", 25),
        )

    # Load KEV ground truth
    kev_set = set()
    kev_path = Path(args.kev_file)
    if kev_path.exists():
        kev_raw = json.loads(kev_path.read_text())
        # KEV JSON has a list under "vulnerabilities" with "cveID" fields
        if isinstance(kev_raw, dict) and "vulnerabilities" in kev_raw:
            kev_set = {v["cveID"] for v in kev_raw["vulnerabilities"]}
        elif isinstance(kev_raw, dict):
            kev_set = set(kev_raw.keys())
        logger.info("Loaded %d KEV entries from %s", len(kev_set), kev_path)
    else:
        logger.warning("KEV file not found at %s — in_kev column will be empty", kev_path)

    # ── Collect CVE IDs to score ──────────────────────────────────────────────

    records: Dict[str, dict] = {}

    if args.cve_ids:
        logger.info("Fetching %d CVEs from NVD ...", len(args.cve_ids))
        for cve_id in args.cve_ids:
            rec = fetch_single_cve(cve_id, args.nvd_api_key)
            if rec:
                records[cve_id] = rec
            else:
                logger.warning("Could not fetch %s", cve_id)
            time.sleep(0.6 if args.nvd_api_key else 6.0)

    elif args.cve_file:
        ids = [l.strip() for l in Path(args.cve_file).read_text().splitlines() if l.strip()]
        logger.info("Fetching %d CVEs from NVD ...", len(ids))
        for cve_id in ids:
            rec = fetch_single_cve(cve_id, args.nvd_api_key)
            if rec:
                records[cve_id] = rec
            time.sleep(0.6 if args.nvd_api_key else 6.0)

    elif args.recent_days:
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=args.recent_days)).strftime("%Y-%m-%d")
        records = fetch_date_range(start, end, args.nvd_api_key)

    elif args.date_range:
        start, end = args.date_range
        records = fetch_date_range(start, end, args.nvd_api_key)

    elif args.temporal_eval:
        if not args.train_cutoff:
            logger.error("--temporal-eval requires --train-cutoff YYYY-MM-DD")
            sys.exit(1)
        cutoff = datetime.fromisoformat(args.train_cutoff)
        eval_end = cutoff + timedelta(days=args.eval_days)
        start = cutoff.strftime("%Y-%m-%d")
        end = eval_end.strftime("%Y-%m-%d")
        logger.info("Temporal eval: CVEs published %s → %s", start, end)
        records = fetch_date_range(start, end, args.nvd_api_key)
        args.eval = True   # always compute metrics in temporal-eval mode

    if not records:
        logger.error("No CVE records to score. Exiting.")
        sys.exit(1)

    logger.info("Scoring %d CVEs ...", len(records))

    # ── Enrich with EPSS scores ───────────────────────────────────────────────

    if not args.no_epss and tab_extractor is not None:
        enrich_with_epss(records, args.nvd_api_key)

    # ── Build TPG graphs ──────────────────────────────────────────────────────

    pipeline = build_pipeline()

    graphs = []
    scored_ids = []
    logger.info("Building TPG graphs ...")
    for cve_id, record in records.items():
        data = cve_to_pyg(pipeline, tab_extractor, cve_id, record)
        if data is not None:
            graphs.append(data)
            scored_ids.append(cve_id)

    logger.info("Built %d graphs from %d CVEs (%d skipped)",
                len(graphs), len(records), len(records) - len(graphs))

    if not graphs:
        logger.error("No valid graphs built. Check CVE descriptions.")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────

    logger.info("Running inference ...")
    probs = run_inference(model, graphs, device, tabular_dim, args.batch_size)

    # ── Assemble and output results ───────────────────────────────────────────

    rows = build_results(scored_ids, records, probs, kev_set, args.threshold)
    print_summary(rows, args.threshold, kev_set)
    save_csv(rows, args.output)

    if args.eval or args.temporal_eval:
        compute_eval_metrics(rows, args.threshold)

    # Tier breakdown
    tier_counts = {}
    for r in rows:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    logger.info("Tier breakdown: %s", tier_counts)


if __name__ == "__main__":
    main()
