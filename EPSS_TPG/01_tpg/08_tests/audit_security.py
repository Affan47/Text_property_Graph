"""Read-only security-TPG diagnostics; writes only derived JSON/TXT reports.

No training, model downloads, source edits or external API calls. Rule-only
corpus coverage is not an accuracy measurement or a hybrid-model benchmark.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import torch
from tpg.paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from tpg.pipeline import SecurityPipeline
from tpg.schema.types import NodeType, SecurityEdgeType, EdgeType
from tpg.frontends.security_frontend import _CVE_PATTERN
from epss.cve_dataset import CVEGraphDataset
from epss.tabular_features import TabularFeatureExtractor
from epss.csv_adapter import _cvss_vector


CASES = {
    "control_affects": "CVE-2024-1111 affects Apache 2.4.1.",
    "control_negated": "CVE-2024-1111 does not affect Apache.",
    "negative_subject": "No CVE-2024-1111 affects Apache.",
    "passive_exploited": "CVE-2024-1111 is exploited by a remote attacker.",
    "passive_fixed": "CVE-2024-1111 is fixed by a security patch.",
    "coordinated_products": "CVE-2024-1111 affects Apache and nginx.",
    "capability_vs_observed": "CVE-2024-1111 allows a remote attacker to cause denial of service.",
    "negated_version": "Apache is not version 2.4.1.",
    "fixed_version": "Apache should be upgraded to 2.4.1.",
    "confidence_not_severity": "CVE-2024-1111 has low confidence.",
    "unknown_product": "CVE-2024-1111 affects NovelWidget 3.2.1.",
    "unicode_offsets": "\u0130 CVE-2024-1111 affects Apache and allows remote code execution.",
}


def describe(graph):
    return {
        "entities": [{"text": n.properties.text, "type": n.properties.entity_type,
                      "span": [n.properties.char_start, n.properties.char_end]}
                     for n in graph.nodes(NodeType.ENTITY) if n.properties.source == "security_frontend"],
        "security_edges": [{"source": graph.get_node(e.source).properties.text,
            "relation": e.edge_type.name, "target": graph.get_node(e.target).properties.text,
            "evidence": e.properties.extra}
            for e in graph.edges() if isinstance(e.edge_type, SecurityEdgeType)],
    }


def probes(pipeline):
    result = {key: dict(text=text, **describe(pipeline.run(text))) for key, text in CASES.items()}
    graph = pipeline.run(CASES["control_affects"])
    before = pipeline.export_pyg(graph, embedding_dim=0, use_security_edge_types=True)
    software = next(n for n in graph.nodes(NodeType.ENTITY) if n.properties.entity_type == "SOFTWARE")
    software.properties.entity_type, software.properties.domain_type = "IMPACT", "rce"
    after = pipeline.export_pyg(graph, embedding_dim=0, use_security_edge_types=True)
    keys = ("x", "edge_index", "edge_type", "edge_features", "edge_attr", "node_pos")
    result["security_category_not_in_tensors"] = {
        "all_model_input_tensors_unchanged": all(before[k] == after[k] for k in keys),
        "method": "Change only one existing node's category after graph construction; do not rerun extraction",
    }
    default_export = pipeline.export_pyg(graph)
    result["export_flag_mismatch"] = {"default_vocab_size": default_export["num_edge_types"],
        "dropped_security_edges": default_export["n_dropped_security_edges"],
        "explicit_vocab_size": before["num_edge_types"]}
    labels = {}
    for mode in ("soft", "binary"):
        adapter = object.__new__(CVEGraphDataset)
        adapter.label_mode = mode
        labels[mode] = adapter._label_from_record({"in_vckev": True})
    result["missing_targets"] = {"input": {"in_vckev": True}, "returned_labels": labels}
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.]))
    target = torch.tensor([.1])
    result["soft_loss_distortion"] = {"target": .1, "positive_weight": 9,
        "analytic_optimal_prediction": .5,
        "loss_at_correct_score": criterion(torch.logit(torch.tensor([.1])), target).item(),
        "loss_at_distorted_score": criterion(torch.logit(torch.tensor([.5])), target).item(),
        "formula": "p* = w*y / (1-y+w*y); no model trained"}
    tab = TabularFeatureExtractor().fit({})
    result["tabular_boundaries"] = {
        "missing_components_generate": _cvss_vector(pd.Series(dtype=object)),
        "feature_names_length_empty_cwe_vocab": len(tab.get_feature_names()),
        "feature_vector_length": tab.feature_dim,
        "age_days": {date: tab._compute_age_days(date) for date in
                     ("2024-01-01", "2025-06-01", "2026-06-01", "06/05/2024")},
    }
    from epss import test_only
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "fixture.json"
        source.write_text("{}")
        with patch.object(test_only, "CVEGraphDataset") as constructor:
            test_only._build_dataset({"no_security_frontend": True, "hybrid": True},
                                     Path(temp), labeled_override=source)
            passed = constructor.call_args.kwargs
    result["test_only_omitted_contract"] = {
        "requested_no_security_frontend": True,
        "passes_use_security_frontend": "use_security_frontend" in passed,
        "passes_fixed_tabular_vocab": "tabular_vocab_path" in passed}
    first = TabularFeatureExtractor().fit({"train": {"cwe_ids": ["CWE-79"]}})
    second = TabularFeatureExtractor().fit({"test": {"cwe_ids": ["CWE-89"]}})
    result["refitted_cwe_mapping"] = {
        "training_column_24": first.get_feature_names()[24],
        "external_column_24": second.get_feature_names()[24],
        "same_encoded_vector_for_different_cwes": bool((
            first.encode({"cwe_ids": ["CWE-79"]}) == second.encode({"cwe_ids": ["CWE-89"]})).all())}
    return result


def call_inventory():
    directory = PROJECT_ROOT / "02_prediction/01_package/epss"
    result = {}
    for name in ("run_pipeline.py", "infer.py", "test_only.py", "cross_distribution_eval.py", "backfill_val_predictions.py"):
        path = directory / name
        tree = ast.parse(path.read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CVEGraphDataset":
                calls.append({"line": node.lineno, "arguments": {k.arg: ast.unparse(k.value) for k in node.keywords}})
        result[str(path.relative_to(PROJECT_ROOT))] = calls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT.parent /
        "SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev.json")
    parser.add_argument("--limit", type=int, default=0, help="0 analyzes every description; otherwise a seeded sample")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT /
        "00_documentation/02_tpg/10_mechanism_review")
    args = parser.parse_args()
    constants = Counter()
    def missing(value):
        constants[value] += 1
        return None
    raw = args.source.read_bytes()
    records = json.loads(raw, parse_constant=missing)
    if not isinstance(records, list) or any(not isinstance(r.get("description"), str) for r in records):
        raise ValueError("Expected the documented revision JSON array with text descriptions")
    source_profile = {
        "path": str(args.source), "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(records), "in_vckev_true": sum(r.get("in_vckev") is True for r in records),
        "nonstandard_json_constants": dict(constants),
        "descriptions_with_any_cve_id": sum(bool(_CVE_PATTERN.search(r["description"])) for r in records),
        "descriptions_containing_own_cve": sum(r["cve"].lower() in r["description"].lower() for r in records),
        "missing_fields": {key: sum(r.get(key) in (None, "") for r in records)
                           for key in ("epss_score", "binary_label", "llm_summary", "social_media_post")},
        "case_expansion_descriptions": sum(len(r["description"].lower()) != len(r["description"]) for r in records),
    }
    torch.set_num_threads(1)
    pipeline = SecurityPipeline(include_security_relations=True)
    checks = probes(pipeline)
    selected = sorted(records, key=lambda r: r["cve"])
    if args.limit > 0 and args.limit < len(selected):
        selected = random.Random(42).sample(selected, args.limit)
    rows, errors = [], []
    edge_totals, graph_counts = Counter(), Counter()
    for index, record in enumerate(selected, start=1):
        try:
            graph = pipeline.run(record["description"], doc_id=record["cve"])
            counts = Counter(e.edge_type.name for e in graph.edges() if isinstance(e.edge_type, SecurityEdgeType))
            edge_totals.update(counts)
            graph_counts.update(counts.keys())
            rows.append({"cve": record["cve"], "nodes": graph.num_nodes, "edges": graph.num_edges,
                         "security_edges": dict(counts), "security_edge_count": sum(counts.values()),
                         "cve_id_mentions": [n.properties.text for n in graph.nodes(NodeType.ENTITY)
                                             if n.properties.entity_type == "CVE_ID"]})
        except Exception as error:
            errors.append({"cve": record["cve"], "error": repr(error)})
        if index % 250 == 0:
            print(f"Analyzed {index}/{len(selected)} descriptions", flush=True)
    sources = sorted((PROJECT_ROOT / "01_tpg/01_core/tpg").rglob("*.py"))
    sources += sorted((PROJECT_ROOT / "02_prediction/01_package/epss").glob("*.py"))
    report = {
        "scope": "Security TPG and EPSS consumer diagnostics; no training or source-data edits",
        "source": source_profile, "probes": checks, "dataset_construction_calls": call_inventory(),
        "corpus": {"mode": "rule_only_security_with_typed_edges", "requested": len(selected),
            "completed": len(rows), "errors": errors,
            "graphs_with_any_security_edge": sum(r["security_edge_count"] > 0 for r in rows),
            "edges_by_type": dict(edge_totals), "graphs_by_edge_type": dict(graph_counts), "per_cve": rows},
        "packages": {p: importlib.metadata.version(p) for p in ("spacy", "en-core-web-sm", "torch", "torch-geometric")},
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": ["No real transformer weights evaluated", "Edge coverage is not semantic accuracy",
                        "Future SMP/EPSS fields are not yet available", "Synthetic examples do not measure corpus error rates"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "security_review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [report["scope"], "\nSOURCE", json.dumps(source_profile, indent=2), "\nCORPUS COVERAGE",
             json.dumps({k: v for k, v in report["corpus"].items() if k != "per_cve"}, indent=2)]
    for name, evidence in checks.items():
        lines.extend(["\n" + name, json.dumps(evidence, indent=2, ensure_ascii=False)])
    lines.extend(["\nDATASET CONSTRUCTION CALLS", json.dumps(report["dataset_construction_calls"], indent=2)])
    (args.output / "security_review.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report["corpus"].items() if k != "per_cve"}, indent=2))


if __name__ == "__main__":
    main()
