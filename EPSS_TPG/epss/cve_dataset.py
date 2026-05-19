"""
CVE Graph Dataset — PyG InMemoryDataset for GNN Training
==========================================================
Converts CVE descriptions → TPG (via HybridSecurityPipeline) → PyG Data objects.

Each CVE becomes a single graph:
    - Nodes: tokens, entities, predicates, CVE-IDs, software, etc. (13+ types)
    - Edges: dependency, coreference, discourse, security relations (13+ types)
    - Node features: one-hot type encoding + SecBERT embeddings (768-dim)
    - Label: binary (1 = exploited via CISA KEV, 0 = not exploited)

Usage:
    dataset = CVEGraphDataset(root="data/epss", labeled_cves_path="data/epss/labeled_cves.json")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable

from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset

# Add TPG project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


class CVEGraphDataset(InMemoryDataset):
    """PyG InMemoryDataset that converts CVE descriptions to TPG graphs.

    Args:
        root: Root directory for dataset storage (raw/ and processed/ subdirs).
        labeled_cves_path: Path to labeled_cves.json from DataCollector.
        label_mode: 'binary' (KEV-based) or 'soft' (EPSS score as regression target).
        embedding_dim: Dimension of SecBERT embeddings (768). Set 0 to skip.
        use_hybrid: If True, use HybridSecurityPipeline (rule + SecBERT model).
                    If False, use SecurityPipeline (rule-only, faster).
        include_tabular: If True, encode tabular features (CVSS, CWE, age, refs)
                        and store as data.tabular for hybrid GNN model.
        include_summary_in_tpg: If True, concatenate `llm_summary` to `description`
                        before feeding the text to the TPG pipeline. Default False
                        preserves the original behaviour (description-only) for
                        reproducibility of the prior 36+ training runs. The
                        colleague-curated `summ_all_sources` / `summ_llama3.1_8b`
                        text is otherwise written to labeled_cves.json but never
                        consumed by the model — see OVERALL_ANALYSIS.md §DI.3.1.
        include_security_edges: If True, run `SecurityRelationsPass` and emit
                        first-class `SecurityEdgeType` edges (SEC_AFFECTS,
                        SEC_EXPLOITED_BY, etc.) between security entities. The
                        edge-type vocabulary expands from 13 to 23 (the GNN's
                        edge embedding gets 10 new slots). Default False
                        preserves the prior 36+ training runs. See
                        TPG_examples/README.md §10 for what this changes.
        max_cves: Limit number of CVEs to process (for development/debugging).
        transform: PyG transform to apply to each Data object.
        pre_transform: PyG pre-transform to apply before saving.
    """

    def __init__(
        self,
        root: str,
        labeled_cves_path: str = "data/epss/labeled_cves.json",
        label_mode: str = "binary",
        embedding_dim: int = 768,
        use_hybrid: bool = True,
        use_security_frontend: bool = True,
        include_tabular: bool = False,
        include_epss_feature: bool = False,
        include_summary_in_tpg: bool = False,
        include_security_edges: bool = False,
        summary_only_tpg: bool = False,
        two_view_tpg: bool = False,
        add_source_labels: bool = False,
        summary_pooling_node: bool = False,
        graph_diagnostics: bool = False,
        tabular_vocab_path: Optional[str] = None,
        max_cves: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.labeled_cves_path = labeled_cves_path
        self.label_mode = label_mode
        self.embedding_dim = embedding_dim
        self.use_hybrid = use_hybrid
        self.use_security_frontend = use_security_frontend
        # When the security frontend is off, security edges are pointless
        # because there are no security entity nodes to connect; force off
        # so the suffix and edge-type vocabulary stay consistent.
        if not use_security_frontend and include_security_edges:
            logger.warning(
                "use_security_frontend=False overrides include_security_edges=True; "
                "no SEC_* edges will be emitted because the security entity nodes "
                "they connect are not created."
            )
            include_security_edges = False
        self.include_tabular = include_tabular
        self.include_epss_feature = include_epss_feature
        self.include_summary_in_tpg = include_summary_in_tpg
        self.include_security_edges = include_security_edges
        self.summary_only_tpg = summary_only_tpg
        self.two_view_tpg = two_view_tpg
        self.add_source_labels = add_source_labels
        self.summary_pooling_node = summary_pooling_node
        self.graph_diagnostics = graph_diagnostics
        self.tabular_vocab_path = tabular_vocab_path
        self.max_cves = max_cves
        if self.summary_only_tpg and self.two_view_tpg:
            raise ValueError("--summary-only-tpg and --two-view-tpg are mutually exclusive")
        self._sync_raw_labeled_file(root, labeled_cves_path)
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["labeled_cves.json"]

    @property
    def processed_file_names(self) -> List[str]:
        suffix = f"_{self.label_mode}_emb{self.embedding_dim}"
        if self.include_tabular:
            suffix += "_tab"
            if self.include_epss_feature:
                suffix += "_epssfeat"
            if self.tabular_vocab_path:
                suffix += f"_fixedtab{self._short_file_hash(self.tabular_vocab_path)}"
        if self.include_summary_in_tpg:
            suffix += "_withsumm"
        if self.summary_only_tpg:
            suffix += "_summonly"
        if self.two_view_tpg:
            suffix += "_twoview"
        if self.add_source_labels:
            suffix += "_srclabel"
        if self.summary_pooling_node:
            suffix += "_summpool"
        if self.include_security_edges:
            suffix += "_secedges"
        if not self.use_security_frontend:
            suffix += "_nosec"
        if self.max_cves:
            suffix += f"_n{self.max_cves}"
        return [f"cve_graphs{suffix}.pt"]

    @staticmethod
    def _short_file_hash(path: str) -> str:
        import hashlib

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]

    @staticmethod
    def _sync_raw_labeled_file(root: str, labeled_cves_path: str) -> None:
        """Copy labeled_cves.json to raw/, even when PyG would skip download().

        Uses a SHA-256 hash to detect when the source file has been regenerated
        (e.g. after running csv_adapter with a shuffled dataset). When the hash
        differs the raw file is overwritten and processed caches are deleted so
        graphs are rebuilt from the fresh data.
        """
        import hashlib
        import shutil

        src = Path(labeled_cves_path)
        if not src.exists():
            raise FileNotFoundError(
                f"labeled_cves.json not found at {src}. "
                "Run DataCollector.fetch_all() or epss.csv_adapter first."
            )

        root_path = Path(root)
        raw_dir = root_path / "raw"
        processed_dir = root_path / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        dst = raw_dir / "labeled_cves.json"

        # Compute hash of source file
        hasher = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        src_hash = hasher.hexdigest()

        hash_file = raw_dir / "labeled_cves.sha256"

        if dst.exists() and hash_file.exists():
            stored_hash = hash_file.read_text().strip()
            if stored_hash == src_hash:
                return  # raw file is up-to-date, nothing to do

        logger.info("labeled_cves.json changed (or first run) — syncing raw file")
        shutil.copy2(src, dst)
        hash_file.write_text(src_hash)

        # Invalidate processed graph caches so they rebuild from fresh raw data
        if processed_dir.exists():
            for pt_file in processed_dir.glob("cve_graphs_*.pt"):
                logger.info("Invalidating stale graph cache: %s", pt_file.name)
                pt_file.unlink()

    def download(self):
        """Copy labeled_cves.json to raw directory, updating if source has changed."""
        self._sync_raw_labeled_file(self.root, self.labeled_cves_path)

    def process(self):
        """Convert each CVE description → TPG → PyG Data object."""
        raw_path = Path(self.raw_dir) / "labeled_cves.json"
        with open(raw_path) as f:
            labeled_cves = json.load(f)

        logger.info("Processing %d CVE records into TPG graphs...", len(labeled_cves))

        # Initialize TPG pipeline
        pipeline = self._init_pipeline()

        # Export and save vocab files (like SemVul's vocab_builder)
        self._export_vocab(pipeline)

        # Initialize tabular feature extractor if needed
        tab_extractor = None
        if self.include_tabular:
            from epss.tabular_features import TabularFeatureExtractor
            tab_extractor = TabularFeatureExtractor(
                top_k_cwes=25,
                include_epss_feature=self.include_epss_feature,
            )
            if self.tabular_vocab_path:
                vocab_path = Path(self.tabular_vocab_path)
                if not vocab_path.exists():
                    raise FileNotFoundError(f"Tabular vocab not found: {vocab_path}")
                with open(vocab_path) as _f:
                    vocab_payload = json.load(_f)
                tab_extractor.set_cwe_vocab(vocab_payload.get("cwe_to_idx", {}))
                logger.info("Using fixed tabular vocab from %s", vocab_path)
            else:
                tab_extractor.fit(labeled_cves)
            logger.info(
                "Tabular features enabled: %d dimensions (include_epss=%s)",
                tab_extractor.feature_dim, self.include_epss_feature,
            )
            # Save CWE vocab so inference can reuse the exact same feature mapping
            tab_vocab_path = Path(self.processed_dir) / "tabular_vocab.json"
            with open(tab_vocab_path, "w") as _f:
                json.dump({"cwe_to_idx": tab_extractor.cwe_to_idx}, _f, indent=2)
            logger.info("Saved tabular vocab → %s", tab_vocab_path.name)

        data_list = []
        cve_ids = list(labeled_cves.keys())
        if self.max_cves:
            cve_ids = cve_ids[: self.max_cves]

        failed = 0
        dropped = 0
        n_with_summary = 0
        n_missing_summary = 0
        graph_stats = []
        for cve_id in tqdm(cve_ids, desc="CVE → TPG → PyG", unit="graph"):
            record = labeled_cves[cve_id]
            description = (record.get("description") or "").strip()
            llm_summary = (record.get("llm_summary") or "").strip()

            if llm_summary:
                n_with_summary += 1
            else:
                n_missing_summary += 1

            if self.summary_only_tpg and not llm_summary:
                dropped += 1
                continue

            try:
                data = self._cve_to_pyg(
                    pipeline,
                    cve_id,
                    description,
                    record,
                    tab_extractor=tab_extractor,
                    summary=llm_summary,
                )
                if data is not None:
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                    if self.graph_diagnostics:
                        graph_stats.append(self._graph_stats(data, has_summary=bool(llm_summary)))
                else:
                    dropped += 1
            except Exception as e:
                logger.warning("Failed to process %s: %s", cve_id, e)
                failed += 1

        logger.info(
            "Processed %d graphs (%d dropped, %d failed) from %d CVEs",
            len(data_list), dropped, failed, len(cve_ids),
        )
        if self.include_summary_in_tpg:
            logger.info(
                "include_summary_in_tpg=True — concatenated llm_summary into "
                "description for %d / %d CVEs (others had empty summary)",
                n_with_summary, len(cve_ids),
            )
        if self.summary_only_tpg:
            logger.info(
                "summary_only_tpg=True — built graphs from llm_summary only; "
                "%d / %d CVEs had empty summaries and were skipped",
                n_missing_summary, len(cve_ids),
            )
        if self.two_view_tpg:
            logger.info(
                "two_view_tpg=True — built description and summary as separate "
                "subgraphs for %d / %d CVEs with summaries",
                n_with_summary, len(cve_ids),
            )
        if self.graph_diagnostics:
            self._save_graph_diagnostics(graph_stats)

        if not data_list:
            raise RuntimeError(
                "No graphs were created. "
                f"Input CVEs={len(cve_ids)}, dropped={dropped}, failed={failed}, "
                f"non_empty_llm_summary={n_with_summary}. "
                "For --summary-only-tpg, check that labeled_cves.json contains "
                "non-empty llm_summary values. If using a CSV directly, make sure "
                "the summary column is named summary, llm_summary, summ_all_sources, "
                "summ_llama3.1_8b, or summ_github_urls."
            )

        self.save(data_list, self.processed_paths[0])

    def _export_vocab(self, pipeline):
        """Export edge_type_vocab.json and node_type_vocab.json (like SemVul's vocab_builder).

        These files document the integer mapping from type names to indices,
        ensuring the GNN's edge-type-aware layers match the data encoding.
        """
        from tpg.exporters.exporters import PyGExporter

        schema = pipeline.schema
        exporter = PyGExporter()
        vocab = exporter.export_vocab(
            schema,
            use_security_edge_types=self.include_security_edges,
        )

        # Save edge type vocab
        edge_vocab_path = Path(self.processed_dir) / "edge_type_vocab.json"
        with open(edge_vocab_path, "w") as f:
            json.dump(vocab["edge_types"], f, indent=2)

        # Save node type vocab
        node_vocab_path = Path(self.processed_dir) / "node_type_vocab.json"
        with open(node_vocab_path, "w") as f:
            json.dump(vocab["node_types"], f, indent=2)

        logger.info(
            "Saved vocab: %d node types → %s, %d edge types → %s",
            vocab["num_node_types"], node_vocab_path.name,
            vocab["num_edge_types"], edge_vocab_path.name,
        )

        # Log the full mapping for transparency
        logger.info("Edge type vocab (name → index):")
        for name, idx in sorted(vocab["edge_types"].items(), key=lambda x: x[1]):
            logger.info("  [%2d] %s", idx, name)

    def _init_pipeline(self):
        """Initialize the TPG pipeline.

        Three modes, selected by the constructor flags:
          * use_security_frontend=False -> plain TPGPipeline (spaCy only).
            No security entity nodes, no SEC_* edges. Used for the
            security-ablation baseline.
          * use_security_frontend=True, use_hybrid=True (default) ->
            HybridSecurityPipeline (rule + SecBERT fallback).
          * use_security_frontend=True, use_hybrid=False ->
            SecurityPipeline (rule-only, faster).
        """
        if not self.use_security_frontend:
            from tpg.pipeline import TPGPipeline
            logger.info("Using plain TPGPipeline (spaCy only, no security frontend, "
                        "no SEC_* edges) -- baseline for the security ablation")
            return TPGPipeline()

        if self.use_hybrid:
            try:
                from tpg.pipeline import HybridSecurityPipeline
                logger.info("Using HybridSecurityPipeline (rule + SecBERT)%s",
                            " + SecurityRelationsPass" if self.include_security_edges else "")
                return HybridSecurityPipeline(
                    include_security_relations=self.include_security_edges,
                )
            except ImportError:
                logger.warning("HybridSecurityPipeline unavailable, falling back to rule-only")

        from tpg.pipeline import SecurityPipeline
        logger.info("Using SecurityPipeline (rule-only)%s",
                    " + SecurityRelationsPass" if self.include_security_edges else "")
        return SecurityPipeline(include_security_relations=self.include_security_edges)

    def _cve_to_pyg(self, pipeline, cve_id: str, description: str, record: dict,
                    tab_extractor=None, summary: str = "") -> Optional[Data]:
        """Convert a single CVE description to a PyG Data object.

        Pipeline: CVE text → HybridSecurityPipeline → TextPropertyGraph → PyGExporter → Data
        """
        if self.two_view_tpg:
            return self._cve_to_two_view_pyg(
                pipeline, cve_id, description, summary, record, tab_extractor
            )

        text = description
        description_len = len(description)
        summary_start = None
        if self.summary_only_tpg:
            text = summary
            description_len = 0
            summary_start = 0
        elif self.include_summary_in_tpg and summary:
            text = description + "\n\n" + summary
            summary_start = len(description) + 2

        if not text or len(text.strip()) < 10:
            return None

        # Run TPG pipeline
        graph = pipeline.run(text, doc_id=cve_id)

        # Skip trivially small graphs (likely rejected/empty CVEs)
        if graph.num_nodes < 3:
            return None

        node_source_type = self._infer_node_sources(
            graph,
            mode="summary_only" if self.summary_only_tpg else "combined",
            description_len=description_len,
            summary_start=summary_start,
        )
        self._mark_graph_sources(graph, node_source_type)

        # Export to PyG dict format using existing exporter
        pyg_dict = pipeline.export_pyg(
            graph,
            embedding_dim=self.embedding_dim,
            use_security_edge_types=self.include_security_edges,
        )

        label = self._label_from_record(record)

        # Convert to PyG Data object
        x = torch.tensor(pyg_dict["x"], dtype=torch.float)
        edge_index = torch.tensor(pyg_dict["edge_index"], dtype=torch.long)
        edge_type = torch.tensor(pyg_dict["edge_type"], dtype=torch.long)
        edge_attr = torch.tensor(pyg_dict["edge_attr"], dtype=torch.float)

        if self.label_mode == "soft":
            y = torch.tensor([label], dtype=torch.float)
        else:
            y = torch.tensor([label], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            y=y,
            num_nodes=pyg_dict["num_nodes"],
        )
        self._attach_source_metadata(data, node_source_type)

        if self.add_source_labels:
            self._append_source_features(data)

        if self.summary_pooling_node and summary:
            self._add_summary_pooling_node(data)
        self._ensure_optional_flags(data)

        # Tabular features for hybrid model
        if tab_extractor is not None:
            tab_vec = tab_extractor.encode(record)
            data.tabular = torch.tensor(tab_vec, dtype=torch.float).unsqueeze(0)  # [1, tab_dim]

        # Store metadata (not used in training but useful for analysis)
        data.cve_id = cve_id
        data.num_node_types = pyg_dict["num_node_types"]
        data.num_edge_types = pyg_dict["num_edge_types"]

        return data

    def _cve_to_two_view_pyg(self, pipeline, cve_id: str, description: str,
                             summary: str, record: dict, tab_extractor=None) -> Optional[Data]:
        """Build separate description and summary subgraphs for source-aware fusion."""
        desc_data = self._text_to_data(
            pipeline, cve_id=f"{cve_id}::description", text=description,
            source_type=0, record=record,
        )
        if desc_data is None:
            return None

        summary_data = None
        if summary and len(summary.strip()) >= 10:
            summary_data = self._text_to_data(
                pipeline, cve_id=f"{cve_id}::summary", text=summary,
                source_type=1, record=record,
            )

        data = self._merge_view_data(desc_data, summary_data)

        if self.add_source_labels:
            self._append_source_features(data)

        if self.summary_pooling_node and summary:
            self._add_summary_pooling_node(data)
        self._ensure_optional_flags(data)

        if tab_extractor is not None:
            tab_vec = tab_extractor.encode(record)
            data.tabular = torch.tensor(tab_vec, dtype=torch.float).unsqueeze(0)

        data.cve_id = cve_id
        data.num_node_types = desc_data.num_node_types
        data.num_edge_types = desc_data.num_edge_types
        return data

    def _text_to_data(self, pipeline, cve_id: str, text: str, source_type: int,
                      record: dict) -> Optional[Data]:
        if not text or len(text.strip()) < 10:
            return None

        graph = pipeline.run(text, doc_id=cve_id)
        if graph.num_nodes < 3:
            return None

        node_source_type = [source_type] * graph.num_nodes
        self._mark_graph_sources(graph, node_source_type)
        pyg_dict = pipeline.export_pyg(
            graph,
            embedding_dim=self.embedding_dim,
            use_security_edge_types=self.include_security_edges,
        )

        label = self._label_from_record(record)
        x = torch.tensor(pyg_dict["x"], dtype=torch.float)
        edge_index = torch.tensor(pyg_dict["edge_index"], dtype=torch.long)
        edge_type = torch.tensor(pyg_dict["edge_type"], dtype=torch.long)
        edge_attr = torch.tensor(pyg_dict["edge_attr"], dtype=torch.float)
        y_dtype = torch.float if self.label_mode == "soft" else torch.long

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=y_dtype),
            num_nodes=pyg_dict["num_nodes"],
        )
        self._attach_source_metadata(data, node_source_type)
        data.cve_id = cve_id
        data.num_node_types = pyg_dict["num_node_types"]
        data.num_edge_types = pyg_dict["num_edge_types"]
        return data

    def _label_from_record(self, record: dict):
        if self.label_mode == "binary":
            return record.get("binary_label", 0)
        if self.label_mode == "soft":
            return record.get("epss_score", 0.0)
        return record.get("binary_label", 0)

    def _merge_view_data(self, desc_data: Data, summary_data: Optional[Data]) -> Data:
        if summary_data is None:
            desc_data.view_present = torch.tensor([[1.0, 0.0]], dtype=torch.float)
            return desc_data

        offset = desc_data.num_nodes
        data = Data(
            x=torch.cat([desc_data.x, summary_data.x], dim=0),
            edge_index=torch.cat(
                [desc_data.edge_index, summary_data.edge_index + offset],
                dim=1,
            ),
            edge_type=torch.cat([desc_data.edge_type, summary_data.edge_type], dim=0),
            edge_attr=torch.cat([desc_data.edge_attr, summary_data.edge_attr], dim=0),
            y=desc_data.y,
            num_nodes=int(desc_data.num_nodes + summary_data.num_nodes),
        )
        data.node_source_type = torch.cat(
            [desc_data.node_source_type, summary_data.node_source_type], dim=0
        )
        data.edge_source_type = torch.cat(
            [desc_data.edge_source_type, summary_data.edge_source_type], dim=0
        )
        data.view_present = torch.tensor([[1.0, 1.0]], dtype=torch.float)
        data.num_node_types = desc_data.num_node_types
        data.num_edge_types = desc_data.num_edge_types
        return data

    def _infer_node_sources(self, graph, mode: str, description_len: int,
                            summary_start: Optional[int]) -> List[int]:
        if mode == "summary_only":
            return [1] * graph.num_nodes
        if summary_start is None:
            return [0] * graph.num_nodes

        sources = []
        for node in graph.nodes():
            start = node.properties.char_start
            end = node.properties.char_end
            if start >= summary_start:
                sources.append(1)
            elif end <= description_len or end == 0:
                sources.append(0)
            else:
                sources.append(2)
        return sources

    def _mark_graph_sources(self, graph, node_source_type: List[int]) -> None:
        names = {0: "description", 1: "summary", 2: "mixed"}
        source_by_node_id = {}
        for node, source_id in zip(graph.nodes(), node_source_type):
            source_name = names.get(int(source_id), "mixed")
            node.properties.extra["source_text_type"] = source_name
            source_by_node_id[node.id] = int(source_id)

        for edge in graph.edges():
            src = source_by_node_id.get(edge.source, 2)
            dst = source_by_node_id.get(edge.target, 2)
            source_id = src if src == dst else 2
            edge.properties.extra["source_text_type"] = names.get(source_id, "mixed")

    def _attach_source_metadata(self, data: Data, node_source_type: List[int]) -> None:
        node_sources = torch.tensor(node_source_type, dtype=torch.long)
        data.node_source_type = node_sources

        if data.edge_index.numel() == 0:
            data.edge_source_type = torch.empty((0,), dtype=torch.long)
            return

        src = data.edge_index[0]
        dst = data.edge_index[1]
        data.edge_source_type = torch.where(
            node_sources[src] == node_sources[dst],
            node_sources[src],
            torch.full_like(src, 2),
        )

    def _append_source_features(self, data: Data) -> None:
        one_hot = torch.zeros((data.num_nodes, 3), dtype=data.x.dtype)
        source = data.node_source_type.clamp(min=0, max=2)
        one_hot[torch.arange(data.num_nodes), source] = 1.0
        data.x = torch.cat([data.x, one_hot], dim=-1)
        data.has_source_label_features = torch.tensor([1], dtype=torch.long)

    def _add_summary_pooling_node(self, data: Data) -> None:
        summary_mask = data.node_source_type == 1
        if not torch.any(summary_mask):
            return

        x_dim = data.x.size(1)
        node_type_dim = int(data.num_node_types)
        feature = torch.zeros((1, x_dim), dtype=data.x.dtype)

        # NodeType.SENTENCE is index 2 in the base TPG schema.
        if node_type_dim > 2:
            feature[0, 2] = 1.0

        emb_start = node_type_dim
        emb_end = min(node_type_dim + self.embedding_dim, x_dim)
        if emb_end > emb_start:
            summary_sentence_mask = summary_mask & (data.x[:, 2] > 0.5)
            emb_source_mask = summary_sentence_mask if torch.any(summary_sentence_mask) else summary_mask
            feature[0, emb_start:emb_end] = data.x[emb_source_mask, emb_start:emb_end].mean(dim=0)

        if self.add_source_labels and x_dim >= node_type_dim + self.embedding_dim + 3:
            feature[0, -3:] = torch.tensor([0.0, 1.0, 0.0], dtype=data.x.dtype)

        new_idx = data.num_nodes
        data.x = torch.cat([data.x, feature], dim=0)
        data.node_source_type = torch.cat([
            data.node_source_type,
            torch.tensor([1], dtype=torch.long),
        ])
        data.num_nodes = int(data.x.size(0))
        data.has_summary_pooling_node = torch.tensor([1], dtype=torch.long)

        doc_candidates = torch.where(data.x[:new_idx, 0] > 0.5)[0]
        doc_idx = int(doc_candidates[0].item()) if len(doc_candidates) else 0
        contains_idx = 9
        data.edge_index = torch.cat([
            data.edge_index,
            torch.tensor([[doc_idx], [new_idx]], dtype=torch.long),
        ], dim=1)
        data.edge_type = torch.cat([
            data.edge_type,
            torch.tensor([contains_idx], dtype=torch.long),
        ])

        edge_attr_dim = int(data.edge_attr.size(1)) if data.edge_attr.dim() == 2 else int(data.num_edge_types)
        edge_feature = torch.zeros((1, edge_attr_dim), dtype=data.edge_attr.dtype)
        if contains_idx < edge_attr_dim:
            edge_feature[0, contains_idx] = 1.0
        data.edge_attr = torch.cat([data.edge_attr, edge_feature], dim=0)
        data.edge_source_type = torch.cat([
            data.edge_source_type,
            torch.tensor([1], dtype=torch.long),
        ])

    def _ensure_optional_flags(self, data: Data) -> None:
        if not hasattr(data, "has_source_label_features"):
            data.has_source_label_features = torch.tensor([0], dtype=torch.long)
        if not hasattr(data, "has_summary_pooling_node"):
            data.has_summary_pooling_node = torch.tensor([0], dtype=torch.long)

    def _graph_stats(self, data: Data, has_summary: bool) -> dict:
        node_sources = getattr(data, "node_source_type", torch.empty((0,), dtype=torch.long))
        edge_sources = getattr(data, "edge_source_type", torch.empty((0,), dtype=torch.long))

        def count_source(values, source_id):
            if values.numel() == 0:
                return 0
            return int((values == source_id).sum().item())

        n_nodes = int(data.num_nodes)
        n_edges = int(data.edge_index.size(1))
        return {
            "cve_id": str(getattr(data, "cve_id", "")),
            "has_summary": bool(has_summary),
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "mean_degree": float(n_edges / max(n_nodes, 1)),
            "description_nodes": count_source(node_sources, 0),
            "summary_nodes": count_source(node_sources, 1),
            "mixed_nodes": count_source(node_sources, 2),
            "description_edges": count_source(edge_sources, 0),
            "summary_edges": count_source(edge_sources, 1),
            "mixed_edges": count_source(edge_sources, 2),
            "has_summary_pooling_node": bool(
                hasattr(data, "has_summary_pooling_node")
                and int(data.has_summary_pooling_node.item()) == 1
            ),
        }

    def _save_graph_diagnostics(self, graph_stats: List[dict]) -> None:
        if not graph_stats:
            return
        summary = {
            "num_graphs": len(graph_stats),
            "mean_nodes": sum(s["num_nodes"] for s in graph_stats) / len(graph_stats),
            "mean_edges": sum(s["num_edges"] for s in graph_stats) / len(graph_stats),
            "mean_degree": sum(s["mean_degree"] for s in graph_stats) / len(graph_stats),
            "mean_description_nodes": sum(s["description_nodes"] for s in graph_stats) / len(graph_stats),
            "mean_summary_nodes": sum(s["summary_nodes"] for s in graph_stats) / len(graph_stats),
            "mean_mixed_nodes": sum(s["mixed_nodes"] for s in graph_stats) / len(graph_stats),
        }
        payload = {"summary": summary, "graphs": graph_stats}
        out_path = Path(self.processed_dir) / f"{Path(self.processed_paths[0]).stem}_graph_diagnostics.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Saved graph diagnostics → %s | mean nodes=%.1f, mean edges=%.1f",
            out_path.name, summary["mean_nodes"], summary["mean_edges"],
        )

    # Threshold used when label_mode='soft' to decide what counts as "positive"
    # for stratified splitting and class-weight computation.
    SOFT_POS_THRESHOLD = 0.1  # EPSS >= 0.1 → positive class

    def _is_positive(self, y_value: float) -> bool:
        """Return True if this sample is 'positive' for stratification purposes.

        binary mode: y == 1 (KEV label)
        soft   mode: EPSS score >= SOFT_POS_THRESHOLD (top ~15% of dataset)
        """
        if self.label_mode == "soft":
            return y_value >= self.SOFT_POS_THRESHOLD
        return y_value == 1

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced classification.

        Works for both binary (KEV) and soft (EPSS regression) label modes.
        In soft mode, 'positive' = EPSS >= SOFT_POS_THRESHOLD.
        """
        labels = [data.y.item() for data in self]
        n_pos = sum(1 for y in labels if self._is_positive(y))
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0, 1.0])
        weight_neg = len(labels) / (2.0 * n_neg)
        weight_pos = len(labels) / (2.0 * n_pos)
        return torch.tensor([weight_neg, weight_pos])

    def get_split_indices(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
    ) -> Dict[str, List[int]]:
        """Split dataset indices into train/val/test with stratification.

        Maintains the same positive/negative ratio in each split.
        Works for both binary and soft label modes.
        """
        import numpy as np
        rng = np.random.RandomState(seed)

        pos_indices = []
        neg_indices = []
        for i in range(len(self)):
            if self._is_positive(self[i].y.item()):
                pos_indices.append(i)
            else:
                neg_indices.append(i)

        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_pos < 3:
            logger.warning(
                "Only %d positive samples — forcing at least 1 per split", n_pos
            )
            # Guarantee at least 1 positive in each split
            pos_split = {
                "train": pos_indices[:max(1, n_pos - 2)],
                "val": pos_indices[max(1, n_pos - 2) : max(1, n_pos - 2) + min(1, n_pos)],
                "test": pos_indices[-min(1, max(0, n_pos - 1)):] if n_pos > 1 else [],
            }
        else:
            n_pos_train = max(1, int(n_pos * train_ratio))
            n_pos_val = max(1, int(n_pos * val_ratio))
            n_pos_test = max(1, n_pos - n_pos_train - n_pos_val)
            pos_split = {
                "train": pos_indices[:n_pos_train],
                "val": pos_indices[n_pos_train : n_pos_train + n_pos_val],
                "test": pos_indices[n_pos_train + n_pos_val :],
            }

        n_neg_train = int(n_neg * train_ratio)
        n_neg_val = int(n_neg * val_ratio)
        neg_split = {
            "train": neg_indices[:n_neg_train],
            "val": neg_indices[n_neg_train : n_neg_train + n_neg_val],
            "test": neg_indices[n_neg_train + n_neg_val :],
        }

        logger.info(
            "Split: train=%d (%d+), val=%d (%d+), test=%d (%d+)",
            len(pos_split["train"]) + len(neg_split["train"]), len(pos_split["train"]),
            len(pos_split["val"]) + len(neg_split["val"]), len(pos_split["val"]),
            len(pos_split["test"]) + len(neg_split["test"]), len(pos_split["test"]),
        )

        return {
            "train": pos_split["train"] + neg_split["train"],
            "val": pos_split["val"] + neg_split["val"],
            "test": pos_split["test"] + neg_split["test"],
        }
