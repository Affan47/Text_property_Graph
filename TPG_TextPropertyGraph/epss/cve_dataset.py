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
        include_tabular: bool = False,
        max_cves: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.labeled_cves_path = labeled_cves_path
        self.label_mode = label_mode
        self.embedding_dim = embedding_dim
        self.use_hybrid = use_hybrid
        self.include_tabular = include_tabular
        self.max_cves = max_cves
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
        if self.max_cves:
            suffix += f"_n{self.max_cves}"
        return [f"cve_graphs{suffix}.pt"]

    def download(self):
        """Copy labeled_cves.json to raw directory if not present."""
        dst = Path(self.raw_dir) / "labeled_cves.json"
        if dst.exists():
            return
        src = Path(self.labeled_cves_path)
        if not src.exists():
            raise FileNotFoundError(
                f"labeled_cves.json not found at {src}. "
                "Run DataCollector.fetch_all() first."
            )
        import shutil
        shutil.copy2(src, dst)

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
            tab_extractor = TabularFeatureExtractor(top_k_cwes=25)
            tab_extractor.fit(labeled_cves)
            logger.info("Tabular features enabled: %d dimensions", tab_extractor.feature_dim)

        data_list = []
        cve_ids = list(labeled_cves.keys())
        if self.max_cves:
            cve_ids = cve_ids[: self.max_cves]

        failed = 0
        for cve_id in tqdm(cve_ids, desc="CVE → TPG → PyG", unit="graph"):
            record = labeled_cves[cve_id]
            description = record["description"]

            try:
                data = self._cve_to_pyg(pipeline, cve_id, description, record,
                                         tab_extractor=tab_extractor)
                if data is not None:
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
            except Exception as e:
                logger.warning("Failed to process %s: %s", cve_id, e)
                failed += 1

        logger.info(
            "Processed %d graphs (%d failed) from %d CVEs",
            len(data_list), failed, len(cve_ids),
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
        vocab = exporter.export_vocab(schema)

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
        """Initialize the TPG pipeline."""
        if self.use_hybrid:
            try:
                from tpg.pipeline import HybridSecurityPipeline
                logger.info("Using HybridSecurityPipeline (rule + SecBERT)")
                return HybridSecurityPipeline()
            except ImportError:
                logger.warning("HybridSecurityPipeline unavailable, falling back to rule-only")

        from tpg.pipeline import SecurityPipeline
        logger.info("Using SecurityPipeline (rule-only)")
        return SecurityPipeline()

    def _cve_to_pyg(self, pipeline, cve_id: str, description: str, record: dict,
                    tab_extractor=None) -> Optional[Data]:
        """Convert a single CVE description to a PyG Data object.

        Pipeline: CVE text → HybridSecurityPipeline → TextPropertyGraph → PyGExporter → Data
        """
        if not description or len(description.strip()) < 10:
            return None

        # Run TPG pipeline
        graph = pipeline.run(description, doc_id=cve_id)

        # Skip trivially small graphs (likely rejected/empty CVEs)
        if graph.num_nodes < 3:
            return None

        # Export to PyG dict format using existing exporter
        pyg_dict = pipeline.export_pyg(graph, embedding_dim=self.embedding_dim)

        # Build label
        if self.label_mode == "binary":
            label = record.get("binary_label", 0)
        elif self.label_mode == "soft":
            label = record.get("epss_score", 0.0)
        else:
            label = record.get("binary_label", 0)

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

        # Tabular features for hybrid model
        if tab_extractor is not None:
            tab_vec = tab_extractor.encode(record)
            data.tabular = torch.tensor(tab_vec, dtype=torch.float).unsqueeze(0)  # [1, tab_dim]

        # Store metadata (not used in training but useful for analysis)
        data.cve_id = cve_id
        data.num_node_types = pyg_dict["num_node_types"]
        data.num_edge_types = pyg_dict["num_edge_types"]

        return data

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced binary classification.

        EPSS-style datasets have ~5% positive rate (exploited).
        """
        labels = []
        for data in self:
            labels.append(data.y.item())
        labels = torch.tensor(labels)
        n_pos = labels.sum().item()
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
        """
        import numpy as np
        rng = np.random.RandomState(seed)

        pos_indices = []
        neg_indices = []
        for i in range(len(self)):
            if self[i].y.item() == 1:
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
