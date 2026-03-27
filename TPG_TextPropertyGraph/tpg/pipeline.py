"""
TPG Pipeline — Main entry point
================================
Mirrors Joern's workflow:
    1. importCode("vulnerable.c")     -> frontend.parse(text)
    2. Joern runs CFG/DFG/PDG passes  -> pipeline runs coref/discourse/entity/topic passes
    3. joern --export graphson        -> exporter.export(graph)

Usage:
    from tpg.pipeline import TPGPipeline
    pipeline = TPGPipeline()
    graph = pipeline.run("The patient took aspirin. However, his condition worsened.")
    pipeline.export_graphson(graph, "output.json")

Level 2 (Security):
    from tpg.pipeline import SecurityPipeline
    pipeline = SecurityPipeline()
    graph = pipeline.run("CVE-2024-1234: buffer overflow in Apache 2.4.51...")

Level 3 (Cross-Modal):
    from tpg.pipeline import CrossModalPipeline
    pipeline = CrossModalPipeline(cpg_data=joern_graphson)
    graph = pipeline.run("CVE advisory text...", doc_id="cve_2024_1234")
"""

from typing import Optional, List, Dict, Any
from tpg.schema.types import TPGSchema, DEFAULT_SCHEMA, SECURITY_SCHEMA, FULL_SCHEMA
from tpg.schema.graph import TextPropertyGraph
from tpg.frontends.base import BaseFrontend
from tpg.frontends.spacy_frontend import SpacyFrontend
from tpg.passes.enrichment import (
    BasePass, CoreferencePass, DiscoursePass, EntityRelationPass, TopicPass
)
from tpg.exporters.exporters import GraphSONExporter, PyGExporter


class TPGPipeline:
    """
    The main TPG processing pipeline — Level 1 (Generic).

    Mirrors Joern's architecture:
        Joern:  Source Code -> Frontend Parser -> AST -> CFG Pass -> DFG Pass -> CPG -> Export
        TPG:    Text -> SpaCy Frontend -> Syntax -> Coref Pass -> Discourse Pass -> TPG -> Export

    Pass order mirrors Joern's pass dependencies:
        1. CoreferencePass  (DFG — needs entities from frontend)
        2. DiscoursePass    (CDG — needs sentences from frontend)
        3. EntityRelationPass (Call Graph — needs entities + predicates)
        4. TopicPass        (MetaData — needs all tokens)
    """

    def __init__(self, frontend: Optional[BaseFrontend] = None,
                 passes: Optional[List[BasePass]] = None,
                 schema: Optional[TPGSchema] = None):
        self.schema = schema or DEFAULT_SCHEMA
        self.frontend = frontend or SpacyFrontend(schema=self.schema)

        self.passes = passes or [
            CoreferencePass(),
            DiscoursePass(),
            EntityRelationPass(),
            TopicPass(),
        ]

        self._graphson_exporter = GraphSONExporter()
        self._pyg_exporter = PyGExporter()

    def run(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """Run the full pipeline. Equivalent to Joern's importCode()."""
        graph = self.frontend.parse(text, doc_id=doc_id)
        for p in self.passes:
            graph = p.run(graph)
        return graph

    def export_graphson(self, graph: TextPropertyGraph, filepath: str) -> str:
        return self._graphson_exporter.export(graph, filepath)

    def export_graphson_string(self, graph: TextPropertyGraph) -> str:
        return self._graphson_exporter.export_string(graph)

    def export_pyg(self, graph: TextPropertyGraph, label: Optional[int] = None,
                   embedding_dim: int = 0) -> dict:
        return self._pyg_exporter.export(graph, label=label, embedding_dim=embedding_dim)


class SecurityPipeline(TPGPipeline):
    """Level 2 — Security-Aware TPG Pipeline."""

    def __init__(self, passes: Optional[List[BasePass]] = None,
                 schema: Optional[TPGSchema] = None):
        from tpg.frontends.security_frontend import SecurityFrontend
        super().__init__(
            frontend=SecurityFrontend(schema=schema or SECURITY_SCHEMA),
            passes=passes,
            schema=schema or SECURITY_SCHEMA,
        )


class CrossModalPipeline(TPGPipeline):
    """Level 3 — Cross-Modal TPG+CPG Pipeline."""

    def __init__(self, cpg_data: Optional[Dict[str, Any]] = None,
                 passes: Optional[List[BasePass]] = None,
                 schema: Optional[TPGSchema] = None):
        from tpg.frontends.security_frontend import SecurityFrontend
        from tpg.passes.cross_modal import CrossModalPass

        all_passes = passes or [
            CoreferencePass(),
            DiscoursePass(),
            EntityRelationPass(),
            TopicPass(),
        ]
        all_passes.append(CrossModalPass(cpg_data=cpg_data))

        super().__init__(
            frontend=SecurityFrontend(schema=schema or FULL_SCHEMA),
            passes=all_passes,
            schema=schema or FULL_SCHEMA,
        )
        self.cpg_data = cpg_data

    def merge_with_cpg(self, tpg_graph: TextPropertyGraph) -> TextPropertyGraph:
        if not self.cpg_data:
            return tpg_graph
        from tpg.passes.cross_modal import CrossModalAligner
        return CrossModalAligner.merge_graphs(
            tpg_graph, self.cpg_data, doc_id=tpg_graph.doc_id)


def parse_text(text: str, doc_id: str = "") -> TextPropertyGraph:
    """One-liner to parse text into a TPG. Like Joern's importCode()."""
    return TPGPipeline().run(text, doc_id=doc_id)

def parse_security_text(text: str, doc_id: str = "") -> TextPropertyGraph:
    """One-liner to parse security text into a security-aware TPG."""
    return SecurityPipeline().run(text, doc_id=doc_id)
