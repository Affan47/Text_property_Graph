"""
TPG Frontend Base
=================
Abstract base class for all frontends (parsers).

In Joern, each language has its own frontend:
    - C/C++  -> CDT parser
    - Java   -> Soot / JavaParser
    - JS     -> JavaScript parser
    - Binary -> Ghidra

In TPG, each NLP toolkit has its own frontend:
    - spaCy frontend (dependency parsing, NER, POS)
    - AMR frontend (abstract meaning representation)
    - Stanza frontend (alternative NLP pipeline)

All frontends produce the same TPGSchema-compliant graph.
"""

from abc import ABC, abstractmethod
from typing import Optional
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import TPGSchema, DEFAULT_SCHEMA


class BaseFrontend(ABC):
    """
    Abstract base class for TPG frontends.

    Every frontend must implement parse() which takes raw text
    and returns a TextPropertyGraph with at minimum:
        - Structural nodes (DOCUMENT, SENTENCE, TOKEN)
        - Syntactic edges (DEP)
        - CONTAINS edges (structural hierarchy)

    Additional edges (COREF, RST, SRL) are added by passes,
    just like Joern adds CFG/DFG edges after the initial AST parse.
    """

    def __init__(self, schema: Optional[TPGSchema] = None):
        self.schema = schema or DEFAULT_SCHEMA
        self.name: str = "base"

    @abstractmethod
    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """
        Parse raw text into a TextPropertyGraph.

        This is the equivalent of Joern's importCode / frontend parse step.
        Returns a graph with structural nodes and syntax edges.

        Args:
            text: Raw input text (can be single sentence or full document)
            doc_id: Identifier for this document

        Returns:
            TextPropertyGraph with initial nodes and edges
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
