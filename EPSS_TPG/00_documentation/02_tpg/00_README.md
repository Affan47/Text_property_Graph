# TPG Reading Guide

The code and artifacts are under `01_tpg/`; root-level TPG aliases are no longer
used. Before running Python commands, follow the one-time editable installation
in [the running guide](../01_start_here/03_RUNNING_THE_PROJECT.md).

1. [General TPG guide](02_TPG_COMPLETE_GUIDE.md): representation and construction.
2. [Security reference](03_SECURITY_TPG_REFERENCE.md): domain-specific extraction and relations.
3. [Four graph examples](04_GRAPH_EXAMPLES.md): compare ordinary prose, vulnerability text, summaries and dedicated security edges.
4. [Application guide](05_APPLICATION_GUIDE.md): document ingestion, retrieval and graph exploration.
5. [Chatbot report](07_CHATBOT_TECHNICAL_REPORT.md): the earlier question-answering application.
6. [Domain examples](08_DOMAIN_EXAMPLES_README.md): other text domains.

The implementation is grouped under [01_tpg](../../01_tpg):

| Folder | Contents |
|---|---|
| `01_core/tpg/` | Schema, frontends, passes, exporters and pipeline |
| `02_examples/01_scripts/` | Runnable examples |
| `02_examples/02_graphson/` | Four reference JSON graphs |
| `02_examples/03_generated/` | Generated outputs and comparisons |
| `02_examples/04_inputs/` | Text and PDF inputs |
| `03_application/tpg_app/` | Document application |
| `04_chatbot/tpg_chatbot/` | Earlier chatbot |
| `05_workspace/` | Local database and uploads |

For the exact script sequence, use [the code reading order](../01_start_here/02_CODE_READING_ORDER.md). The central documentation keeps explanatory files separate from code and generated graphs.
