"""
TPG Document Chatbot
====================
A question-answering chatbot that uses your TPG knowledge graph as the
retrieval back-end and the Claude API for answer generation.

Architecture
────────────

    ┌──────────────────────────────────────────────────────────────────┐
    │                 TPG DOCUMENT QA SYSTEM                           │
    └──────────────────────────────────────────────────────────────────┘

    INGESTION (run once)
    ┌────────────────────────────────────────────────┐
    │  PDFs / DOCX / TXT                             │
    │        │                                       │
    │        ▼                                       │
    │  pdfplumber / python-docx                      │
    │  → paragraph chunks                            │
    │        │                                       │
    │        ▼                                       │
    │  TPG SecurityPipeline (SpaCy + rule NER)       │
    │  → nodes: ENTITY, PREDICATE, CONCEPT           │
    │  → edges: ENTITY_REL, COREF, RST_RELATION      │
    │        │                                       │
    │        ▼                                       │
    │  GraphStore (entity_index + passage_store)     │
    │  → saved to store.json                         │
    └────────────────────────────────────────────────┘

    QUERY (each turn)
    ┌────────────────────────────────────────────────┐
    │  User question                                 │
    │        │                                       │
    │        ▼                                       │
    │  TPG parse question                            │
    │  → extract entities + predicates               │
    │        │                                       │
    │        ▼                                       │
    │  GraphStore.retrieve()                         │
    │  → entity_index lookup (exact + partial)       │
    │  → score: 2×entity_hits + 1×predicate_hits     │
    │  → top-6 passages                              │
    │        │                                       │
    │        ▼                                       │
    │  Claude API (claude-sonnet-4-6)                │
    │  System:  role + graph context (CACHED)        │
    │  User:    retrieved passages + question        │
    │        │                                       │
    │        ▼                                       │
    │  Streaming answer with source attribution      │
    └────────────────────────────────────────────────┘

    Prompt caching (Anthropic)
    ──────────────────────────
    The system prompt is marked with cache_control so Anthropic caches it
    across turns. This cuts cost and latency for multi-turn conversations
    over the same document set.

Usage
─────
    # Step 1 — Build the graph store (run once)
    cd /home/ayounas/Text_property_Graph/EPSS_TPG
    python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json

    # Step 2 — Start the chatbot
    export ANTHROPIC_API_KEY=sk-ant-...
    python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json

    # One-shot query
    python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json --query "What CVEs affect Apache?"
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from tpg_chatbot.graph_store import GraphStore
from tpg_chatbot.retriever import retrieve, format_context, extract_question_entities


# ── Claude client (with prompt caching) ─────────────────────────────────────

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a document intelligence assistant powered by a Text Property Graph (TPG) knowledge base.

The TPG system has parsed the user's documents into a structured knowledge graph:
  - ENTITY nodes: named entities (CVEs, products, vendors, people, concepts)
  - PREDICATE nodes: actions and relationships between entities
  - COREF edges: coreference chains (pronouns resolved to antecedents)
  - ENTITY_REL edges: semantic relationships between entities
  - RST_RELATION edges: discourse structure (cause, contrast, elaboration)

When answering:
1. Base your answer strictly on the retrieved context passages provided.
2. Cite the source document and page for each factual claim (shown in [Source: ...] tags).
3. If the context does not contain enough information to answer, say so clearly — do not hallucinate.
4. When multiple passages discuss the same entity, synthesise them into a coherent answer.
5. Preserve technical precision: CVE IDs, CVSS scores, CWE codes, version numbers must be exact.
"""


def build_messages(
    conversation_history: List[dict],
    question: str,
    context: str,
) -> List[dict]:
    """
    Build the messages list for the Claude API.
    Injects retrieved context into each user turn.
    """
    messages = list(conversation_history)

    user_content = f"""\
<retrieved_context>
{context}
</retrieved_context>

Question: {question}"""

    messages.append({"role": "user", "content": user_content})
    return messages


def ask(
    client: anthropic.Anthropic,
    store: GraphStore,
    question: str,
    conversation_history: List[dict],
    top_k: int = 6,
    stream: bool = True,
) -> str:
    """
    Single-turn QA with TPG retrieval + Claude generation.

    Returns the assistant response text and appends both user and
    assistant turns to conversation_history (for multi-turn context).
    """
    # ── 1. TPG retrieval ──────────────────────────────────────────────
    entities, predicates = extract_question_entities(question)
    passages = retrieve(store, question, top_k=top_k)

    if passages:
        context = format_context(passages)
        n_sources = len({p.doc_id for p in passages})
        print(f"\n[GRAPH] Matched {len(passages)} passages from {n_sources} document(s)")
        if entities:
            print(f"[GRAPH] Query entities: {', '.join(entities[:6])}")
    else:
        context = "No relevant passages found in the knowledge graph for this question."
        print("\n[GRAPH] No matching passages — answering from general knowledge only")

    # ── 2. Build prompt ───────────────────────────────────────────────
    messages = build_messages(conversation_history, question, context)

    # ── 3. Claude API call (with prompt caching) ──────────────────────
    system_with_cache = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # Cache system prompt across turns
        }
    ]

    response_text = ""

    if stream:
        print("\nAssistant: ", end="", flush=True)
        with client.messages.stream(
            model=MODEL,
            max_tokens=2048,
            system=system_with_cache,
            messages=messages,
        ) as stream_resp:
            for chunk in stream_resp.text_stream:
                print(chunk, end="", flush=True)
                response_text += chunk
        print()  # newline after streaming
    else:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_with_cache,
            messages=messages,
        )
        response_text = resp.content[0].text
        print(f"\nAssistant: {response_text}")

    # ── 4. Update conversation history ────────────────────────────────
    user_content = f"<retrieved_context>\n{context}\n</retrieved_context>\n\nQuestion: {question}"
    conversation_history.append({"role": "user", "content": user_content})
    conversation_history.append({"role": "assistant", "content": response_text})

    return response_text


# ── CLI chatbot loop ─────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            TPG Document Intelligence Chatbot                 ║
║  Powered by Text Property Graph + Claude claude-sonnet-4-6   ║
╚══════════════════════════════════════════════════════════════╝
Commands:
  /quit      — exit
  /stats     — show graph store statistics
  /clear     — clear conversation history
  /sources   — list indexed documents
  /reload    — reload the graph store from disk

Input:
  Single-line  — type your question and press Enter
  Multi-line   — press Enter after each line; blank Enter to submit
"""


def read_input() -> str:
    """
    Multi-line input reader.

    Behaviour:
      - Commands (/quit, /stats, etc.) submit on the first Enter.
      - Any other input: press Enter after each line to continue typing.
        Press Enter on a blank line to submit the full question.

    This lets the user paste multi-paragraph questions without the
    chatbot treating each line as a separate query.
    """
    try:
        first_line = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        raise

    # Commands and empty input submit immediately
    if not first_line or first_line.startswith("/"):
        return first_line

    # Collect continuation lines until a blank Enter
    lines = [first_line]
    while True:
        try:
            line = input("... ").strip()
        except (KeyboardInterrupt, EOFError):
            raise
        if not line:
            break
        lines.append(line)

    return " ".join(lines)


def run_cli(store: GraphStore, store_path: str, one_shot_query: Optional[str] = None) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ERROR] Set ANTHROPIC_API_KEY environment variable before running.")
        print("        export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    conversation_history: List[dict] = []

    if one_shot_query:
        ask(client, store, one_shot_query, conversation_history)
        return

    print(BANNER)
    print(f"[STORE] {store.summary()}\n")

    while True:
        try:
            question = read_input()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting]")
            break

        if not question:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if question == "/quit":
            break
        elif question == "/stats":
            print(f"[STORE] {store.summary()}")
            continue
        elif question == "/clear":
            conversation_history.clear()
            print("[INFO] Conversation history cleared.")
            continue
        elif question == "/sources":
            docs = sorted({p.doc_id for p in store.passages.values()})
            print(f"[STORE] {len(docs)} indexed documents:")
            for d in docs:
                count = sum(1 for p in store.passages.values() if p.doc_id == d)
                print(f"  • {d}  ({count} passages)")
            continue
        elif question == "/reload":
            store = GraphStore.load(store_path)
            print(f"[STORE] Reloaded: {store.summary()}")
            continue

        # ── QA ────────────────────────────────────────────────────────
        ask(client, store, question, conversation_history)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPG Document Intelligence Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to the GraphStore JSON file (built by ingest.py)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="One-shot query — print answer and exit (no interactive loop)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of passages to retrieve per question (default: 6)",
    )
    args = parser.parse_args()

    store_file = Path(args.store)
    if not store_file.exists():
        print(f"[ERROR] Store file not found: {store_file}")
        print("        Run ingestion first:")
        print(f"        python -m tpg_chatbot.ingest --input <documents_dir> --store {store_file}")
        sys.exit(1)

    print(f"[STORE] Loading {store_file}...")
    store = GraphStore.load(str(store_file))

    run_cli(store, str(store_file), one_shot_query=args.query)


if __name__ == "__main__":
    main()
