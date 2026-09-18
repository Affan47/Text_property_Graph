"""
TPG Platform CLI
================
    python -m tpg_app.cli ingest <path|url|dir> [--domain security] [--db work.db]
    python -m tpg_app.cli query  "question"     [--top-k 6]
    python -m tpg_app.cli ask    "question"          # LLM answer (needs ANTHROPIC_API_KEY)
    python -m tpg_app.cli entities [--limit 30] [--etype CVE_ID]
    python -m tpg_app.cli explore  <entity>
    python -m tpg_app.cli path     <entity A> <entity B>
    python -m tpg_app.cli stats
    python -m tpg_app.cli export   <passage_id> --fmt cypher|graphson
    python -m tpg_app.cli serve  [--host 0.0.0.0] [--port 8742]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".tpg-project-root").is_file()
)))


from tpg.paths import DEFAULT_DATABASE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tpg", description="TPG document intelligence")
    p.add_argument("--db", default=DEFAULT_DATABASE, help="SQLite store path")
    p.add_argument("--domain", default="general",
                   help="domain overlay (general, security, medical, legal, "
                        "financial, scientific, or a registered custom one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="ingest a file, directory, URL or text")
    s.add_argument("source")
    s.add_argument("--name", default=None)
    s.add_argument("--overwrite", action="store_true")

    s = sub.add_parser("query", help="graph + BM25 retrieval")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=6)

    s = sub.add_parser("ask", help="retrieval + Claude answer")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=6)

    s = sub.add_parser("entities", help="top entities")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--etype", default=None)

    s = sub.add_parser("explore", help="entity neighborhood")
    s.add_argument("entity")

    s = sub.add_parser("path", help="multi-hop path between two entities")
    s.add_argument("source")
    s.add_argument("target")
    s.add_argument("--max-hops", type=int, default=4)

    sub.add_parser("stats", help="store statistics")

    s = sub.add_parser("export", help="export one passage's TPG")
    s.add_argument("passage_id")
    s.add_argument("--fmt", choices=["graphson", "cypher"], default="graphson")

    s = sub.add_parser("serve", help="start API + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8742)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "serve":
        import os
        os.environ.setdefault("TPG_DB", args.db)
        os.environ.setdefault("TPG_DOMAIN", args.domain)
        os.environ["TPG_HOST"] = args.host
        os.environ["TPG_PORT"] = str(args.port)
        from tpg_app.server import main as serve_main
        serve_main()
        return

    from tpg_app.engine import TPGEngine
    engine = TPGEngine(args.db, domain=args.domain)

    if args.cmd == "ingest":
        src = Path(args.source)
        if src.is_dir():
            results = engine.ingest_directory(src, overwrite=args.overwrite)
            ok = [r for r in results if "error" not in r]
            for r in results:
                if "error" in r:
                    print(f"  ✗ {r['name']}: {r['error']}")
                else:
                    flag = " (skipped, already indexed)" if r.get("skipped") else ""
                    print(f"  ✓ {r['name']}: {r['passages']} passages, "
                          f"{r['entities']} entities{flag}")
            print(f"\n{len(ok)}/{len(results)} sources ingested.")
        else:
            r = engine.ingest(args.source, name=args.name,
                              overwrite=args.overwrite)
            if r.get("skipped"):
                print(f"'{r['name']}' already indexed — pass --overwrite to redo.")
            else:
                print(f"✓ {r['name']}: {r['passages']} passages, "
                      f"{r['entities']} entities, {r['relations']} relations")
        print(engine.stats())

    elif args.cmd == "query":
        hits = engine.query(args.question, top_k=args.top_k)
        if not hits:
            print("No matches.")
        for h in hits:
            loc = f"page {h.page}" if h.page else (h.section or "")
            print(f"\n[{h.score}] {h.doc_name} {loc}")
            if h.matched_entities:
                print(f"    matched: {', '.join(h.matched_entities[:6])}")
            print(f"    {h.text[:400]}")

    elif args.cmd == "ask":
        try:
            result = engine.ask(args.question, top_k=args.top_k)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        print(result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            loc = f"page {s['page']}" if s["page"] else (s["section"] or "")
            print(f"  • {s['doc']} {loc} (score {s['score']})")

    elif args.cmd == "entities":
        for e in engine.top_entities(limit=args.limit, etype=args.etype):
            print(f"  {e['n']:>4} × {e['display']}  [{e['etype'] or '—'}]")

    elif args.cmd == "explore":
        nb = engine.neighborhood(args.entity)
        print(f"Relations of '{args.entity}':")
        for r in nb["relations"] or []:
            print(f"  {r['src']} —{r['rel']}→ {r['dst']}  (×{r['n']})")
        print("Co-mentions:")
        for c in (nb["co_mentions"] or [])[:20]:
            print(f"  {c['display']} [{c['etype'] or '—'}] ×{c['n']}")

    elif args.cmd == "path":
        path = engine.find_path(args.source, args.target,
                                max_hops=args.max_hops)
        if not path:
            print("No path found.")
        for hop in path:
            print(f"  {hop['from']} —{hop['rel']}→ {hop['to']}")

    elif args.cmd == "stats":
        print(json.dumps(engine.stats(), indent=2))

    elif args.cmd == "export":
        result = engine.passage_graph(args.passage_id, fmt=args.fmt)
        if result is None:
            print(f"No stored graph for '{args.passage_id}'")
            sys.exit(1)
        print(result)

    engine.close()


if __name__ == "__main__":
    main()
