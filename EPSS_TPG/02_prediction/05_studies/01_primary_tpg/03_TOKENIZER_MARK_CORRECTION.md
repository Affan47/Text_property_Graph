# Resume After the Nonspacing-Mark Error

## What failed

The SMP text for CVE-2021-30769 contains U+06E7, named ARABIC SMALL HIGH YEH,
at chunk offsets 824 through 825 (end-exclusive). Its Unicode category is
`Mn`, a nonspacing mark. The pinned SecBERT normalizer reduces this isolated
character to an empty string. It produces no standalone token IDs, and no
contextual token offset overlaps its span. There is consequently no
source-aligned embedding to attach to this node.

The previous coverage rule handled specified control characters and U+FFFD,
but not this nonspacing mark. It classified the mark as ordinary text and
raised the missing-word-embedding error. That was a coverage-policy gap, not
an out-of-memory error or evidence that an ordinary word was truncated.

The surrounding stored SMP text contains many replacement characters and
garbled fragments. The correction does **not** repair that source quality
problem or claim that the text is meaningful. It preserves it for inspection.

## Narrow correction

The separate v2 policy permits an absent token embedding only when all of
these additional checks pass:

1. The token consists entirely of nonspacing (`Mn`) characters.
2. Its text matches its original source span exactly.
3. The pinned tokenizer's normalizer removes the whole token.
4. Tokenizing that token produces no token IDs.
5. The containing sentence matches the original source text, and tokenizing
   that sentence without truncation shows no offset overlapping the span.

The node, text, offsets, node type and graph edges are retained. The embedding
is an explicit zero vector with embedding-present mask zero; it is excluded
from token-embedding pooling. Its codepoints, Unicode names, span and source
context are recorded in the coverage audit.

Ordinary words, accented words and mixed letter/mark tokens are not exempt.
A missing embedding for a token that has contextual tokenizer coverage still
fails. All original graph, edge, mask and tensor validation remains active.
This is not a blanket exemption for non-English text.

## Verified locally

- The actual failing 1,200-character chunk now passes with unchanged text.
  Its graph has 676 nodes, 574 embedded nodes and one newly audited missing
  nonspacing token, node 321. Other unembedded nodes follow the existing
  structural, whitespace, replacement and control-character rules.
- The unchanged original validator still rejects the corrected bundle,
  reproducing the policy difference rather than silently redefining v1.
- Ten focused new tests cover token normalization, ordinary/accented word
  failures, exact spans, nonzero-vector corruption, contextual coverage,
  unchanged legacy behavior and bounded queue failure. All 41 combined
  CS01/PS01 tests pass.
- The v2 smoke reuses the original 20 completed chunks. No encoder rerun is
  needed for those chunks. No full expansion or training was launched.

The real failing-chunk evidence is saved under:

```text
EPSS_TPG/01_tpg/05_workspace/primary_day60_v2/checks/failing_chunk/
```

`analysis.txt` and `analysis.json` explain the result. `checked_statistics.json`
and the receipt record the exact missing-token evidence. Graphs and tensors
remain inspectable in their normal subfolders.

## Preserve the interrupted work

The original v1 implementation, exported data, completed smoke results and
unfinished full cache remain unchanged. There were **11,266 receipt files**
in the unfinished full cache when inspected. This is a count of saved receipts,
not a claim that all have been independently reverified in this repair turn.

The new builder checks each matching v1 receipt, its text/processing identity,
graph files and tensors, then copies valid completed chunks into the separate
v2 cache. Reused receipts still pass their original coverage policy. Missing
chunks use the new policy. No successful chunk needs re-encoding solely
because of this repair. Checks and copying can still take time and disk space.

The new process pool queues at most twice the worker count. A failure cancels
pending work and waits only for already-running bounded tasks; it does not
queue the entire remaining corpus. Every completed chunk has its own receipt
so it can be recovered after interruption.

## Commands

Do **not** repeat the old `02_prepare_primary_study.py --stage cache` build
command for the failing full run. Use the new versioned entry point.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies -p 'test_primary_mark_coverage.py'
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/04_prepare_primary_cache_v2.py --check-failure
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/04_prepare_primary_cache_v2.py --scale full --workers 4
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/04_prepare_primary_cache_v2.py --scale full --verify-only
```

The first full v2 invocation automatically reuses verified v1 completions;
it does not need `--resume`. If the **v2** build is interrupted, continue with:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/04_prepare_primary_cache_v2.py --scale full --workers 4 --resume
```

The output is `EPSS_TPG/01_tpg/05_workspace/primary_day60_v2/full`.
`coverage_audit.json` lists any newly encountered qualifying nonspacing tokens.
The underlying graph/tensor layout is unchanged; the cache identity and
coverage policy are versioned. A later full-training adapter must explicitly
accept this v2 cache rather than pretending it is the old v1 artifact.
