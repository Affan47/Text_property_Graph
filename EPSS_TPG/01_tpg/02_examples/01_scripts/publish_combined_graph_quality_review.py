"""Publish the checked graph-quality review; no model or graph-cache changes."""
import argparse
from collections import Counter
import io
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package'):
    sys.path.insert(0,str(ROOT/rel))
from tpg_combined_v2 import common as c
from tpg_combined_quality_v1 import audit, inspection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    output=audit.OUTPUT/'review'
    if a.verify_only:
        for dataset in ('reddit','previous'):
            for stage in ('sweep','inspection','annotation_preview','cis_context_preview','report'):
                audit.verify_output(audit.OUTPUT/dataset/stage)
        audit.verify_output(output);return
    c.require(not output.exists(),'Review exists; use --verify-only')
    studies={ds:{stage:c.read(audit.OUTPUT/ds/stage/'REPORT.json') for stage in
        ('sweep','inspection','annotation_preview','cis_context_preview')} for ds in ('reddit','previous')}
    output.mkdir(parents=True)
    # Persist a real standard-library test run rather than hand-writing a pass claim.
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'01_tpg/08_tests'),pattern='test_combined_quality.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    audit.save_text(output/'tests.txt',log.getvalue())
    c.require(result.wasSuccessful(),'Quality tests failed; see tests.txt')
    rows=[];counts=Counter();repairs=Counter();examples=[]
    for ds,s in studies.items():
        counts.update({key:s['sweep']['totals'][key] for key in ('graphs','nodes','edges')})
        for metric,selection in s['sweep']['selection'].items():
            for label,row in selection.items():rows.append({'dataset':ds,'selection':metric+'/'+label,**row})
        seen=set()
        for stage in ('annotation_preview','cis_context_preview'):
            for row in s[stage]['graphs']:
                if row['key'] not in seen:repairs.update(row['rule_counts']);seen.add(row['key'])
        for case in s['inspection']['graphs']:
            if 'by_nodes/largest' not in case['selection']:continue
            g=c.read(case['graph_path']);nodes={n['id']:n for n in g['vertices']}
            pv=Counter(nodes[e['outV']]['properties']['DOC_IDX'] for e in g['edges'] if e['label']=='SEC_HAS_VERSION')
            cases=case['document_quality']['documents']
            for doc in cases:doc['product_version_edges']=pv[doc['document_index']]
            examples.append({'dataset':ds,'cve':case['cve'],'documents':cases,
                'near_repeat_pairs':case['document_quality']['near_repeat_pairs_jaccard_at_least_0_9']})
    storage={'dense_float32_node_embeddings_gib':counts['nodes']*768*4/2**30,
             'dense_float32_edge_features_gib':counts['edges']*88*4/2**30,
             'note':'Uncompressed numeric payload estimates only; exclude indices, Python overhead, optimizer state and intermediate arrays.'}
    text=['COMBINED-TEXT TPG QUALITY REVIEW', '',
        'Decision: agree with the main external review, with important qualifications.',
        f'Checked {counts["graphs"]:,} original graph objects, {counts["nodes"]:,} nodes and {counts["edges"]:,} edges.',
        'No violations of the implemented structural checks were found. This is not a guarantee that every linguistic or security label is true.',
        'All original graph and receipt bytes are preserved. No embeddings or training were run.',
        '', '1. Six representative graphs by NODE COUNT',
        'Dataset | Example | CVE | Text characters | Nodes | Edges']
    for r in rows:
        if r['selection'].startswith('by_nodes/'):
            text.append(f'{r["dataset"]} | {r["selection"].split("/")[1]} | {r["cve"]} | {r["characters"]:,} | {r["nodes"]:,} | {r["edges"]:,}')
    text+=['','The shortest text is not always the graph with the fewest nodes. Shortest/median/longest BY TEXT LENGTH:']
    for r in rows:
        if r['selection'].startswith('by_characters/'):
            text.append(f'{r["dataset"]} | {r["selection"].split("/")[1]} | {r["cve"]} | {r["characters"]:,} | {r["nodes"]:,} | {r["edges"]:,}')
    text+=['','2. What text correctness means',
        'Selected graphs were compared with actual raw description, post and summary fields; a fresh spaCy parse reproduced tokens, lemmas, parts of speech, dependencies, sentences and noun phrases.',
        'Spanned node text matches the source. Discontinuous CLAUSE/VERB_PHRASE text is reconstructed from its declared member tokens, not every word inside its bounding span.',
        'TOPIC nodes are derived lemmas and are intentionally not literal source spans. A literal-text equality rule would be incorrect for them.',
        'Every graph has one weakly connected component, no isolated nodes, valid endpoints and complete reading-order chains.',
        'Only marked reading-order edges cross source-document boundaries. A few entities span sentences inside one document under explicit flags; they are not falsely asserted as sentence-local relations.',
        'The six ablation views matched their preserved screened input documents for every development CVE.',
        '', '3. The labels can still be wrong',
        'Example: the old dataset includes the post ",redteam". It is only eight characters. Its graph faithfully stores that text, but spaCy treats redteam as a verb; this is not evidence of a real security action.',
        'The smallest Reddit graph has 117 characters but only two TOKEN nodes: "Security" plus a long URL. It is a short link-bearing post, not the scraped webpage summary.',
        'The actual shortest Reddit text is "CVE-2022-1771 tweet" (19 characters). It provides very little descriptive content.',
        'In the median Reddit description, spaCy treating "insecure" in "due to insecure handling" as a verb is a parser issue. We do not add a blanket rule declaring that word can never be a verb.',
        '', 'Pattern | Reddit occurrences | Older occurrences']
    for rule in ('safeguard_version_candidate','code_word_inside_vulnerability_phrase','capitalized_system_code_candidate',
                 'bare_by_agent','generic_type_on_security_identifier','identifier_topic'):
        text.append(f'{rule} | {studies["reddit"]["sweep"]["quality_pattern_counts"].get(rule,0):,} | {studies["previous"]["sweep"]["quality_pattern_counts"].get(rule,0):,}')
    text+=['Counts above are occurrences across graph objects, including related ablation views; they are not unique CVEs or an independently annotated error rate.',
        'The older dataset has two Safeguard keyword matches, but both refer to the genuine product "One Identity Safeguard for Privileged Passwords" before 7.5.2 (CVE-2024-45488). These were retained.',
        'The free/buffer count requires overlap with an extracted vulnerability phrase. It is deliberately narrower than counting every occurrence of those words.',
        'CVE/CWE topic counts include full identifiers and fragments; the external report used a narrower fragment count.',
        'Security edges anchored to a different CVE are not automatically errors. They remain marked as other; no false target entity is inserted into descriptions that omit the ID.',
        '', '4. Why the largest graphs need source-quality review']
    for case in examples:
        text.append(f'{case["dataset"]}, {case["cve"]}:')
        for doc in case['documents']:
            text.append(f'  Source {doc["source"]}, document {doc["document_index"]}: {doc["characters"]:,} characters; {doc["tokens"]:,} tokens; {doc["product_version_edges"]} product-version edges; {doc["target_id_mentions"]} literal target-ID mentions.')
        text.append(f'  Near-repeat document pairs with 5-word-shingle Jaccard >= 0.90: {len(case["near_repeat_pairs"])}.')
    text+=['A faithful graph of a repeated/off-target page is still a poor candidate input. Length or a rare ID mention alone does not establish irrelevance.',
        'Do not silently truncate, delete posts, invent missing summaries, or merge distinct CVEs. Freeze a separate documented source policy before changing those inputs.',
        '', '5. Implemented next step: a versioned annotation-only preview',
        'The original caches remain frozen. Separate previews fix explicit passive-agent objects, supported CIS control/product confusion, plain system nouns, free/buffer vulnerability terms, generic ID-label conflicts, and ID topics.',
        'Real CIS advisory examples remove only candidates satisfying the bounded evidence rule. Cases without enough local context remain for review; no blanket Safeguard blacklist is used.',
        'Genuine System()/free() calls and genuine Safeguard product releases are negative tests and remain valid.',
        'Every change has original node IDs and context. All source text, TOKEN nodes, parser dependencies and reading order remain identical.',
        f'{result.testsRun} tests passed, including 19 deliberate corruption cases for the independent structural verifier.',
        'The annotation preview, including its graph-to-tensor mapping, was replayed deterministically. It is not yet a replacement production cache.',
        '', '6. Embeddings and training are not verified yet',
        f'Dense float32 embeddings for these original nodes alone would contain about {storage["dense_float32_node_embeddings_gib"]:.1f} GiB.',
        f'The 88-value dense edge features would add about {storage["dense_float32_edge_features_gib"]:.1f} GiB, before indices and other overhead.',
        'These are calculated payload estimates, not files already generated. Recheck disk capacity and cache reuse before launching full embedding generation.',
        'Use the confirmed graph/member-token spans to generate frozen embeddings only after the graph policy is approved. Then verify token coverage and tensor alignment with actual vectors.',
        'Review source repetition and relevance separately; a larger TPG does not guarantee a better GNN prediction.',
        'Reddit validation still has no accepted webpage summaries. This remains a limitation for the U/DU/DMU comparisons.',
        '', '7. Where to inspect the evidence',
        'reddit/ and previous/ each contain sweep/, inspection/, annotation_preview/, cis_context_preview/ and report/.',
        'inspection/<key>/nodes.csv lists every selected node and its text; edges.csv lists every edge and its endpoint IDs with clearly marked text excerpts.',
        'The original graph paths are recorded in inspection.json. Original graphs remain compressed JSON (.json.gz).',
        'REPORT.json, graphs.csv and the SHA-256 locks support machine-readable inspection. No target-score optimization or test-cohort inspection was performed.']
    c.create(output/'REPORT.json',{'studies':studies,'examples':examples,'totals':counts,'preview_rule_counts':repairs,
        'storage_estimate':storage,'unit_tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}})
    audit.save_text(output/'REPORT.txt','\n'.join(text)+'\n')
    audit.write_csv(output/'size_examples.csv',rows,list(rows[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picked=[r for r in rows if r['selection'].startswith('by_nodes/')]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    bars=ax.barh([r['dataset']+' '+r['selection'].split('/')[1]+'\n'+r['cve'] for r in picked],
                [r['nodes'] for r in picked],color=['#2e7d6d']*3+['#b24b57']*3)
    ax.set_xscale('log');ax.set_xlabel('Number of nodes (logarithmic scale)')
    ax.set_ylabel('Dataset and example selected by node count');ax.set_title('Small, middle and large TPG examples')
    ax.bar_label(bars,labels=[f'{r["nodes"]:,}' for r in picked],padding=4)
    ax.set_xlim(right=max(r['nodes'] for r in picked)*4)
    fig.savefig(output/'graph_sizes.png',dpi=170);plt.close(fig)
    inputs={}
    for ds in studies:
        for stage in studies[ds]:
            path=audit.OUTPUT/ds/stage/'artifact_lock.json';inputs[str(path)]=c.sha(path)
    for f in (ROOT/'01_tpg/10_extensions/tpg_combined_quality_v1').glob('*.py'):inputs[str(f)]=c.sha(f)
    for f in (Path(__file__),ROOT/'01_tpg/08_tests/test_combined_quality.py'):inputs[str(f)]=c.sha(f)
    inspection.finish(output,inputs)
    print(f'Combined review saved: {output/"REPORT.txt"}')


if __name__=='__main__':main()
