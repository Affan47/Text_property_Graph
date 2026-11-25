"""Check real CIS advisory false positives and genuine Safeguard counterexamples."""
from pathlib import Path
from tqdm import tqdm
from tpg_combined_v2 import common as c
from . import audit, inspection, reviewed_correction as repair


def run(dataset, verify=False):
    output=audit.OUTPUT/dataset/'cis_context_preview'
    if verify: audit.verify_output(output)
    else: c.require(not output.exists(), 'Existing supplemental preview; use --verify-only')
    directory=c.directory(dataset,'full');plan=c.load_plan(directory)
    sweep=c.read(audit.OUTPUT/dataset/'sweep/REPORT.json')
    keys={r['key'] for r in sweep['quality_examples'].get('safeguard_version_candidate',[])}
    if dataset=='reddit':
        # Training example actually inspected while reproducing the report.
        cid='CVE-2022-1471'
        key=plan['index'][cid]['views']['DMU']['object_key']
        c.require(plan['entries'][key]['split']=='train','Supplemental training example changed role')
        keys.add(key)
    rows, inputs=[],{}
    for key in tqdm(sorted(keys),desc=f'Check CIS context versus genuine product names: {dataset}',unit='graph'):
        path=directory/'objects'/key/'graph.json.gz';old=c.read(path);before=c.sha(path)
        new=repair.correct(old);repair.validate_preview(plan['entries'][key],old,new)
        c.require(repair.correct(new)==new,'Supplemental correction is not idempotent')
        old_nodes={n['id']:n for n in old['vertices']}
        candidates=[e for e in old['edges'] if e['label']=='SEC_HAS_VERSION' and 'safeguard' in old_nodes[e['outV']]['properties']['TEXT'].casefold()]
        kept={e['id'] for e in new['edges']}
        rows.append({'cve':plan['entries'][key]['cve'],'key':key,'split':plan['entries'][key]['split'],
            'candidate_product_version_edges':len(candidates),'candidate_edges_retained':sum(e['id'] in kept for e in candidates),
            'candidate_edges_removed':sum(e['id'] not in kept for e in candidates),
            'products':sorted({old_nodes[e['outV']]['properties']['TEXT'] for e in candidates}),
            'rule_counts':new['metadata']['annotation_repair_preview']['rule_counts']})
        if verify:c.require(c.read(output/'graphs'/key/'graph.json.gz')==new,'Supplemental graph changed')
        else:c.create(output/'graphs'/key/'graph.json.gz',new)
        c.require(c.sha(path)==before,'Original graph changed');inputs[str(path)]=before
    payload={'dataset':dataset,'graphs':rows,'policy':repair.POLICY,'original_graphs_changed':0,
        'interpretation':'CIS numbered controls and One Identity Safeguard product releases must not be conflated.'}
    if verify:c.require(c.read(output/'REPORT.json')==payload,'Supplemental report changed')
    else:
        c.create(output/'REPORT.json',payload)
        audit.save_text(output/'REPORT.txt','Real Safeguard context examples\n'+__import__('json').dumps(payload,indent=2)+'\n')
        for module in (repair,repair.base):inputs[str(Path(module.__file__))]=c.sha(module.__file__)
        inputs[str(Path(__file__))]=c.sha(__file__)
        inspection.finish(output,inputs)
    print(f'CIS context preview {"verified" if verify else "saved"}: {output}')
