"""VK07 source-aware export and class-conditional coverage, with frozen VK02 policy."""
import hashlib
from pathlib import Path

from . import adapter_v2 as data, development_v1 as dev, graph_cache_v1 as graph

CONFIG = data.ROOT / '02_prediction/04_configs/02_vulncheck/06_multisource_ablation_v1.json'
CLI = data.SCRIPTS / '12_run_multisource_ablation.py'
SOURCES = ('description', 'github_summary', 'non_github_summary', 'smp')


def config():
    p = data.read(CONFIG)
    data.require(p['source_order'] == list(SOURCES) and list(p['variants']) == ['D','DGN','DM','DGNM'], 'Wrong source protocol')
    return p


def pins():
    files = [CONFIG, CLI, data.SCRIPTS/'test_vulncheck_multisource_v1.py']
    files += [Path(__file__).with_name('multisource_'+s+'_v1.py') for s in ('data','cache','model','training')]
    return {**graph.pins(), **{str(p):data.sha(p) for p in files}}


def canonical_documents(row, raw, source_fields):
    """Recover transport IDs from exact source/text/platform identity, not fuzzy text matching."""
    data.require(set(row)=={'cve','split','documents','source_mask','structured'} and row['split'] in ('train','validation'),
                 'Unexpected fields or forbidden CVE role')
    mapping = {}
    for source in SOURCES:
        value = raw.get(source_fields[source])
        items = [(post.get('post'),post.get('platform')) for post in value or []] if source == 'smp' else [(value,None)]
        for text, platform in items:
            if not isinstance(text,str) or not text.strip():
                continue
            key = source+':'+data.digest({'source':source,'text':text,'platform':platform})
            canonical = source+':'+hashlib.sha256(((platform or '')+'\0'+text).encode()).hexdigest()
            mapping[key] = (canonical, text, source)
    docs, audit, seen = [], [], set()
    for doc in row['documents']:
        data.require(doc['document_id'] in mapping, 'Accepted document identity missing from original source')
        canonical, text, source = mapping[doc['document_id']]
        data.require(text == doc['text'] and source == doc['source_text_type'], 'Source text changed')
        data.require(canonical not in seen, 'Ambiguous canonical identity; refuse silent deduplication')
        seen.add(canonical)
        docs.append({**doc, 'document_id':canonical})
        audit.append({'cve':row['cve'], 'accepted_document_id':doc['document_id'], 'canonical_document_id':canonical})
    data.require(sum(d['source_text_type']=='description' for d in docs)==1, 'Need exactly one accepted description')
    return {**row,'documents':docs}, audit


def prepare(scale, verify=False):
    p = config()
    _, source, rows, labels, state, export, dependencies = dev.load()
    ids = graph.select(rows,labels,scale)
    selected = set(ids['train']+ids['validation'])
    rows = [r for r in rows if r['cve'] in selected]
    labels = [r for r in labels if r['cve'] in selected]
    decisions = [r for r in data.read(source/'decisions.json') if r['cve'] in selected]
    audit = dev.audit_sources(rows,labels,decisions,SOURCES)
    overlap = dev.overlap_audit(rows)
    raw_path = data.ROOT.parent / export['config']['source']
    data.require(data.sha(raw_path)==export['config']['source_sha256'], 'Raw source changed')
    raw = {r['cve']:r for r in data.read(raw_path) if r['cve'] in selected}
    out, mapping = [], []
    for row in dev.tqdm(rows, desc='Map exact accepted source identities', unit='CVE'):
        converted, mapped = canonical_documents(row,raw[row['cve']],export['config']['source_fields'])
        out.append(converted); mapping += mapped
    identity = {'schema':p['schema'], 'scale':scale, 'config':p, 'ids':ids,
                'pins':{**dependencies, **pins(), str(raw_path):data.sha(raw_path)},
                'training':False, 'test_export':False}
    coverage = {split:{s:sum(r['source_mask'][i] for r in out if r['split']==split) for i,s in enumerate(SOURCES)} for split in ids}
    summary = {'schema':p['schema'], 'scale':scale, 'cves':{s:len(v) for s,v in ids.items()},
               'accepted_source_coverage':coverage, 'accepted_documents':sum(len(r['documents']) for r in out),
               'new_full_fits':18, 'reused_full_D_fits':6, 'structured_dimension':len(state['feature_names']),
               'interpretation':p['interpretation'], 'training':False, 'test_export':False}
    lines = ['VK07 source preparation',data.dump(summary),'',
             'Split | Listed label | Source | CVEs in class | Source before filtering | Source after filtering | CVEs losing source']
    lines += [' | '.join(str(r[k]) for k in ('split','listed','source','cves_in_class','cves_with_source_before','cves_with_source_after','cves_losing_entire_source')) for r in audit['by_source_and_class']]
    lines += ['', 'Inputs and labels remain separate; coverage/disclosure decisions are audit-only, not predictors.',
              'No CVEs dropped for absent summaries/posts. GitHub and non-GitHub remain separate sources within DGN.',
              'Structured features stay the same 29 columns; no EPSS, platform counts or catalogue fields are added.',
              'Filtering can differ by class; it is not proof that remaining inputs are leakage-free.',
              'No historical availability or causal social-media effect is claimed. Shared wording does not merge CVEs.']
    data.publish(data.ROOT/p['data_root']/scale, {'inputs.json':{'source_order':list(SOURCES),'rows':out},
        'labels.json':labels,'structured_encoder.json':state,'document_identity_map.json':mapping,
        'source_coverage.json':audit,'source_coverage.csv':dev.csv_text(audit['by_source_and_class']),
        'shared_text.json':overlap,'summary.json':summary,'summary.txt':'\n'.join(lines)+'\n',
        'run_identity.json':identity},identity['pins'],verify)
    print(f'VK07 {scale} source export '+('reproduced.' if verify else 'saved.')+' No graphs, training or test export.')


def load(scale):
    p = config(); directory = data.ROOT/p['data_root']/scale
    data.check_lock(directory)
    identity = data.read(directory/'run_identity.json')
    data.require(identity['config']==p and identity['scale']==scale, 'Preparation identity differs')
    data.check_pins(identity['pins'])
    rows = data.read(directory/'inputs.json')['rows']; labels = data.read(directory/'labels.json')
    data.require([(r['cve'],r['split']) for r in rows]==[(r['cve'],r['split']) for r in labels], 'Label/input join differs')
    data.require(set(identity['ids'])=={'train','validation'}, 'Unexpected role')
    for split,ids in identity['ids'].items():
        data.require(sorted(r['cve'] for r in rows if r['split']==split)==ids,'Cohort differs')
    return p,directory,rows,labels,identity
