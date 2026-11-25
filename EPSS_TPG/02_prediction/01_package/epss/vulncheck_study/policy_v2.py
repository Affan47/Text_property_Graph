"""Narrow source policy v2; engineering-reviewed cues, not a learned truth verifier."""
import re

VERSION = 'vulncheck_source_policy_v2'
DASHES = str.maketrans({chr(c):'-' for c in (0x2010,0x2011,0x2012,0x2013,0x2014,0x2212)})
CATALOGUE = {
    'vulncheck_mention': re.compile(r'\bvuln\s*check\b',re.I),
    'kev_catalogue_mention': re.compile(r'\b(?:vc[\s_-]*)?kev\b',re.I),
    'known_exploited_catalogue_phrase': re.compile(r'\bknown[\s-]+exploited[\s-]+vulnerabilit(?:y|ies)\b',re.I),
    'cisa_catalogue_url': re.compile(r'cisa\.gov/[^\s]*known-exploited',re.I),
}
FUNCTION = re.compile(r'\bdef\s+(vulncheck)\s*\(',re.I)
EPSS = re.compile(r'\bEPSS\b.{0,80}\d|\d.{0,40}\bEPSS\b',re.I|re.S)
CVE = re.compile(r'\bCVE-\d{4}-\d{4,}\b',re.I)
EXPLICIT_ID = re.compile(r'\bCVE\s+(?:identifier|ID)\s*(?:is\b|[:=])\s*[`*\"\s]*(CVE-\d{4}-\d{4,})\b',re.I)


def inspect(text, cve, source):
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Policy expects a nonempty source document')
    # One-character dash substitution preserves character offsets for audit spans.
    normalized=text.translate(DASHES)
    function_spans={m.span(1) for m in FUNCTION.finditer(normalized)}
    cues=[]; ignored=[]
    for name,pattern in CATALOGUE.items():
        for m in pattern.finditer(normalized):
            if name=='vulncheck_mention' and m.span() in function_spans:
                ignored.append({'reason':'python_function_definition_identifier','start':m.start(),'end':m.end()})
            else:
                cues.append({'reason':name,'start':m.start(),'end':m.end()})
    if m:=EPSS.search(normalized):
        cues.append({'reason':'numeric_epss_disclosure_candidate','start':m.start(),'end':m.end()})
    mentioned=sorted(set(m.upper() for m in CVE.findall(normalized)))
    claims=[m for m in EXPLICIT_ID.finditer(normalized)]
    if source in ('github_summary','non_github_summary') and cve not in mentioned:
        for m in claims:
            if m.group(1).upper()!=cve:
                cues.append({'reason':'summary_explicit_other_target','start':m.start(),'end':m.end()})
    reasons=sorted({v['reason'] for v in cues})
    return {'policy':VERSION,'decision':'quarantine' if reasons else 'accept',
            'reasons':reasons,'evidence_spans':sorted(cues,key=lambda r:(r['start'],r['reason'])),
            'ignored_spans':ignored,'other_cve_ids':[c for c in mentioned if c!=cve],
            'target_cve_mentioned':cve in mentioned,
            'review_warning':'Accepted means passed these rules, not verified true/relevant or historically available.'}


def review_cases():
    return [
        {'case':'python_identifier','text':'def vulncheck(ipaddress, port): return True',
         'cve':'CVE-2023-23333','source':'smp','expected':'accept',
         'rationale':'A function definition alone does not assert catalogue listing.'},
        {'case':'identifier_plus_catalogue','text':'def vulncheck(x): pass\nListed in VulnCheck KEV.',
         'cve':'CVE-2023-23333','source':'smp','expected':'quarantine',
         'rationale':'The exception is span-local and must not hide a separate catalogue statement.'},
        {'case':'explicit_listing','text':'This vulnerability is listed in the VulnCheck KEV database.',
         'cve':'CVE-2024-9644','source':'non_github_summary','expected':'quarantine',
         'rationale':'Direct catalogue disclosure.'},
        {'case':'explicit_absence','text':'The CISA KEV list does not flag it.',
         'cve':'CVE-2017-4965','source':'non_github_summary','expected':'quarantine',
         'rationale':'Negative catalogue assertions can also reveal outcomes; CISA is distinct from VulnCheck.'},
        {'case':'wrong_explicit_summary_id','text':'CVE identifier is ```CVE-2023-12345```',
         'cve':'CVE-2019-1367','source':'non_github_summary','expected':'quarantine',
         'rationale':'An explicit summary identity claim differs from its assigned CVE and does not mention the target.'},
        {'case':'legitimate_distinction','text':'A different issue than CVE-2006-2369.',
         'cve':'CVE-2006-2450','source':'description','expected':'accept',
         'rationale':'Do not discard all other-CVE references.'},
    ]
