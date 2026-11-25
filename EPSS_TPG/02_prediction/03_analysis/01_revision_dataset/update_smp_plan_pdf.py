"""Build the editable current plan with wrapped tables and verified SMP findings.

Preserve the supplied PDF as a baseline, but reflow its content rather than
copying the original overflowing tables. Raw datasets are never modified.
"""
import json
import re
import shutil
from datetime import date, timedelta

import pymupdf as fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verify_smp_dates import OUT, ROOT, SOURCE, digest
from smp_embedded_dates import embedded_report
from target_plan_layout import build_plan, verify_layout
from analyze_dated_smp_epss import SOURCE as DATED_SOURCE, report as dated_report

PAPERS = ROOT / '00_documentation/06_papers'
PDF = PAPERS / '02_pdf/epss_target_verification.pdf'
BUILD = PAPERS / '03_build_files/epss_target_verification'
MANUSCRIPT = PAPERS / '01_sources/epss_target_verification.md'
EXAMPLE_IDS = ('CVE-2025-4039:smp:0', 'CVE-2025-43928:smp:2', 'CVE-2025-2907:smp:1')


def draw_examples(raw, audit):
    selected = {r['association_id']: r for r in audit}
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.1), layout='constrained')
    for ax, ident in zip(axes, EXAMPLE_IDS):
        row = selected[ident]
        record = raw[row['cve']]
        pub = date.fromisoformat(record['published_date'])
        ys = [record['epss_scores'].get((pub + timedelta(days=k)).isoformat()) for k in range(61)]
        ax.plot(range(61), [float('nan') if v is None else v for v in ys], color='#196c83', linewidth=1.6, label='Stored EPSS')
        ax.axvline(row['embedded_date_review']['day_after_nvd_publication'], color='#a6394c', linewidth=1.6, label='Date written in text')
        ax.axvline(30, color='#707070', linestyle='--', linewidth=1, label='Day-30 prediction cutoff')
        ax.set_title(row['cve'] + ' / ' + row['platform'].title() + ' / text date ' + row['embedded_date_review']['alignment_date'], fontsize=10, loc='left')
        ax.set(xlim=(-1, 61), ylabel='EPSS score')
        ax.grid(alpha=.18)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc='upper right', fontsize=7, frameon=False)
    axes[-1].set_xlabel('Days since NVD publication (Day 0 = publication date)')
    target = OUT / 'figures/smp_body_dates_epss_60d.png'
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=180)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)
    return target


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    baseline = BUILD / 'provided_before_date_review.pdf'
    previous_check = OUT / 'current_plan_pdf_verification.json'
    allowed = {digest(baseline)}
    if previous_check.exists():
        allowed.add(json.loads(previous_check.read_text())['output_pdf_sha256'])
    assert digest(PDF) in allowed, 'The current PDF was edited again; review those changes before rebuilding'
    backup = BUILD / 'before_layout_repair.pdf'
    if not backup.exists():
        shutil.copyfile(PDF, backup)

    raw = {r['cve']: r for r in json.loads(SOURCE.read_text())}
    audit = json.loads((OUT / 'smp_candidate_event_audit.json').read_text())
    summary = json.loads((OUT / 'smp_date_verification.json').read_text())
    assert digest(SOURCE) == summary['source_hashes'][str(SOURCE)]
    dated = json.loads((OUT / 'smp_dated_epss_analysis.json').read_text())
    assert digest(DATED_SOURCE) == dated['source_sha256']
    assert (OUT / 'SMP_DATED_EPSS_ANALYSIS.md').read_text() == dated_report(dated)
    by_id = {r['association_id']: r for r in audit}
    for ident, day, at, after in ((EXAMPLE_IDS[0], 0, None, .0004),
                                (EXAMPLE_IDS[1], 7, .00054, .00054),
                                (EXAMPLE_IDS[2], 31, .03356, .03642)):
        e = by_id[ident]['embedded_date_review']
        assert e['day_after_nvd_publication'] == day
        assert e['alignment']['60']['offsets']['0']['score'] == at
        assert e['alignment']['60']['offsets']['7']['score'] == after
    embedded = embedded_report(summary['embedded_date_review'])
    markdown = MANUSCRIPT.read_text().replace('{{smp_review}}', embedded).replace(
        '{{dated_smp_review}}', (OUT / 'SMP_DATED_EPSS_ANALYSIS.md').read_text())
    (BUILD / 'embedded_date_section.md').write_text(embedded)
    (BUILD / 'complete_manuscript.md').write_text(markdown)
    figure = draw_examples(raw, audit)
    temporary = BUILD / 'updated.pdf'
    boxes = build_plan(markdown, figure, temporary)
    with fitz.open(temporary) as result:
        text = '\n'.join(p.get_text() for p in result)
        normalized = re.sub(r'\s+', ' ', text)
        assert 'The current SMP date represents extraction/scraping time' not in text
        for value in ('2,658', '485', '742', 'CVE-2025-4039', 'CVE-2025-43928',
                      'CVE-2025-2907', 'Figure 8.', '11. VulnCheck', '12. Implementation',
                      '9,699', '9,574', '10.12 What This Update Resolves'):
            assert value in normalized, value
        layout = verify_layout(result, boxes)
        checks = dict(layout, pages=len(result), full_document_reflowed=True,
                      prior_date_findings_preserved=True, new_dated_analysis_included=True,
                      unsupported_scraping_assertion_removed=True,
                      input_pdf_sha256=digest(backup), source_sha256=digest(SOURCE),
                      dated_source_sha256=digest(DATED_SOURCE),
                      dated_analysis_sha256=digest(OUT / 'smp_dated_epss_analysis.json'),
                      manuscript_sha256=digest(MANUSCRIPT),
                      output_pdf_sha256=digest(temporary), model_training_performed=False)
    shutil.copyfile(temporary, PDF)
    for suffix in ('json', 'txt'):
        (OUT / f'current_plan_pdf_verification.{suffix}').write_text(json.dumps(checks, indent=2)+'\n')
    (BUILD / 'layout_boxes.json').write_text(json.dumps(boxes, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
