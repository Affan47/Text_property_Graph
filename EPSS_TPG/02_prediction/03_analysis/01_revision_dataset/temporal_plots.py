"""Descriptive temporal figures; plotted values and cohort definitions are saved."""
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_temporal_figures(folder, rows, report):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.right': False, 'axes.spines.top': False,
                         'savefig.facecolor': 'white'})
    figures = []
    listed = [r for r in rows if r['in_vckev'] and r['delta_days_exact'] is not None]
    signed = np.array([r['delta_days_exact'] for r in listed], dtype=float)
    positive_rows = [r for r in listed if r['delta_days_exact'] > 0]
    positive = np.array([r['delta_days_exact'] for r in positive_rows], dtype=float)

    def save(fig, name, explanation, plotted):
        for ext in ('png', 'svg'):
            fig.savefig(folder/f'{name}.{ext}', dpi=160)
        plt.close(fig)
        figures.append({'name': name, 'explanation': explanation,
                        'files': [f'02_figures/{name}.png', f'02_figures/{name}.svg'], 'plotted_data': plotted})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    data = {}
    for ax, label, values in ((axes[0], 'All usable listed records', signed),
                              (axes[1], 'Positive delays up to 365 days', positive[positive <= 365])):
        counts, edges, _ = ax.hist(values, bins=40, color='#237f8b', edgecolor='white', linewidth=.4)
        ax.set(title=f'{label} (n={len(values)})', xlabel='VulnCheck addition - NVD publication (stored days)', ylabel='CVEs')
        ax.axvline(0, color='#a9483b', lw=1)
        data[label] = {'n': len(values), 'bin_edges': edges.tolist(), 'counts': counts.tolist()}
    fig.suptitle('Recorded publication-to-VulnCheck delay: full range and first-year detail')
    save(fig, '01_publication_to_vckev_histogram',
         'Negative values are retained on the left. The right panel is an explicitly restricted view, not the entire cohort.', data)

    categories = [x for x in report['temporal_bracket_counts'] if x['category'] != 'NOT_LISTED']
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    bars = ax.bar([x['category'] for x in categories], [x['count'] for x in categories],
                  color=['#b65743'] + ['#287f8b']*(len(categories)-1))
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, max([x['count'] for x in categories], default=1)*1.18+1)
    ax.tick_params(axis='x', labelrotation=20)
    ax.set(ylabel='Listed CVEs', xlabel='Exclusive category; NEGATIVE has priority over SAME_DAY',
           title=f"Stored-time brackets (listed n={report['vckev_listed']}); same calendar day including negatives: {report['same_calendar_day_including_negative']}")
    save(fig, '02_exclusive_temporal_categories',
         'Each listed CVE appears exactly once. SAME_DAY excludes negative same-day values. Unlisted records are excluded, not assigned a delay.', categories)

    x = np.sort(positive)
    y = np.arange(1, len(x)+1)/len(x)*100 if len(x) else np.array([])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax in axes:
        ax.step(x, y, where='post', color='#267d83')
        for h in (30, 60, 90):
            ax.axvline(h, color='#9c6850', alpha=.5, lw=.9)
        ax.set(xlabel='Stored days since NVD publication', ylabel='Cumulative share of observed positive delays (%)', ylim=(0, 102))
    axes[0].set_xscale('symlog', linthresh=1)
    axes[0].set_xlim(0, max(1, float(x[-1])*1.08) if len(x) else 180)
    axes[0].set_title(f'All positive-delay events (n={len(x)}); symmetric-log x axis')
    axes[1].set(xlim=(0, 180), title='First 180 days; same denominator as full range')
    fig.suptitle('Conditional empirical CDF, not an exploitation probability or survival estimate')
    save(fig, '03_cumulative_addition_delay',
         'Denominator is only listed CVEs with strictly positive stored delays. Negative cases and unlisted/censored records are not a risk set in this figure.',
         {'delta_days': x.tolist(), 'cumulative_percent': y.tolist(), 'denominator': len(x)})

    years = sorted({r['publication_year'] for r in positive_rows if r['publication_year'] is not None})
    groups = [[r['delta_days_exact'] for r in positive_rows if r['publication_year'] == year] for year in years]
    fig, ax = plt.subplots(figsize=(13, 6), layout='constrained')
    if groups:
        ax.boxplot(groups, tick_labels=[f'{year}\nn={len(group)}' for year, group in zip(years, groups)], showfliers=True)
    else:
        ax.text(.5, .5, 'No eligible positive delays', ha='center', transform=ax.transAxes)
    ax.set_yscale('symlog', linthresh=1)
    ax.tick_params(axis='x', labelrotation=55)
    ax.set(ylabel='Positive stored delta days (symmetric-log scale)', xlabel='NVD publication year; counts are observed positive-delay events',
           title=f'Publication-year delay distributions (n={sum(map(len, groups))}); not adjusted for cohort selection or censoring')
    save(fig, '04_delay_by_publication_year',
         'Boxplots summarize positive delays with outliers shown. They cannot establish faster future inclusion for newer cohorts.',
         {'years': years, 'values': groups})

    labels = sorted({f"{r['cvss_vector_version'] or 'MISSING'} | {r['cvss_vector_severity'] or 'MISSING'}" for r in positive_rows})
    groups = [[r['delta_days_exact'] for r in positive_rows
               if f"{r['cvss_vector_version'] or 'MISSING'} | {r['cvss_vector_severity'] or 'MISSING'}" == label] for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    if groups:
        ax.boxplot(groups, tick_labels=[f'{label}\nn={len(group)}' for label, group in zip(labels, groups)])
    else:
        ax.text(.5, .5, 'No eligible positive delays', ha='center', transform=ax.transAxes)
    ax.set_yscale('symlog', linthresh=1)
    ax.tick_params(axis='x', labelrotation=35)
    ax.set(ylabel='Positive stored delta days (symmetric-log scale)', xlabel='Current CVSS version | stored severity',
           title=f'Positive delay by CVSS severity and version (n={len(positive_rows)}); retrospective association only')
    save(fig, '05_delay_by_cvss_severity',
         'Version groups remain separate. Stored current severity is neither a catalogue label nor necessarily available at publication.',
         {'groups': labels, 'values': groups})

    negative = [r for r in listed if r['delta_days_exact'] < 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    negdata = {}
    for subtype, color in [('earlier_calendar_day', '#b35843'), ('same_calendar_day_earlier_timestamp', '#277e88')]:
        values = [r['delta_days_exact'] for r in negative if r['negative_subtype'] == subtype]
        transformed = -np.log1p(np.abs(values))
        counts, edges, _ = axes[0].hist(transformed, bins=30, histtype='step', linewidth=2,
                                        label=f'{subtype.replace("_", " ")} (n={len(values)})', color=color)
        negdata[subtype] = {'delta_days': values, 'transformed_edges': edges.tolist(), 'counts': counts.tolist()}
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel='-log(1 + absolute stored delta days)', ylabel='CVEs', title=f'All negative delays (n={len(negative)})')
    same = [24*r['delta_days_exact'] for r in negative if r['negative_subtype']=='same_calendar_day_earlier_timestamp']
    counts, edges, _ = axes[1].hist(same, bins=np.linspace(-24, 0, 25), color='#277e88')
    axes[1].set(xlabel='Stored delta hours on the same UTC day', ylabel='CVEs', title=f'Same-day earlier timestamps (n={len(same)})')
    negdata['same_day_hours_histogram'] = {'edges': edges.tolist(), 'counts': counts.tolist()}
    fig.suptitle('Negative chronology retained for review; no automatic timestamp correction')
    save(fig, '06_negative_delta_distribution',
         'Day-level ambiguity can explain uncertain ordering for same-day cases; it cannot by itself account for earlier-calendar-day cases.', negdata)

    cisa = [r for r in listed if r['vckev_to_cisa_days'] is not None]
    values = [r['vckev_to_cisa_days'] for r in cisa]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    counts, edges, _ = axes[0].hist(values, bins=40, color='#84607b')
    axes[0].axvline(0, color='black', lw=1)
    axes[0].set(xlabel='CISA addition - VulnCheck addition (stored days)', ylabel='Paired CVEs', title='Positive values mean VulnCheck has the earlier stored date')
    axes[1].scatter([r['delta_days_exact'] for r in cisa], [r['publication_to_cisa_days'] for r in cisa], s=13, alpha=.5, color='#287d86')
    axes[1].axline((0, 0), slope=1, color='#b75d42', ls='--', label='Equal stored delay')
    axes[1].set(xlabel='NVD publication to VulnCheck (days)', ylabel='NVD publication to CISA (days)')
    axes[1].legend()
    fig.suptitle(f'Catalogue timing comparison (paired n={len(cisa)}); not an operational detection-speed claim')
    save(fig, '07_vckev_cisa_comparison',
         'Only paired CISA metadata within the VulnCheck subset is shown. The catalogues and their entry rules are not interchangeable.',
         {'counts': counts.tolist(), 'edges': edges.tolist(), 'paired_cves': [r['cve'] for r in cisa],
          'publication_to_vckev': [r['delta_days_exact'] for r in cisa],
          'publication_to_cisa': [r['publication_to_cisa_days'] for r in cisa], 'comparison': report['cisa_comparison']})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout='constrained')
    policies = [('primary_precision_aware', 'Primary: precision-aware\nno implicit complete-snapshot cutoff'),
                ('stored_timestamp_only', 'Stored timestamps as points\nno day-precision adjustment'),
                ('documented_snapshot_sensitivity', 'Documented snapshot assumption\nsensitivity only')]
    for ax, (policy, title) in zip(axes, policies):
        subset = [r for r in report['horizon_label_counts'] if r['policy']==policy]
        x = np.arange(3)
        for offset, kind, color in [(-.25,'positive','#277f88'),(0,'negative','#bb6941'),(.25,'unknown','#777777')]:
            bars = ax.bar(x+offset, [r[kind] for r in subset], width=.25, color=color, label=kind)
            ax.bar_label(bars, fontsize=8, padding=3, rotation=90)
        ax.set_xticks(x, ['30d','60d','90d'])
        ax.set(title=title, xlabel='Cumulative horizon', ylim=(0, max(len(rows)*1.15, 1)))
    axes[0].set_ylabel('CVEs')
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3, fontsize=8)
    fig.suptitle('Provisional horizon labels: 0, 1 and unknown are kept separate')
    save(fig, '08_horizon_label_counts',
         'A late observed event can support a horizon zero, while an unlisted CVE needs adequate documented observation. Unknown is not a negative class.', report['horizon_label_counts'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    coverage = report['observation_coverage']
    for ax, title, names in [(axes[0], 'Primary observation duration', ['primary_full_duration','primary_short_duration','primary_unknown_duration']),
                             (axes[1], 'Documented snapshot-day assumption', ['documented_full_duration','documented_short_duration','documented_unknown_duration'])]:
        bottom = np.zeros(3)
        for name, color in zip(names, ['#287e88','#b96a42','#888888']):
            values = np.array([r[name] for r in coverage])
            bars = ax.bar(['30d','60d','90d'], values, bottom=bottom, color=color, label=name.replace('_',' '))
            for bar, value, low in zip(bars, values, bottom):
                if value:
                    tiny = value < .02*len(rows)
                    ax.annotate(str(value), (bar.get_x()+bar.get_width()/2, low+value if tiny else low+value/2),
                                xytext=(0, 7 if tiny else 0), textcoords='offset points',
                                ha='center', va='bottom' if tiny else 'center', fontsize=9)
            bottom += values
        ax.set(title=title, ylabel='CVEs', xlabel='Required follow-up from publication')
        ax.set_ylim(0, max(1, len(rows)*1.12))
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0,-.15))
    fig.suptitle('Duration coverage: short follow-up differs from an unknown observation cutoff')
    save(fig, '09_followup_and_censoring',
         'Full duration alone does not validate labels. Short duration is right censoring only when no resolving event was observed; negative/precision-review cases remain separate in the tables.', coverage)

    return figures
