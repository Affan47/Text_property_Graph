"""Static, offline figures. Every figure's plotted values accompany the reports."""
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = 'cves_unique_with_source_links_clean'
REF = 'cves_merged_refetched'
KEV = 'cves_merged_with_url_dates_vc_kev'


def make_figures(directory, profiles, datasets, member, drift, correlations, *, missing_kind, parse_date):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    figures = []

    def usable_score(value):
        return type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 10

    def save(fig, name, description, data):
        for extension in ('png', 'svg'):
            fig.savefig(directory / f'{name}.{extension}', dpi=160)
        plt.close(fig)
        figures.append({'name': name, 'description': description,
                        'files': [f'04_figures/{name}.png', f'04_figures/{name}.svg'],
                        'plotted_data': data})

    names = list(profiles)
    labels = ['Base CSV', 'Base JSON', 'Refetched CSV', 'Refetched JSON', 'URL dates JSON', 'KEV JSON']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    counts = [profiles[n]['rows'] for n in names]
    fields = [profiles[n]['top_level_feature_count'] for n in names]
    for ax, values, title, color in zip(axes, [counts, fields], ['Records', 'Distinct top-level fields'], ['#217a87', '#ba6438']):
        bars = ax.barh(labels, values, color=color)
        ax.bar_label(bars, padding=3)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_title(title)
    fig.suptitle('Six files, four related dataset stages (not independent cohorts)')
    save(fig, '01_file_overview', 'Union field counts differ from fields present on each individual record.',
         {'files': names, 'rows': counts, 'top_level_fields': fields})

    keys = sorted({k for p in profiles.values() for k in p['top_level_features']})
    matrix = np.array([[profiles[n]['features'][k]['missing_percent'] if k in profiles[n]['features'] else np.nan
                        for n in names] for k in keys])
    fig, ax = plt.subplots(figsize=(11, max(9, len(keys) * .25)), layout='constrained')
    cmap = plt.colormaps['YlOrRd'].copy()
    cmap.set_bad('#d5d8dc')
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_xticks(range(len(names)), labels, rotation=25, ha='right')
    ax.set_title('Unavailable field values (%)\nGray = field absent from the entire file; empty lists are not missing')
    fig.colorbar(im, ax=ax, label='Missing values / file records (%)', shrink=.5)
    save(fig, '02_feature_missingness', 'Separates absent schema columns from within-file missingness.',
         {'fields': keys, 'files': names, 'missing_percent': [[None if np.isnan(v) else float(v) for v in line] for line in matrix]})

    keys = sorted(member['availability_by_membership'])
    matrix = [[member['availability_by_membership'][k][g]['missing_percent'] for g in ('not_listed', 'listed')] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 11), layout='constrained')
    im = ax.imshow(np.asarray(matrix, dtype=float), cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks([0, 1], [f"Not listed (n={member['counts']['not_listed']})", f"Listed (n={member['counts']['listed']})"])
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    for i, values in enumerate(matrix):
        for j, value in enumerate(values):
            if value is not None:
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=7, color='white' if value > 60 else 'black')
    ax.set_title('Missingness by VulnCheck membership (%)\nUndefined groups are masked; inspect availability before choosing inputs')
    fig.colorbar(im, ax=ax, shrink=.5)
    save(fig, '03_membership_missingness', 'Missingness is descriptive, not a suggested predictor.',
         {'fields': keys, 'groups': ['not_listed', 'listed'], 'missing_percent': matrix})

    years = sorted(member['publication_year_counts'])
    listed = np.array([member['publication_year_counts'][y].get('listed', 0) for y in years])
    other = np.array([member['publication_year_counts'][y].get('not_listed', 0) for y in years])
    total = listed + other
    rates = np.divide(100 * listed, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    latest = max([int(y) for y in years if y.isdigit()], default=0)
    x = np.array([int(y) if y.isdigit() else latest + 1 for y in years])
    axes[0].bar(x, other, label='Not listed', color='#267f8e')
    axes[0].bar(x, listed, bottom=other, label='Listed', color='#bb5941')
    axes[0].set_ylabel('CVEs')
    axes[0].legend()
    axes[1].plot(x, rates, marker='o', color='#8c3b65')
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('Listed (%)')
    axes[1].set_xticks(x, years, rotation=60)
    axes[1].set_xlabel('NVD publication year; small yearly cohorts can produce extreme rates')
    fig.suptitle('Publication cohorts and observed catalogue membership')
    save(fig, '04_publication_membership', 'Counts and rates; not a temporal prediction evaluation.',
         {'years': years, 'listed': listed.tolist(), 'not_listed': other.tolist(), 'listed_percent': rates.tolist()})

    rows = datasets[KEV + '.json']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    distributions = {}
    colors = ['#227c8b', '#b26035', '#746284']
    for version, color in zip([2.0, 3.0, 3.1], colors):
        group = [r for r in rows if r.get('cvss_vector_version') == version]
        values = [r['cvss_vector_score'] for r in group if usable_score(r.get('cvss_vector_score'))]
        counts, bins, _ = axes[0].hist(values, bins=np.arange(0, 10.5, .5), histtype='step', linewidth=2,
                                      label=f'{version} (n={len(values)})', color=color)
        distributions[str(version)] = {'bin_edges': bins.tolist(), 'counts': counts.tolist(),
                                       'version_group_rows': len(group), 'excluded_missing_or_invalid_scores': len(group) - len(values)}
    axes[0].legend()
    axes[0].set_xlabel('Stored score (not a cross-version severity scale)')
    axes[0].set_ylabel('CVEs')
    missing = [sum(r.get('cvss_vector_version') == v and bool(missing_kind(r.get('cvss_vector_severity'))) for r in rows) for v in (2.0, 3.0, 3.1)]
    absent = sum(r.get('cvss_vector_version') not in {2.0, 3.0, 3.1} and bool(missing_kind(r.get('cvss_vector_severity'))) for r in rows)
    axes[1].bar(['2.0', '3.0', '3.1', 'Other/missing version'], missing + [absent], color=colors + ['#666666'])
    axes[1].set_ylabel('CVEs without a stored severity')
    axes[1].set_title('Missingness depends on metric version')
    fig.suptitle('CVSS coverage and stored score distributions')
    save(fig, '05_cvss_coverage', 'Version-specific histograms and severity availability.',
         {'histograms': distributions, 'severity_missing_by_version': dict(zip(['2.0', '3.0', '3.1', 'other_or_missing_version'], missing + [absent]))})

    matrix = np.array(correlations['matrix'], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    names = correlations['features']
    if matrix.size:
        cmap = plt.colormaps['RdBu_r'].copy()
        cmap.set_bad('#d5d8dc')
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, shrink=.65, label='Spearman rho; gray = undefined')
    else:
        ax.text(.5, .5, 'No eligible varying numeric features in the CVSS 3.1 cohort',
                transform=ax.transAxes, ha='center', va='center', wrap=True)
    ax.set_xticks(range(len(names)), names, rotation=55, ha='right')
    ax.set_yticks(range(len(names)), names)
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            if np.isfinite(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value) > .65 else 'black')
    ax.set_title(f"Spearman association: CVSS 3.1 cohort (n={correlations['cohort_rows']})\nPairwise sample sizes are provided in the JSON/TXT report")
    save(fig, '06_numeric_associations', correlations['note'], correlations)

    hosts = profiles[KEV + '.json']['quality']['top_reference_hosts_by_distinct_cves']
    top = list(hosts)[:20]
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    ax.barh(top[::-1], [hosts[k] for k in top[::-1]], color='#297d76')
    ax.set_xlabel('Distinct CVEs referencing this exact host (a CVE may have multiple hosts)')
    ax.set_title('Reference-host coverage; repeated URL slots count once per CVE')
    save(fig, '07_reference_hosts', 'Exact parsed hostnames; no network checks or domain-family merging.', {k: hosts[k] for k in top})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    distributions = {}
    for flag, label, color in [(False, 'Not listed', '#227c8b'), (True, 'Listed', '#bb5941')]:
        group = [r for r in rows if r.get('in_vckev') is flag]
        distributions[label] = {}
        for ax, name, values in [(axes[0], 'description_words', [len(r['description'].split()) for r in group]),
                                  (axes[1], 'unique_reference_urls', [len(set(r['all_sources_urls'])) for r in group])]:
            maximum = max([len(r['description'].split()) if name == 'description_words' else len(set(r['all_sources_urls'])) for r in rows], default=0)
            edges = np.linspace(0, max(float(np.log1p(maximum)), 1.0), 26)
            counts, edges, _ = ax.hist(np.log1p(values), bins=edges, weights=np.ones(len(values)) / max(len(values), 1),
                                       histtype='step', linewidth=2, label=f'{label} (n={len(group)})', color=color)
            distributions[label][name] = {'log1p_bin_edges': edges.tolist(), 'fractions': counts.tolist(), 'cohort_n': len(group)}
    for ax, name in zip(axes, ['Description word count', 'Unique reference count']):
        ax.set_xlabel(f'log(1 + {name.lower()})')
        ax.set_ylabel('Fraction within membership group')
        ax.legend()
    fig.suptitle('Text and reference-size distributions (log transform; all records retained)')
    save(fig, '08_text_reference_sizes', 'Within-group normalization prevents majority-class counts dominating the comparison.', distributions)

    pairs = [(r['orig_cvss_score'], r['cvss_vector_score']) for r in rows
             if r.get('orig_cvss_version') == 3.1 and r.get('cvss_vector_version') == 3.1
             and usable_score(r.get('orig_cvss_score')) and usable_score(r.get('cvss_vector_score'))]
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    counts = Counter(pairs)
    ax.scatter([a for a, b in counts], [b for a, b in counts], s=[12 + 9 * np.sqrt(n) for n in counts.values()], alpha=.5, color='#287d86')
    ax.plot([0, 10], [0, 10], '--', color='#a9483a', label='Unchanged score')
    ax.set(xlim=(0, 10.3), ylim=(0, 10.3), xlabel='Original CVSS 3.1 score', ylabel='Refetched CVSS 3.1 score')
    ax.legend()
    ax.set_title(f'Original versus refetched scores: {len(pairs)} paired CVEs\nMarker area increases with count (square-root scaling)')
    save(fig, '09_score_revision', 'Only same-version 3.1 pairs; repeated score pairs are aggregated.',
         {'pairs': [{'original': a, 'current': b, 'count': n} for (a, b), n in sorted(counts.items())], 'overall_drift': drift})

    lags = {'Modification - publication (CVEs)': [], 'First reference - publication (URLs)': [], 'KEV addition - publication (listed CVEs)': []}
    for row in rows:
        published = parse_date(row.get('published_timestamp'))
        if not published:
            continue
        for label, value in [(list(lags)[0], row.get('last_modified_timestamp')), (list(lags)[2], row.get('vckev_date_added'))]:
            d = parse_date(value)
            if d:
                lags[label].append((d - published).total_seconds() / 86400)
        for item in row.get('url_change_dates', []):
            d = parse_date(item.get('earliest_added_date'))
            if d:
                lags[list(lags)[1]].append((d - published).total_seconds() / 86400)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout='constrained')
    distributions = {}
    for ax, (label, values), color in zip(axes, lags.items(), ['#267f8e', '#b5693c', '#796088']):
        transformed = np.sign(values) * np.log1p(np.abs(values))
        counts, edges, _ = ax.hist(transformed, bins=40, color=color)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(f'{label}, n={len(values)}')
        ax.set_xlabel('sign(days) x log(1 + abs(days)); negative values are retained')
        ax.set_ylabel('Observations')
        distributions[label] = {'transformed_bin_edges': edges.tolist(), 'counts': counts.tolist()}
    save(fig, '10_temporal_intervals', 'Different event clocks, not feature-availability guarantees or time-to-exploitation measurements.', distributions)

    modes = profiles[KEV + '.json']['quality']['history_match_modes']
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    bars = ax.bar(list(modes), list(modes.values()), color='#856040')
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, max(modes.values(), default=1) * 1.15)
    ax.set_ylabel('Reference-history entries (not CVEs)')
    ax.set_title('Reference-history matching methods')
    save(fig, '11_reference_history_matching', 'An exact or fallback match does not verify the original web publication time.', modes)
    return figures
