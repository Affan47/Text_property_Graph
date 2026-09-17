"""Refinement of the coverage audit.

Fixes two things in the first pass:
 (1) "entered EPSS after the post" was computed against the coarse grid, which
     overstates it by up to the grid spacing (7d in the dense period). Recomputed
     with proper bracketing: certain / certain-not / ambiguous.
 (2) The version-boundary filter is only needed if the outcome is raw EPSS. The
     daily files ship a within-day percentile, which is renormalised every day,
     so a model migration shifts levels but not ranks. Sample sizes are therefore
     reported both with and without that filter.
"""
import pandas as pd, numpy as np, os, glob

OUT = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Temporal EPSS analysis"
ev = pd.read_csv(os.path.join(OUT, "tables", "events_with_coverage.csv"),
                 parse_dates=["event_date", "first_seen", "last_seen", "entry_lower"])
panel = pd.read_parquet(os.path.join(OUT, "tables", "grid_panel.parquet"))
grid = np.array(sorted(panel.grid_date.unique()))
print("events:", len(ev), "| grid points:", len(grid))

# ---------- (1) When did the CVE enter EPSS, relative to the post? ----------
# true entry lies in (entry_lower, first_seen].  entry_lower is NaT if the CVE
# was already present on the very first grid date.
after_certain  = ev["entry_lower"] >= ev["event_date"]          # absent at a grid date on/after t
before_certain = ev["first_seen"]  <= ev["event_date"]          # present at a grid date on/before t
ambiguous = ~(after_certain | before_certain)
print("\n--- D (corrected). CVE's EPSS entry vs the post date ---")
print(f"  certainly entered EPSS AFTER the post : {int(after_certain.sum()):5d} ({after_certain.mean()*100:.1f}%)")
print(f"  certainly entered EPSS BEFORE/ON post : {int(before_certain.sum()):5d} ({before_certain.mean()*100:.1f}%)")
print(f"  ambiguous within grid spacing         : {int(ambiguous.sum()):5d} ({ambiguous.mean()*100:.1f}%)")
print("  (first pass reported 73.3% 'after' by comparing against the coarse grid alone;")
print("   the bracketed figure above is the defensible one.)")

# how tight is post timing to EPSS entry?
lag_hi = (ev["first_seen"] - ev["event_date"]).dt.days      # upper bound on entry-minus-post
print("\n  upper bound on (EPSS entry date - post date), days:")
print("   ", lag_hi.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_dict())

# ---------- (2) Sample sizes for each candidate design ----------
specific = ~ev["first_post_shared"]
print("\n--- E (revised). Usable sample per design ---")

print("\n  DESIGN 1 - forward-looking forecast study (needs post-window only):")
d1 = specific.copy()
print(f"    CVE-specific first post, post-window available : {int(d1.sum()):5d}")
print(f"      of which first post is a Mastodon feed item  : {int((d1 & (ev.first_source=='Mastodon')).sum()):5d}")
print(f"      non-Mastodon                                  : {int((d1 & (ev.first_source!='Mastodon')).sum()):5d}")

print("\n  DESIGN 2 - two-sided event study (needs pre-window):")
rows = []
for W in (7, 14, 30):
    ok = ev[f"pre{W}_ok"] == "yes"
    rows.append(dict(window=f"+/-{W}d",
                     with_prewindow=int(ok.sum()),
                     plus_specific_text=int((ok & specific).sum()),
                     plus_version_filter=int((ok & specific & ~ev[f"window{W}_crosses_version"]).sum())))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print("    -> the last column is only required if the outcome is raw EPSS;")
print("       with a percentile outcome the middle column is the usable n.")

for W in (7, 14, 30):
    m = (ev[f"pre{W}_ok"] == "yes") & specific
    print(f"\n    +/-{W}d, percentile outcome (n={int(m.sum())}) by first platform:")
    print("     ", ev.loc[m, "first_source"].value_counts().to_dict())

# ---------- (3) Does the mature-CVE subsample even look different? ----------
mature = (ev["pre30_ok"] == "yes")
print("\n--- F. Is the pre-window subsample representative? ---")
cmp = pd.DataFrame({
    "has 30d pre-history": ev.loc[mature, ["cve_year", "cvss_score", "epss_at_collection"]].mean(),
    "no 30d pre-history":  ev.loc[~mature, ["cve_year", "cvss_score", "epss_at_collection"]].mean()}).round(3)
print(cmp.to_string())
print("\n  first-platform mix, mature vs fresh (%):")
mix = pd.crosstab(ev["first_source"], mature, normalize="columns").mul(100).round(1)
mix.columns = ["no 30d pre-history", "has 30d pre-history"]
print(mix.to_string())

ev["entry_vs_post"] = np.select([after_certain, before_certain], ["after", "before_or_on"], default="ambiguous")
ev.to_csv(os.path.join(OUT, "tables", "events_with_coverage.csv"), index=False)
R.to_csv(os.path.join(OUT, "tables", "design_sample_sizes.csv"), index=False)
print("\nupdated -> events_with_coverage.csv, design_sample_sizes.csv")
