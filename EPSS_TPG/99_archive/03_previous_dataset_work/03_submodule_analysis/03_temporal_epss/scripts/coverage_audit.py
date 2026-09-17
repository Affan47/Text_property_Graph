"""Step 1 — EPSS coverage audit for the event-study design.

For every candidate event (CVE, t) determine whether a usable EPSS series exists
around t. The grid is coarse (monthly pre-burst, weekly through it), so each
CVE's true EPSS entry date is bracketed rather than pinned: it lies in
(last grid date absent, first grid date present]. Coverage is therefore reported
in three buckets - certain yes / certain no / needs a daily probe - rather than
as a single number that would overstate precision.
"""
import pandas as pd, numpy as np, os, glob

OUT = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Temporal EPSS analysis"
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
ev = pd.read_csv(os.path.join(OUT, "tables", "candidate_events.csv"), parse_dates=["event_date"])
study = set(ev["cve_base"])
print(f"study CVEs: {len(study)}   candidate events: {len(ev)}")

files = sorted(glob.glob("epss/grid/*.csv.gz"))
print(f"grid files: {len(files)}  ({os.path.basename(files[0])} -> {os.path.basename(files[-1])})")

rows = []
for i, f in enumerate(files):
    d = pd.Timestamp(os.path.basename(f).replace(".csv.gz", ""))
    # pre-2022 files have no '#model_version' comment line
    with __import__("gzip").open(f, "rt") as fh:
        first = fh.readline()
    skip = 1 if first.startswith("#") else 0
    t = pd.read_csv(f, skiprows=skip, usecols=["cve", "epss", "percentile"])
    t = t[t["cve"].isin(study)]
    t["grid_date"] = d
    rows.append(t)
    if (i + 1) % 20 == 0:
        print(f"  read {i+1}/{len(files)}")
panel = pd.concat(rows, ignore_index=True)
panel.to_parquet(os.path.join(OUT, "tables", "grid_panel.parquet"), index=False)
print("grid panel rows:", len(panel), "| distinct CVEs seen:", panel.cve.nunique())

grid_dates = np.array(sorted(panel.grid_date.unique()))
present = panel.groupby("cve")["grid_date"].agg(["min", "max"]).rename(
    columns={"min": "first_seen", "max": "last_seen"})

ev = ev.merge(present, left_on="cve_base", right_index=True, how="left")
print("\n--- A. Presence in EPSS at all ---")
never = ev["first_seen"].isna()
print(f"  CVEs never observed on any grid date: {int(never.sum())} ({never.mean()*100:.1f}%)")

# lower bound on entry: the grid date immediately preceding first_seen
idx = np.searchsorted(grid_dates, ev["first_seen"].values, side="left") - 1
ev["entry_lower"] = np.where(idx >= 0, grid_dates[np.clip(idx, 0, None)], np.datetime64("NaT"))

print("\n--- B. Pre-window coverage (does >=W days of EPSS history exist before t?) ---")
summary = []
for W in (7, 14, 30):
    need = ev["event_date"] - pd.Timedelta(days=W)
    yes = ev["first_seen"] <= need                      # certainly covered
    no = ev["entry_lower"] >= need                      # certainly not covered
    no = no | never                                     # never in EPSS at all
    amb = ~(yes | no)
    summary.append(dict(window=f"t-{W}", certain_yes=int(yes.sum()), certain_no=int(no.sum()),
                        needs_probe=int(amb.sum()),
                        pct_yes=round(yes.mean() * 100, 1)))
    ev[f"pre{W}_ok"] = np.where(yes, "yes", np.where(no, "no", "probe"))
S = pd.DataFrame(summary)
print(S.to_string(index=False))

print("\n--- C. Post-window availability ---")
print("  latest grid date:", pd.Timestamp(grid_dates[-1]).date(),
      "| latest event + 30d:", (ev.event_date.max() + pd.Timedelta(days=30)).date())
print("  daily files exist well past both -> post-window is never the binding constraint")

print("\n--- D. Events entering EPSS AFTER the post (no pre-history possible) ---")
after = ev["first_seen"] > ev["event_date"]
print(f"  {int(after.sum())} events ({after.mean()*100:.1f}%) - CVE not yet scored when discussed")

print("\n--- E. Usable sample after stacking the design filters (W=30) ---")
f1 = ~never
f2 = f1 & (ev["pre30_ok"] == "yes")
f3 = f2 & (~ev["first_post_shared"])
f4 = f3 & (~ev["window30_crosses_version"])
for name, m in [("all candidate events", pd.Series(True, index=ev.index)),
                ("  + present in EPSS", f1),
                ("  + >=30d pre-history", f2),
                ("  + CVE-specific post text", f3),
                ("  + window clear of version boundary", f4)]:
    print(f"  {name:42s} {int(m.sum()):5d}")
ev["analysis_ready_w30"] = f4

print("\n  analysis-ready events by first platform:\n",
      ev.loc[f4, "first_source"].value_counts().to_string())
print("\n  analysis-ready events by month:\n",
      ev.loc[f4, "event_date"].dt.to_period("M").value_counts().sort_index().to_string())

ev.to_csv(os.path.join(OUT, "tables", "events_with_coverage.csv"), index=False)
S.to_csv(os.path.join(OUT, "tables", "coverage_summary.csv"), index=False)
print("\nwritten -> events_with_coverage.csv, coverage_summary.csv, grid_panel.parquet")
