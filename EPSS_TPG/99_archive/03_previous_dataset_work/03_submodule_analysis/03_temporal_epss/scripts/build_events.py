"""Build the candidate event table for the temporal/event-study stage.

Event = (CVE, first non-Telegram post date). Telegram is excluded because its
date_posted records message-craft time, not publication time.
"""
import pandas as pd, numpy as np, os

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
OUT = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Temporal EPSS analysis"
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

df = pd.read_csv(os.path.join(D, "gemma_combined_summ.csv"), low_memory=False)
df["cve_base"] = df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"] = pd.to_datetime(df["date_posted"])
df["post"] = df["social_media_post"].fillna("").astype(str)

# flag posts whose text is shared across >1 CVE (contamination, defect D1)
code = pd.factorize(df["post"])[0]
ncve = pd.DataFrame({"c": code, "cve": df["cve_base"]}).groupby("c")["cve"].nunique()
df["text_shared"] = pd.Series(code, index=df.index).isin(set(ncve.index[ncve > 1]))

nt = df[df["source"] != "Telegram"].copy()

ev = (nt.sort_values(["date", "cve"])
        .groupby("cve_base", as_index=False)
        .agg(event_date=("date", "first"),
             first_source=("source", "first"),
             n_posts=("cve", "size"),
             n_platforms=("source", "nunique"),
             last_post=("date", "max"),
             any_specific_text=("text_shared", lambda s: (~s).any()),
             first_post_shared=("text_shared", "first"),
             epss_at_collection=("epss_score", "first"),
             epss_status=("epss_status", "first"),
             cvss_score=("cvss_score", "first"),
             attack_vector=("attack_vector", "first"),
             sources_available=("sources_available", "first"),
             gh_code=("github_links_with_code_available", "first")))
ev["cve_year"] = ev["cve_base"].str[4:8].astype(int)
ev["span_days"] = (ev["last_post"] - ev["event_date"]).dt.days

# EPSS model-version era of the event date (boundaries verified against the daily files)
V3 = pd.Timestamp("2023-03-07"); V4 = pd.Timestamp("2025-03-17")
ev["epss_model_era"] = np.select(
    [ev.event_date < pd.Timestamp("2022-02-01"), ev.event_date < V3, ev.event_date < V4],
    ["pre-v2022", "v2022.01.01", "v2023.03.01"], default="v2025.03.14")
for w in (7, 14, 30):
    ev[f"window{w}_crosses_version"] = (
        ((ev.event_date - pd.Timedelta(days=w) < V3) & (ev.event_date + pd.Timedelta(days=w) >= V3)) |
        ((ev.event_date - pd.Timedelta(days=w) < V4) & (ev.event_date + pd.Timedelta(days=w) >= V4)))

ev = ev.sort_values("event_date").reset_index(drop=True)
ev.to_csv(os.path.join(OUT, "tables", "candidate_events.csv"), index=False)

print("candidate events:", len(ev))
print("date range:", ev.event_date.min().date(), "->", ev.event_date.max().date())
print("\nby EPSS model era:\n", ev.epss_model_era.value_counts().to_string())
print("\n+/-30d window crosses a version boundary:", int(ev.window30_crosses_version.sum()),
      f"({ev.window30_crosses_version.mean()*100:.1f}%)")
print("+/-14d:", int(ev.window14_crosses_version.sum()), "| +/-7d:", int(ev.window7_crosses_version.sum()))
print("\nfirst post text is CVE-specific:", int((~ev.first_post_shared).sum()),
      f"({(~ev.first_post_shared).mean()*100:.1f}%)")
print("has at least one CVE-specific post:", int(ev.any_specific_text.sum()))
print("\nfirst_source:\n", ev.first_source.value_counts().to_string())
print("\nwritten ->", os.path.join(OUT, "tables", "candidate_events.csv"))
