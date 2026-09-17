import pandas as pd, numpy as np, os, re
from scipy import stats
pd.set_option("display.width", 220)
D="/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df=pd.read_csv(os.path.join(D,"gemma_combined_summ.csv"),low_memory=False)
df["cve_base"]=df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"]=pd.to_datetime(df["date_posted"],errors="coerce")
df["post"]=df["social_media_post"].fillna("").astype(str)
df["post_ch"]=df["post"].str.len(); df["post_wd"]=df["post"].str.split().str.len()

print("### HackerNews giant posts -- what are they?")
hn=df[df["source"]=="HackerNews"].sort_values("post_ch",ascending=False)
print(hn["post_ch"].describe())
t=hn["post"].iloc[0]
print("longest post first 600 chars:\n", t[:600].replace("\n"," "))
print("\n... contains '|' separators:", t.count("|"), " newlines:", t.count("\n"))
print("HN posts identical text share:", hn["post"].duplicated().mean().round(3))
print("HN: n unique post texts:", hn["post"].nunique(), "of", len(hn))
print("\nduplicate post text by source (share of rows whose post text repeats):")
print(df.assign(d=df.duplicated("post",keep=False)).groupby("source")["d"].mean().round(3))

print("\n### post length vs EPSS")
for src in [None]+sorted(df["source"].unique()):
    sub = df if src is None else df[df["source"]==src]
    sp=stats.spearmanr(sub["post_ch"], sub["epss_score"])
    print(f"  {'ALL' if src is None else src:18s} n={len(sub):5d} spearman(post_chars, epss)={sp[0]:+.3f} p={sp[1]:.2e}")

print("\n### TEMPORAL (Telegram EXCLUDED)")
nt=df[df["source"]!="Telegram"].copy()
print("n rows:", len(nt), " unique CVEs:", nt["cve_base"].nunique())
print("date range:", nt["date"].min().date(), "->", nt["date"].max().date())
mo=nt.groupby(nt["date"].dt.to_period("M")).agg(posts=("cve","size"), cves=("cve_base","nunique"),
        mean_epss=("epss_score","mean"), med_epss=("epss_score","median"), mean_cvss=("cvss_score","mean"))
print(mo.round(4).to_string())

print("\n### monthly source mix (Telegram excluded), last 12 months")
mix=pd.crosstab(nt["date"].dt.to_period("M"), nt["source"])
print(mix.tail(14).to_string())

print("\n### day-of-week / weekday effect (Telegram excluded)")
nt["dow"]=nt["date"].dt.day_name()
print(nt.groupby("dow").agg(n=("cve","size"), mean_epss=("epss_score","mean"), med_epss=("epss_score","median")).round(4)
      .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).to_string())

print("\n### post date vs EPSS trend (Telegram excluded)")
nt["days_from_start"]=(nt["date"]-nt["date"].min()).dt.days
sp=stats.spearmanr(nt["days_from_start"], nt["epss_score"])
print("spearman(post recency, epss) =", round(sp[0],3), f"p={sp[1]:.2e}")
for s in sorted(nt["source"].unique()):
    sub=nt[nt["source"]==s]
    sp=stats.spearmanr(sub["days_from_start"], sub["epss_score"])
    print(f"   {s:18s} n={len(sub):5d} rho={sp[0]:+.3f} p={sp[1]:.2e}")

print("\n### CVE-id year vs post year gap (Telegram excluded)")
nt["cve_year"]=nt["cve_base"].str.extract(r'CVE-(\d{4})')[0].astype(int)
nt["gap"]=nt["date"].dt.year-nt["cve_year"]
print(nt["gap"].value_counts().sort_index().to_string())
print("share posted in same year as CVE id:", (nt["gap"]==0).mean().round(3))
print("negative gap (posted before CVE year):", (nt["gap"]<0).sum())
print("\nmean EPSS by gap:")
print(nt.groupby("gap")["epss_score"].agg(["count","mean","median"]).round(4).to_string())
