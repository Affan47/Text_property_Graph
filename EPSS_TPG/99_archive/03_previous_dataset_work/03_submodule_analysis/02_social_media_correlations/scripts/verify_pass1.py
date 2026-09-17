"""Independent re-verification of every headline number, recomputed from a
different file (mistral) and with different code paths where possible."""
import pandas as pd, numpy as np, os, csv, sys
csv.field_size_limit(10**9)
from scipy import stats
pd.set_option("display.width",220)
D="/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"

# --- V1: raw csv row count via csv module (independent of pandas parsing)
for f in os.listdir(D):
    with open(os.path.join(D,f), newline="", encoding="utf-8") as fh:
        r=csv.reader(fh); hdr=next(r); n=sum(1 for _ in r)
    print(f"V1 {f}: csv-module data rows={n} header cols={len(hdr)}")

m=pd.read_csv(os.path.join(D,"mistral_combined_summ.csv"),low_memory=False)
g=pd.read_csv(os.path.join(D,"gemma_combined_summ.csv"),low_memory=False)
m["cve_base"]=m["cve"].str.split("-").str[:3].str.join("-")
print("\nV2 mistral rows",len(m),"unique base CVE",m["cve_base"].nunique())
print("V2 base-CVE parse cross-check vs regex:",
      (m["cve_base"]==m["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]).all())

print("\nV3 source counts (mistral):", m["source"].value_counts().to_dict())
print("V3 identical to gemma:", (m["source"].value_counts()==g["source"].value_counts()).all())

print("\nV4 EPSS: mean %.6f median %.6f min %.6f max %.6f" %
      (m.epss_score.mean(), m.epss_score.median(), m.epss_score.min(), m.epss_score.max()))
print("V4 share <0.001:", round((m.epss_score<0.001).mean(),4),
      "| share >=0.9:", round((m.epss_score>=0.9).mean(),4),
      "| count >=0.9:", int((m.epss_score>=0.9).sum()))
print("V4 EPSS never exactly 0 or 1:", (m.epss_score>0).all(), (m.epss_score<1).all())

print("\nV5 CVSS mean %.4f median %.2f" % (m.cvss_score.mean(), m.cvss_score.median()))
sev=pd.cut(m.cvss_score,[-.01,3.9,6.9,8.9,10.0],labels=["Low","Medium","High","Critical"])
print("V5 severity counts:", sev.value_counts().to_dict())
print("V5 severity shares:", (sev.value_counts(normalize=True)*100).round(2).to_dict())

print("\nV6 spearman(cvss,epss) row-level:", round(stats.spearmanr(m.cvss_score,m.epss_score)[0],4))
cl=m.groupby("cve_base",as_index=False).first()
print("V6 spearman(cvss,epss) CVE-level:", round(stats.spearmanr(cl.cvss_score,cl.epss_score)[0],4), "n=",len(cl))
print("V6 pearson(cvss,epss) row-level:", round(stats.pearsonr(m.cvss_score,m.epss_score)[0],4))
# median-based re-check of the sign flip
print("V6 median EPSS by CVSS severity, ROW level:")
print(m.assign(sev=sev).groupby("sev",observed=True)["epss_score"].agg(["count","median","mean"]).round(5).to_string())
print("V6 median EPSS by CVSS severity, CVE level:")
sevc=pd.cut(cl.cvss_score,[-.01,3.9,6.9,8.9,10.0],labels=["Low","Medium","High","Critical"])
print(cl.assign(sev=sevc).groupby("sev",observed=True)["epss_score"].agg(["count","median","mean"]).round(5).to_string())

print("\nV7 cve_year vs epss (CVE level) spearman:", round(stats.spearmanr(
      cl["cve_base"].str[4:8].astype(int), cl.epss_score)[0],4))
print("V7 mean EPSS 2025 CVEs:", round(cl.loc[cl.cve_base.str.startswith('CVE-2025'),'epss_score'].mean(),5),
      "| <=2022 CVEs:", round(cl.loc[cl.cve_base.str[4:8].astype(int)<=2022,'epss_score'].mean(),5))

print("\nV8 occurrence_count vs realised row count mismatches:",
      int((m.occurrence_count != m.cve_base.map(m.cve_base.value_counts())).sum()))

print("\nV9 dates")
d=pd.to_datetime(m.date_posted)
print("  range", d.min().date(), d.max().date(), "| NaT:", d.isna().sum())
nt=m[m.source!="Telegram"]; dnt=pd.to_datetime(nt.date_posted)
print("  non-Telegram rows", len(nt), "range", dnt.min().date(), dnt.max().date())
print("  non-Telegram unique CVEs", nt.cve_base.nunique())
print("  Telegram rows", (m.source=="Telegram").sum(), "Telegram unique CVEs", m.loc[m.source=='Telegram','cve_base'].nunique())
print("  share of non-Telegram rows in 2025:", round((dnt.dt.year==2025).mean(),4))

print("\nV10 post-text reuse (recomputed with factorize, mistral)")
post=m.social_media_post.fillna("").astype(str)
code=pd.factorize(post)[0]
tmp=pd.DataFrame({"c":code,"cve":m.cve_base})
ncve=tmp.groupby("c")["cve"].nunique()
shared=set(ncve.index[ncve>1])
print("  unique texts:",len(ncve),"| texts on >1 CVE:",len(shared),
      "| rows affected:", int(tmp.c.isin(shared).sum()), f"({tmp.c.isin(shared).mean()*100:.1f}%)")
print("  max CVEs per single text:", int(ncve.max()))

print("\nV11 HackerNews post size")
hn=m[m.source=="HackerNews"]; L=hn.social_media_post.fillna("").astype(str).str.len()
print("  n",len(hn),"mean chars %.0f median %.0f max %d | unique texts %d"%(L.mean(),L.median(),L.max(),hn.social_media_post.nunique()))

print("\nV12 nulls in key cols (mistral vs gemma)")
for c in ["days_since_latest_git_source","source_links","github_urls"]:
    print(f"  {c}: mistral {m[c].isna().sum()} gemma {g[c].isna().sum()}")

print("\nV13 summary non-null counts")
for k,d in [("gemma",g),("mistral",m),("gpt",pd.read_csv(os.path.join(D,'gpt_combined_summ.csv'),low_memory=False))]:
    print(f"  {k}: all_sources {d.summ_all_sources.notna().sum()}, github {d.summ_github_urls.notna().sum()}, cvss {d.summ_cvss_metrics.notna().sum()}")

print("\nV14 epss_status")
print(m.groupby("epss_status")["epss_score"].agg(["count","mean","median"]).round(5).to_string())
print(pd.crosstab(m.source,m.epss_status).to_string())
