import pandas as pd, numpy as np, os
from scipy import stats
pd.set_option("display.width", 200)

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df = pd.read_csv(os.path.join(D, "gemma_combined_summ.csv"), low_memory=False)
df["cve_base"] = df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"] = pd.to_datetime(df["date_posted"], errors="coerce")

print("### Repeated exact EPSS values -- which sources?")
for v in [0.00043, 0.00885, 0.00045, 0.00091, 0.01156]:
    sub = df[np.isclose(df["epss_score"], v)]
    print(f"\n epss={v} n={len(sub)}")
    print("   by source:", sub["source"].value_counts().to_dict())
    print("   by status:", sub["epss_status"].value_counts().to_dict())
    print("   n unique base CVE:", sub["cve_base"].nunique())

print("\n\n### Do duplicate rows of the same CVE differ only in post?")
multi = df[df.duplicated("cve_base", keep=False)]
print("rows in multi-post CVEs:", len(multi), "CVEs:", multi['cve_base'].nunique())
g = multi.groupby("cve_base")
print("CVEs whose duplicate rows span >1 source:", (g["source"].nunique()>1).sum())
print("CVEs whose duplicate rows have identical social_media_post:", (g["social_media_post"].nunique()==1).sum())
print("exact duplicate (source+post) rows:", df.duplicated(subset=["cve_base","source","social_media_post"]).sum())
print("exact duplicate social_media_post text overall:", df["social_media_post"].duplicated().sum())

print("\n### CVE-LEVEL (deduplicated, first row per CVE) correlations")
cl = df.sort_values("date").groupby("cve_base", as_index=False).first()
print("n CVE-level rows:", len(cl))
print("epss describe:\n", cl["epss_score"].describe())
for c in ["cvss_score","occurrence_count","days_since_latest_git_source","days_since_oldest_git_source"]:
    sub = cl[[c,"epss_score"]].dropna()
    sp = stats.spearmanr(sub[c], sub["epss_score"])
    print(f"  spearman({c}, epss) = {sp[0]:+.3f} p={sp[1]:.2e} n={len(sub)}")
def eps2(H,n,k): return (H-k+1)/(n-k)
for c in ["attack_vector","privileges_required","confidentiality_impact","integrity_impact",
          "availability_impact","scope","user_interaction","attack_complexity","source","epss_status"]:
    groups=[gg["epss_score"].values for _,gg in cl.groupby(c) if len(gg)>=5]
    if len(groups)<2: continue
    H,p = stats.kruskal(*groups)
    print(f"  KW {c}: H={H:.1f} p={p:.2e} eps2={eps2(H,len(cl),len(groups)):.4f}")
print("\nCVE-level source medians:\n", cl.groupby("source")["epss_score"].agg(["count","mean","median"]).round(5))

print("\n### CVE age vs EPSS  (year of CVE id)")
cl["cve_year"] = cl["cve_base"].str.extract(r'CVE-(\d{4})')[0].astype(int)
print(cl.groupby("cve_year")["epss_score"].agg(["count","mean","median"]).round(4).to_string())
sp = stats.spearmanr(cl["cve_year"], cl["epss_score"])
print("spearman(cve_year, epss) =", round(sp[0],3), "p=", f"{sp[1]:.2e}")

print("\n### disclosure lag: date_posted - CVE year (proxy), excluding Telegram")
nt = df[df["source"]!="Telegram"].copy()
nt["cve_year"] = nt["cve_base"].str.extract(r'CVE-(\d{4})')[0].astype(int)
nt["post_year"] = nt["date"].dt.year
print(pd.crosstab(nt["post_year"], nt["cve_year"]).to_string())
