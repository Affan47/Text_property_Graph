import pandas as pd, numpy as np, os
from scipy import stats

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df = pd.read_csv(os.path.join(D, "gemma_combined_summ.csv"), low_memory=False)
df["cve_base"] = df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"] = pd.to_datetime(df["date_posted"], errors="coerce")
df["logit_epss"] = np.log10(df["epss_score"].clip(1e-5, 1-1e-6) / (1 - df["epss_score"].clip(1e-5, 1-1e-6)))
df["log_epss"] = np.log10(df["epss_score"])

print("### EPSS histogram (raw, 20 bins)")
h, e = np.histogram(df["epss_score"], bins=20, range=(0,1))
for i in range(20):
    print(f"  [{e[i]:.2f},{e[i+1]:.2f}): {h[i]:5d}  {h[i]/len(df)*100:5.2f}%")
print("\nEPSS histogram (log10 scale)")
h, e = np.histogram(df["log_epss"], bins=10)
for i in range(10):
    print(f"  10^[{e[i]:.2f},{e[i+1]:.2f}): {h[i]:5d}")
print("\nmost frequent exact EPSS values:")
print(df["epss_score"].value_counts().head(15))

print("\n### Numeric correlations with EPSS (row level, n=%d)" % len(df))
num = ["cvss_score","occurrence_count","days_since_latest_git_source","days_since_oldest_git_source"]
for c in num:
    sub = df[[c,"epss_score","log_epss"]].dropna()
    pr = stats.pearsonr(sub[c], sub["epss_score"])
    sp = stats.spearmanr(sub[c], sub["epss_score"])
    prl = stats.pearsonr(sub[c], sub["log_epss"])
    print(f"  {c:32s} n={len(sub):5d} pearson(raw)={pr[0]:+.3f} (p={pr[1]:.2e}) spearman={sp[0]:+.3f} (p={sp[1]:.2e}) pearson(log10)={prl[0]:+.3f}")

print("\nnumeric-numeric spearman matrix:")
print(df[num+["epss_score","cvss_score"]].drop(columns=["cvss_score"]).corr(method="spearman").round(3))

print("\n### Binary features vs EPSS")
for c in ["sources_available","github_links_with_code_available"]:
    a = df.loc[df[c]==True,"epss_score"]; b = df.loc[df[c]==False,"epss_score"]
    u = stats.mannwhitneyu(a,b)
    # rank-biserial
    rb = 2*u[0]/(len(a)*len(b)) - 1
    print(f"  {c}: mean True={a.mean():.4f} (med {a.median():.5f}, n={len(a)}) | mean False={b.mean():.4f} (med {b.median():.5f}, n={len(b)}) MWU p={u[1]:.2e} rank-biserial={rb:+.3f}")

print("\n### Categorical CVSS components vs EPSS (Kruskal-Wallis + epsilon^2)")
cats = ["attack_vector","attack_complexity","privileges_required","user_interaction","scope",
        "confidentiality_impact","integrity_impact","availability_impact","source","epss_status","cvss_version"]
def eps2(H, n, k):
    return (H - k + 1) / (n - k)
for c in cats:
    groups = [gg["epss_score"].values for _, gg in df.groupby(c) if len(gg) >= 5]
    k = len(groups)
    if k < 2: continue
    H, p = stats.kruskal(*groups)
    print(f"\n  {c}: H={H:.1f} p={p:.2e} epsilon^2={eps2(H,len(df),k):.4f}")
    print(df.groupby(c)["epss_score"].agg(["count","mean","median"]).round(5).to_string())

print("\n### Cramer's V among categorical features")
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = chi2/n
    phi2c = max(0, phi2 - (k-1)*(r-1)/(n-1))
    rc = r - (r-1)**2/(n-1); kc = k - (k-1)**2/(n-1)
    return np.sqrt(phi2c/max(1e-12, min(kc-1, rc-1)))
cc = ["source","attack_vector","attack_complexity","privileges_required","user_interaction","scope",
      "confidentiality_impact","integrity_impact","availability_impact","epss_status"]
M = pd.DataFrame(index=cc, columns=cc, dtype=float)
for i in cc:
    for j in cc:
        M.loc[i,j] = cramers_v(df[i], df[j])
print(M.round(3).to_string())
