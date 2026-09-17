import pandas as pd, numpy as np, os

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df = pd.read_csv(os.path.join(D, "gemma_combined_summ.csv"), low_memory=False)
df["cve_base"] = df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"] = pd.to_datetime(df["date_posted"], errors="coerce")

print("### occurrence_count mismatch rows")
cnt = df["cve_base"].value_counts()
mm = df[df["occurrence_count"] != df["cve_base"].map(cnt)]
print("n mismatch:", len(mm))
print(mm[["cve","source","occurrence_count"]].assign(actual=mm["cve_base"].map(cnt)).head(20).to_string())

print("\n### EPSS")
print(df["epss_score"].describe())
for q in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]:
    print(f"  q{q}: {df['epss_score'].quantile(q):.5f}")
print("zeros/near-zero (<0.001):", (df["epss_score"]<0.001).mean())
print("==1.0 :", (df["epss_score"]>=0.99).sum(), " >0.5:", (df["epss_score"]>0.5).sum())
print("\nepss_status vs source:\n", pd.crosstab(df["source"], df["epss_status"], normalize="index").round(3))
print("\nEPSS by status:\n", df.groupby("epss_status")["epss_score"].describe())

print("\n### is epss constant per base CVE?")
gg = df.groupby("cve_base")["epss_score"].nunique()
print("base CVEs with >1 distinct epss:", (gg>1).sum(), "of", len(gg))
gg2 = df.groupby("cve_base")["cvss_score"].nunique()
print("base CVEs with >1 distinct cvss:", (gg2>1).sum())
gg3 = df.groupby("cve_base")["description"].nunique()
print("base CVEs with >1 distinct description:", (gg3>1).sum())

print("\n### CVSS")
print(df["cvss_score"].describe())
def sev(s):
    return pd.cut(s,[-.01,3.9,6.9,8.9,10],labels=["Low","Medium","High","Critical"])
df["severity"] = sev(df["cvss_score"])
print(df["severity"].value_counts(normalize=True).round(4))
for c in ["attack_vector","attack_complexity","privileges_required","user_interaction","scope",
          "confidentiality_impact","integrity_impact","availability_impact"]:
    print(f"\n{c}:\n{df[c].value_counts(dropna=False)}")

print("\n### boolean/count features")
print("sources_available:\n", df["sources_available"].value_counts())
print("github_links_with_code_available:\n", df["github_links_with_code_available"].value_counts())
print("occurrence_count describe:\n", df["occurrence_count"].describe())
print("days_since_latest_git_source:\n", df["days_since_latest_git_source"].describe())
print("days_since_oldest_git_source:\n", df["days_since_oldest_git_source"].describe())
print("neg days latest:", (df["days_since_latest_git_source"]<0).sum(), "neg oldest:", (df["days_since_oldest_git_source"]<0).sum())
print("latest<=oldest violations:", (df["days_since_latest_git_source"]>df["days_since_oldest_git_source"]).sum())

print("\n### github_links_with_code_available vs github_urls nullity")
print(pd.crosstab(df["github_links_with_code_available"], df["github_urls"].notna()))
print("\ngithub flag vs days_since_latest notna:")
print(pd.crosstab(df["github_links_with_code_available"], df["days_since_latest_git_source"].notna()))
print("\nsources_available vs source_links notna:")
print(pd.crosstab(df["sources_available"], df["source_links"].notna()))
