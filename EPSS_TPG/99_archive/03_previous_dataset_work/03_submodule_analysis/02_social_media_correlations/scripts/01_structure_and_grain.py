import pandas as pd, numpy as np, os, re, hashlib

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
files = {"gemma": "gemma_combined_summ.csv", "gpt": "gpt_combined_summ.csv", "mistral": "mistral_combined_summ.csv"}
dfs = {k: pd.read_csv(os.path.join(D, f), low_memory=False) for k, f in files.items()}

SHARED = ['cve','source','date_posted','time_posted','social_media_post','epss_score','epss_status',
          'description','cvss_version','cvss_score','attack_vector','attack_complexity','privileges_required',
          'user_interaction','scope','confidentiality_impact','integrity_impact','availability_impact',
          'occurrence_count','sources_available','github_links_with_code_available','days_since_latest_git_source',
          'days_since_oldest_git_source','source_links','github_urls']

print("### 1. Are shared columns identical across the 3 files (row-aligned)?")
g, p, m = dfs["gemma"], dfs["gpt"], dfs["mistral"]
for c in SHARED:
    a, b, cc = g[c], p[c], m[c]
    eq_gp = a.equals(b); eq_gm = a.equals(cc)
    if not (eq_gp and eq_gm):
        # compare treating NaN==NaN
        d1 = ((a.isna() & b.isna()) | (a.astype(str) == b.astype(str))).all()
        d2 = ((a.isna() & cc.isna()) | (a.astype(str) == cc.astype(str))).all()
        print(f"  {c}: gemma==gpt {eq_gp}/{d1}, gemma==mistral {eq_gm}/{d2}")
print("  (no output above => all shared cols byte-identical)")

df = g.copy()

print("\n### 2. CVE id structure")
print(df["cve"].head(5).tolist())
base = df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
suffix = df["cve"].str.extract(r'^CVE-\d{4}-\d+-(\d+)$')[0]
print("rows:", len(df), "| parsed base non-null:", base.notna().sum(), "| unique base CVEs:", base.nunique())
print("suffix parse non-null:", suffix.notna().sum())
print("suffix value counts (top):\n", suffix.value_counts().head(10))
df["cve_base"] = base
df["dup_idx"] = pd.to_numeric(suffix, errors="coerce")
print("\nposts per base CVE distribution:\n", base.value_counts().value_counts().sort_index().head(20))
print("top base CVEs by #posts:\n", base.value_counts().head(10))
print("cve year distribution:\n", base.str.extract(r'CVE-(\d{4})')[0].value_counts().sort_index())

print("\n### 3. occurrence_count vs actual duplicate count consistency")
cnt = base.value_counts()
df["_n_rows_for_cve"] = base.map(cnt)
print(pd.crosstab(df["occurrence_count"], df["_n_rows_for_cve"]).iloc[:8, :8])
print("occurrence_count == n_rows_for_cve fraction:", (df["occurrence_count"] == df["_n_rows_for_cve"]).mean())

print("\n### 4. Dates")
d = pd.to_datetime(df["date_posted"], errors="coerce")
print("unparseable dates:", d.isna().sum())
print("global range:", d.min(), "->", d.max())
print("\nper-source date range / count:")
tmp = pd.DataFrame({"source": df["source"], "date": d})
print(tmp.groupby("source")["date"].agg(["min", "max", "count", "nunique"]))
print("\ntime_posted: unique values count =", df["time_posted"].nunique())
print(df["time_posted"].value_counts().head(10))
print("\nfraction time == 00:00 per source:")
print((df.assign(zero=df["time_posted"].astype(str).str.startswith("00:00")).groupby("source")["zero"].mean()))
