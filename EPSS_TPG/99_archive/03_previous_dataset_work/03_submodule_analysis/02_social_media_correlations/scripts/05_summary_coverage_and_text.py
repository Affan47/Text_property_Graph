import pandas as pd, numpy as np, os, re
pd.set_option("display.width", 220)

D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
files = {"gemma": "gemma_combined_summ.csv", "gpt": "gpt_combined_summ.csv", "mistral": "mistral_combined_summ.csv"}
dfs = {k: pd.read_csv(os.path.join(D, f), low_memory=False) for k, f in files.items()}
base = dfs["gemma"]

SUMS = ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]

print("### Summary-column coverage vs the input that should drive it")
print("rows:", len(base))
print("sources_available=True :", (base['sources_available']).sum(), " source_links notna:", base['source_links'].notna().sum())
print("github_urls notna     :", base['github_urls'].notna().sum(), " gh_code flag True:", base['github_links_with_code_available'].sum())
for k, d in dfs.items():
    print(f"\n-- {k}")
    for c in SUMS:
        nn = d[c].notna().sum()
        blank = (d[c].fillna("").astype(str).str.strip()=="" ).sum()
        print(f"   {c:20s} non-null={nn:5d}  missing={len(d)-nn:5d}  blank-after-strip={blank}")
    # mismatch: summary present but no input
    print("   summ_all_sources present but source_links MISSING:",
          (d["summ_all_sources"].notna() & base["source_links"].isna()).sum())
    print("   summ_all_sources MISSING but source_links present:",
          (d["summ_all_sources"].isna() & base["source_links"].notna()).sum())
    print("   summ_github_urls present but github_urls MISSING:",
          (d["summ_github_urls"].notna() & base["github_urls"].isna()).sum())
    print("   summ_github_urls MISSING but github_urls present:",
          (d["summ_github_urls"].isna() & base["github_urls"].notna()).sum())
    print("   summ_cvss_metrics MISSING:", d["summ_cvss_metrics"].isna().sum())

print("\n\n### Text length statistics (characters and whitespace-words)")
def stats_txt(s):
    t = s.dropna().astype(str)
    ch = t.str.len(); wd = t.str.split().str.len()
    return dict(n=len(t), ch_mean=ch.mean(), ch_med=ch.median(), ch_p95=ch.quantile(.95), ch_max=ch.max(),
                wd_mean=wd.mean(), wd_med=wd.median(), wd_max=wd.max())
rows=[]
rows.append(dict(col="social_media_post", model="-", **stats_txt(base["social_media_post"])))
rows.append(dict(col="description", model="-", **stats_txt(base["description"])))
for k,d in dfs.items():
    for c in SUMS:
        rows.append(dict(col=c, model=k, **stats_txt(d[c])))
R = pd.DataFrame(rows)
print(R.round(1).to_string(index=False))

print("\n### social_media_post length by source")
b = base.copy(); b["post_ch"]=b["social_media_post"].astype(str).str.len(); b["post_wd"]=b["social_media_post"].astype(str).str.split().str.len()
print(b.groupby("source")[["post_ch","post_wd"]].agg(["count","mean","median","max"]).round(1).to_string())

print("\n### Placeholder / anonymisation tokens in posts")
for tok in ["<URL>","<USER>","<EMAIL>","http"]:
    m = base["social_media_post"].astype(str).str.contains(re.escape(tok), case=False)
    print(f"  {tok:8s}: rows containing = {m.sum():5d} ({m.mean()*100:.1f}%)")
print("  mean <URL> per post:", base["social_media_post"].astype(str).str.count("<URL>").mean().round(2))
print("  by source, share of posts with <URL>:")
print(base.assign(u=base["social_media_post"].astype(str).str.contains("<URL>")).groupby("source")["u"].mean().round(3))
