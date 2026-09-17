import pandas as pd, numpy as np, os, re
pd.set_option("display.width", 220)
D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
files = {"gemma":"gemma_combined_summ.csv","gpt":"gpt_combined_summ.csv","mistral":"mistral_combined_summ.csv"}
dfs = {k: pd.read_csv(os.path.join(D,f), low_memory=False) for k,f in files.items()}
base = dfs["gemma"]

print("### Precise CVSS numeric-consistency check on summ_cvss_metrics")
# capture only tokens that are explicitly a *score*
pat = re.compile(r'(?:cvss\s*(?:base\s*)?score[^0-9\n]{0,20}|score\s*(?:of|is|:|=)\s*|severity[^0-9\n]{0,20}\(\s*(?:cvss\s*score:?\s*)?)(\d{1,2}(?:\.\d)?)', re.I)
for k,d in dfs.items():
    txt = d["summ_cvss_metrics"].fillna("")
    ext = txt.str.extract(pat)[0].astype(float)
    ok = ext.notna()
    true = base["cvss_score"]
    match = np.isclose(ext[ok], true[ok])
    print(f"  {k:8s}: score parsed in {ok.sum():5d} rows | equals true score {match.sum():5d} ({match.mean()*100:.2f}%) | contradicts {(~match).sum()}")
    bad = ok.copy(); bad[ok] = ~match
    ex = pd.DataFrame({"stated":ext[bad], "true":true[bad]}).head(8)
    print(ex.to_string())

print("\n### Explicit 'temporal disbelief' markers (model doubts a 2024/2025 CVE is real)")
pats = {
 "not_real": r"(hypothetical|fictional|does not (?:actually )?exist|not a real|appears to be (?:a )?(?:fake|placeholder)|likely a placeholder|may not be a real)",
 "future_cve": r"(in the future|future date|year 2025 (?:suggests|indicates)|is in the future|has not (?:yet )?been (?:released|published))",
 "beyond_cutoff": r"(knowledge cutoff|training data|as of my last update|my knowledge)",
}
for k,d in dfs.items():
    print(f" -- {k}")
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        t=d[c].fillna(""); n=(t.str.strip()!="").sum()
        s=[]
        for m,p in pats.items():
            h=t.str.contains(p,case=False,regex=True).sum(); s.append(f"{m}={h} ({h/max(n,1)*100:.2f}%)")
        print(f"    {c:20s} n={n:5d} " + "  ".join(s))

print("\n### 'Exploitation likelihood' language -- is it EPSS-correlated (leakage check)?")
kw = {"high_likelihood": r"(high(?:ly)? likelihood|highly likely|likelihood[^.]{0,20}high|actively exploited|exploited in the wild|known exploited)",
      "low_likelihood" : r"(low likelihood|unlikely to be exploited|likelihood[^.]{0,20}low)"}
for k,d in dfs.items():
    print(f" -- {k}")
    for c in ["summ_all_sources","summ_cvss_metrics"]:
        t=d[c].fillna("")
        hi=t.str.contains(kw["high_likelihood"],case=False)
        lo=t.str.contains(kw["low_likelihood"],case=False)
        m = t.str.strip()!=""
        print(f"    {c:20s} high-lang {hi.sum():5d} -> mean EPSS {base.loc[hi,'epss_score'].mean():.4f} (med {base.loc[hi,'epss_score'].median():.5f}) | "
              f"low-lang {lo.sum():5d} -> mean EPSS {base.loc[lo,'epss_score'].mean():.4f} (med {base.loc[lo,'epss_score'].median():.5f}) | "
              f"neither {(m&~hi&~lo).sum():5d} -> mean {base.loc[m&~hi&~lo,'epss_score'].mean():.4f}")

print("\n### Truncation check (summary ends without terminal punctuation)")
for k,d in dfs.items():
    r=[]
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        t=d[c].dropna().astype(str).str.strip()
        t=t[t!=""]
        trunc=~t.str.endswith((".","!","?",")","`","*",'"',"’","”"))
        r.append(f"{c}={trunc.sum()} ({trunc.mean()*100:.1f}%)")
    print(f"  {k:8s} " + "  ".join(r))

print("\n### Do summaries leak the exact EPSS number?")
for k,d in dfs.items():
    hits=0
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        t=d[c].fillna("")
        hits += t.str.contains(r"EPSS", case=False).sum()
    print(f"  {k}: rows mentioning 'EPSS' across summary cols = {hits}")
