import pandas as pd, numpy as np, os, re
pd.set_option("display.width",250)
D="/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
dfs={k:pd.read_csv(os.path.join(D,f"{k}_combined_summ.csv"),low_memory=False) for k in ["gemma","gpt","mistral"]}
b=dfs["gemma"]

print("W1 summary coverage (recount)")
for k,d in dfs.items():
    row=[]
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        t=d[c]; nonempty=t.notna()&(t.astype(str).str.strip()!="")
        row.append(f"{c}={int(nonempty.sum())}")
    print("  ",k, " ".join(row))
print("   denominators: source_links present =",int(b.source_links.notna().sum()),
      "| github_urls present =",int(b.github_urls.notna().sum()),"| all rows =",len(b))

print("\nW2 coverage GAP vs its own input")
for k,d in dfs.items():
    g1=int((d.summ_all_sources.isna()&b.source_links.notna()).sum())
    g2=int((d.summ_github_urls.isna()&b.github_urls.notna()).sum())
    g3=int(d.summ_cvss_metrics.isna().sum())
    print(f"   {k:8s} missing-all_sources={g1:4d}  missing-github={g2:4d}  missing-cvss={g3:4d}")

print("\nW3 median summary length in words")
for k,d in dfs.items():
    r=[]
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        w=d[c].dropna().astype(str).str.split().str.len()
        r.append(f"{c}={w.median():.0f}")
    print("  ",k," ".join(r))

print("\nW4 markdown fingerprint on summ_cvss_metrics (share of non-empty)")
for k,d in dfs.items():
    t=d.summ_cvss_metrics.dropna().astype(str)
    print(f"   {k:8s} bold={t.str.contains(r'\*\*').mean()*100:5.1f}%  bulleted={t.str.contains(r'(?m)^\s*[-*] ').mean()*100:5.1f}%")

print("\nW5 'not a real vulnerability' style hallucination (tight regex, manual-verified wording)")
pat=r"(hypothetical|fictional|placeholder|does not exist|not a real|non-existent)"
for k,d in dfs.items():
    r=[]
    for c in ["summ_all_sources","summ_github_urls"]:
        t=d[c].dropna().astype(str); h=t.str.contains(pat,case=False)
        r.append(f"{c}: {int(h.sum())}/{len(t)} = {h.mean()*100:.2f}%")
    print("  ",k,"  |  ".join(r))

print("\nW6 CVSS numeric faithfulness in summ_cvss_metrics (strict 'score' anchor)")
pat2=re.compile(r'(?:cvss\s*(?:base\s*)?score\s*(?:is|of|:|=)?\s*|score\s*(?:of|is|:|=)\s*|\(\s*cvss\s*score:?\s*)(\d{1,2}(?:\.\d)?)',re.I)
for k,d in dfs.items():
    ext=d.summ_cvss_metrics.fillna("").str.extract(pat2)[0].astype(float)
    ok=ext.notna(); ma=np.isclose(ext[ok],b.cvss_score[ok])
    print(f"   {k:8s} parsed={int(ok.sum()):5d}  agree={int(ma.sum()):5d} ({ma.mean()*100:.2f}%)  disagree={int((~ma).sum())}")

print("\nW7 truncation (no terminal punctuation)")
for k,d in dfs.items():
    r=[]
    for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"]:
        t=d[c].dropna().astype(str).str.strip(); t=t[t!=""]
        tr=~t.str.endswith((".","!","?",")","`","*",'"'))
        r.append(f"{c}={int(tr.sum())} ({tr.mean()*100:.1f}%)")
    print("  ",k," ".join(r))

print("\nW8 no summary mentions EPSS (leak check)")
for k,d in dfs.items():
    n=sum(int(d[c].fillna('').str.contains('EPSS',case=False).sum()) for c in ["summ_all_sources","summ_github_urls","summ_cvss_metrics"])
    print("  ",k,n)

print("\nW9 misc integrity")
print("   social_media_post nulls:", int(b.social_media_post.isna().sum()))
print("   distinct description per base CVE >1:",
      int((b.assign(cb=b.cve.str.extract(r'^(CVE-\d{4}-\d+)')[0]).groupby('cb')['description'].nunique()>1).sum()))
print("   cvss_score deterministic given 8 components:",
      int((b.groupby(["attack_vector","attack_complexity","privileges_required","user_interaction","scope",
                      "confidentiality_impact","integrity_impact","availability_impact"])["cvss_score"].nunique()>1).sum()),
      "combos violating (0 = deterministic)")
print("   negative days_since_latest_git_source:", int((b.days_since_latest_git_source<0).sum()),
      "of", int(b.days_since_latest_git_source.notna().sum()))
print("   github flag True but no git-date:", int((b.github_links_with_code_available&b.days_since_latest_git_source.isna()).sum()))
print("   github flag False but git-date present:", int((~b.github_links_with_code_available&b.days_since_latest_git_source.notna()).sum()))
