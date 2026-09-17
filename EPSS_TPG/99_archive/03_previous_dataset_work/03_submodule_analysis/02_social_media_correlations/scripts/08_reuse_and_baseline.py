import pandas as pd, numpy as np, os, hashlib
from scipy import stats
pd.set_option("display.width",220)
D="/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df=pd.read_csv(os.path.join(D,"gemma_combined_summ.csv"),low_memory=False)
df["cve_base"]=df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["post"]=df["social_media_post"].fillna("").astype(str)
df["ph"]=df["post"].map(lambda s: hashlib.md5(s.encode()).hexdigest())

print("### Post-text reuse across different CVEs")
g=df.groupby("ph").agg(n_rows=("cve","size"), n_cves=("cve_base","nunique"), n_src=("source","nunique"),
                       src=("source", lambda s: s.mode().iat[0]), chars=("post", lambda s: len(s.iat[0])))
print("unique post texts:", len(g), "for", len(df), "rows")
print("post texts attached to >1 distinct CVE:", (g["n_cves"]>1).sum())
print("rows covered by such shared texts:", g.loc[g["n_cves"]>1,"n_rows"].sum(), f"({g.loc[g['n_cves']>1,'n_rows'].sum()/len(df)*100:.1f}% of rows)")
print("\ntop shared post texts:")
print(g.sort_values("n_cves",ascending=False).head(12)[["n_rows","n_cves","n_src","src","chars"]].to_string())
print("\nreuse by source (share of rows whose post text is attached to >1 CVE):")
sh=set(g.index[g["n_cves"]>1])
print(df.assign(x=df["ph"].isin(sh)).groupby("source")["x"].agg(["mean","sum"]).round(3).to_string())

print("\n### Same CVE, same source, identical post -> exact duplicate records")
dup=df.duplicated(subset=["cve_base","source","post"],keep=False)
print("rows in exact-duplicate groups:", dup.sum())

print("\n### Predictive-signal baseline: classify EPSS>=0.1 (high) from tabular features")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder
y=(df["epss_score"]>=0.1).astype(int)
print("positive rate:", y.mean().round(4), "n_pos:", y.sum())
cats=["attack_vector","attack_complexity","privileges_required","user_interaction","scope",
      "confidentiality_impact","integrity_impact","availability_impact"]
X_cvss=pd.concat([df[["cvss_score"]], pd.DataFrame(OrdinalEncoder().fit_transform(df[cats]),columns=cats)],axis=1)
X_meta=df[["occurrence_count"]].join(df[["sources_available","github_links_with_code_available"]].astype(int))
X_meta["days_latest"]=df["days_since_latest_git_source"].fillna(-9999)
X_meta["days_oldest"]=df["days_since_oldest_git_source"].fillna(-9999)
X_src=pd.get_dummies(df["source"],prefix="src").astype(int)
X_year=df["cve_base"].str.extract(r'CVE-(\d{4})')[0].astype(int).to_frame("cve_year")
groups=df["cve_base"]
sets={"CVSS only":X_cvss,"metadata only":X_meta,"source only":X_src,"cve_year only":X_year,
      "CVSS+meta":pd.concat([X_cvss,X_meta],axis=1),
      "ALL tabular":pd.concat([X_cvss,X_meta,X_src,X_year],axis=1)}
cv=GroupKFold(n_splits=5)
for name,X in sets.items():
    m=GradientBoostingClassifier(random_state=0)
    pr=cross_val_predict(m,X,y,cv=cv,groups=groups,method="predict_proba")[:,1]
    print(f"  {name:16s} grouped-CV ROC-AUC={roc_auc_score(y,pr):.3f}  PR-AUC={average_precision_score(y,pr):.3f}")
