import pandas as pd, numpy as np, os
from scipy import stats
pd.set_option("display.width",250)
D="/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"
df=pd.read_csv(os.path.join(D,"gemma_combined_summ.csv"),low_memory=False)
df["cve_base"]=df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"]=pd.to_datetime(df["date_posted"])

print("### Cross-platform CVE overlap (all sources)")
piv=pd.crosstab(df["cve_base"],df["source"])>0
print("CVEs covered by n platforms:\n", piv.sum(1).value_counts().sort_index().to_string())
srcs=list(piv.columns)
J=pd.DataFrame(index=srcs,columns=srcs,dtype=float)
for a in srcs:
    for b in srcs:
        A=set(piv.index[piv[a]]); B=set(piv.index[piv[b]])
        J.loc[a,b]=len(A&B)/len(A|B)
print("\nJaccard overlap of CVE sets between platforms:\n", J.round(3).to_string())
print("\nabsolute shared-CVE counts:\n", (piv.T.astype(int)@piv.astype(int)).to_string())

print("\n### Which platform posts FIRST about a CVE (Telegram excluded)")
nt=df[df["source"]!="Telegram"]
multi=nt.groupby("cve_base")["source"].nunique()
mc=set(multi.index[multi>1])
sub=nt[nt["cve_base"].isin(mc)]
first=sub.sort_values("date").groupby("cve_base").first()["source"]
print("CVEs covered by >1 non-Telegram platform:", len(mc))
print("first-poster share:\n", first.value_counts().to_string())
# median lead time vs the others
lead=[]
for cve,gg in sub.groupby("cve_base"):
    gg=gg.sort_values("date")
    f=gg.iloc[0]
    others=gg[gg["source"]!=f["source"]]
    if len(others):
        lead.append((f["source"], (others["date"].min()-f["date"]).days))
L=pd.DataFrame(lead,columns=["src","lead_days"])
print("\nmedian lead over next platform (days):\n", L.groupby("src")["lead_days"].agg(["count","median","mean"]).round(1).to_string())

print("\n### Can cvss_score be reconstructed from its components? (redundancy check)")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
cats=["attack_vector","attack_complexity","privileges_required","user_interaction","scope",
      "confidentiality_impact","integrity_impact","availability_impact"]
X=OrdinalEncoder().fit_transform(df[cats])
r2=cross_val_score(RandomForestRegressor(n_estimators=120,random_state=0,n_jobs=-1),X,df["cvss_score"],cv=5,scoring="r2")
print("  5-fold R^2 predicting cvss_score from the 8 components:", r2.round(4), "mean", r2.mean().round(4))
print("  unique component combinations:", df[cats].drop_duplicates().shape[0], "-> unique cvss values:", df["cvss_score"].nunique())
chk=df.groupby(cats)["cvss_score"].nunique()
print("  component-combos mapping to >1 cvss score:", (chk>1).sum(), "of", len(chk))

print("\n### Mastodon: why so low EPSS?")
mas=df[df["source"]=="Mastodon"]
print("  Mastodon CVE year mix:\n", mas["cve_base"].str[4:8].value_counts().sort_index().to_string())
print("  share 2025 CVEs -- Mastodon %.3f | rest %.3f" % (
    mas["cve_base"].str.startswith("CVE-2025").mean(), df[df.source!="Mastodon"]["cve_base"].str.startswith("CVE-2025").mean()))
print("  enriched share Mastodon %.3f | rest %.3f" % ((mas.epss_status=="enriched").mean(),(df[df.source!='Mastodon'].epss_status=='enriched').mean()))

print("\n### Source effect AFTER conditioning on CVE year (2025 CVEs only, CVE-level)")
cl=df.sort_values("date").groupby("cve_base",as_index=False).first()
c25=cl[cl["cve_base"].str.startswith("CVE-2025")]
print(c25.groupby("source")["epss_score"].agg(["count","mean","median"]).round(5).to_string())
groups=[gg["epss_score"].values for _,gg in c25.groupby("source") if len(gg)>=5]
H,p=stats.kruskal(*groups); print("  KW H=%.1f p=%.2e eps2=%.4f"%(H,p,(H-len(groups)+1)/(len(c25)-len(groups))))
c22=cl[cl["cve_base"].str[4:8].astype(int)<=2022]
print("\n<=2022 CVEs only, CVE-level:")
print(c22.groupby("source")["epss_score"].agg(["count","mean","median"]).round(5).to_string())
