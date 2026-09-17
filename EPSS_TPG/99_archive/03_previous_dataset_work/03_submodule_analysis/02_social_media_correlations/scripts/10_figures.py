import pandas as pd, numpy as np, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/SMP corelation analysis"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(OUT, "tables");  os.makedirs(TAB, exist_ok=True)
D = "/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files"

S1,S2,S3,S4,S5 = "#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})
def finish(ax, title, sub=None):
    ax.set_title(title, loc="left", pad=14 if sub else 8)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

df = pd.read_csv(os.path.join(D,"gemma_combined_summ.csv"), low_memory=False)
df["cve_base"]=df["cve"].str.extract(r'^(CVE-\d{4}-\d+)')[0]
df["date"]=pd.to_datetime(df["date_posted"])
df["cve_year"]=df["cve_base"].str[4:8].astype(int)
cl = df.sort_values(["date","cve"]).groupby("cve_base", as_index=False).first()

# ---- FIG 1: EPSS distribution on log10 axis
fig, ax = plt.subplots(figsize=(7.5,3.8))
bins = np.logspace(np.log10(df.epss_score.min()), 0, 45)
ax.hist(df.epss_score, bins=bins, color=S1, edgecolor=SURF, linewidth=0.6)
ax.set_xscale("log"); ax.grid(axis="y"); ax.set_axisbelow(True)
ax.set_xlabel("EPSS score (log scale)"); ax.set_ylabel("posts")
ax.axvline(df.epss_score.median(), color=INK2, ls="--", lw=1.2)
ax.text(df.epss_score.median()*1.15, ax.get_ylim()[1]*0.85, f"median {df.epss_score.median():.5f}",
        color=INK2, fontsize=9)
finish(ax, "EPSS is extremely right-skewed and bimodal",
       "9,218 posts · 51.9% below 0.001 · a second mass of 540 posts at ≥ 0.90")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig1_epss_distribution.png")); plt.close(fig)

# ---- FIG 2: monthly volume by platform (Telegram excluded)
nt = df[df.source!="Telegram"].copy()
mix = pd.crosstab(nt["date"].dt.to_period("M"), nt["source"])
mix = mix.loc["2024-09":]
order = ["Mastodon","Reddit","BleepingComputer","HackerNews","ExploitDB"]
cols  = [S1,S2,S3,S4,S5]
fig, ax = plt.subplots(figsize=(8.5,4.0))
bot = np.zeros(len(mix))
x = np.arange(len(mix))
for name,c in zip(order, cols):
    v = mix[name].values
    ax.bar(x, v, bottom=bot, color=c, width=0.72, label=name, linewidth=1.6, edgecolor=SURF)
    bot += v
ax.set_xticks(x); ax.set_xticklabels([str(p) for p in mix.index], rotation=45, ha="right", fontsize=8)
ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_ylabel("posts")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
finish(ax, "Collection is a burst, not a stream (Telegram excluded)",
       "82.9% of non-Telegram posts fall in 2025; each platform occupies its own window")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig2_monthly_volume_by_platform.png")); plt.close(fig)

# ---- FIG 3: median EPSS by CVE year (CVE-level)
yr = cl[cl.cve_year>=2016].groupby("cve_year")["epss_score"].agg(["count","median"])
fig, ax = plt.subplots(figsize=(7.5,3.8))
ax.bar(yr.index.astype(str), yr["median"], color=S1, width=0.7)
ax.set_yscale("log"); ax.grid(axis="y"); ax.set_axisbelow(True)
ax.set_ylabel("median EPSS (log scale)"); ax.set_xlabel("year in CVE identifier")
for xx,(n,v) in enumerate(zip(yr["count"], yr["median"])):
    ax.text(xx, v*1.25, f"n={n}", ha="center", fontsize=7.5, color=INK2)
finish(ax, "CVE vintage dominates EPSS",
       "Unique CVEs (n=5,692). Spearman(year, EPSS) = −0.59; the corpus is almost all new CVEs")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig3_epss_by_cve_year.png")); plt.close(fig)

# ---- FIG 4: Simpson's paradox, CVSS severity vs EPSS
bins=[-.01,3.9,6.9,8.9,10.0]; labs=["Low","Medium","High","Critical"]
row = df.assign(s=pd.cut(df.cvss_score,bins,labels=labs)).groupby("s",observed=True)["epss_score"].median()
cve = cl.assign(s=pd.cut(cl.cvss_score,bins,labels=labs)).groupby("s",observed=True)["epss_score"].median()
x=np.arange(4); w=0.36
fig, ax = plt.subplots(figsize=(7.0,3.8))
ax.bar(x-w/2, row.values, w, color=S1, label="per post (n=9,218)")
ax.bar(x+w/2, cve.values, w, color=S2, label="per unique CVE (n=5,692)")
for xx,v in zip(x-w/2,row.values): ax.text(xx,v*1.04,f"{v:.5f}",ha="center",fontsize=7.5,color=INK2)
for xx,v in zip(x+w/2,cve.values): ax.text(xx,v*1.04,f"{v:.5f}",ha="center",fontsize=7.5,color=INK2)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.grid(axis="y"); ax.set_axisbelow(True)
ax.set_ylabel("median EPSS"); ax.legend(frameon=False, fontsize=9)
finish(ax, "CVSS severity does not order EPSS",
       "De-duplication flips the sign: Spearman +0.066 per post, −0.082 per unique CVE")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig4_cvss_vs_epss_simpson.png")); plt.close(fig)

# ---- FIG 5: post-text reuse by platform
h = df.social_media_post.fillna("").astype(str)
code = pd.factorize(h)[0]
ncve = pd.DataFrame({"c":code,"cve":df.cve_base}).groupby("c")["cve"].nunique()
shared = set(ncve.index[ncve>1])
reuse = df.assign(x=pd.Series(code, index=df.index).isin(shared)).groupby("source")["x"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7.0,3.4))
ax.barh(reuse.index, reuse.values*100, color=S2, height=0.62)
for i,v in enumerate(reuse.values*100): ax.text(v+1, i, f"{v:.1f}%", va="center", fontsize=9, color=INK2)
ax.set_xlim(0,105); ax.grid(axis="x"); ax.set_axisbelow(True)
ax.set_xlabel("% of a platform's posts whose text is attached to more than one CVE")
finish(ax, "One post text, many CVE labels",
       "35.8% of all rows; a single Hacker News page dump carries 68 different CVEs")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig5_post_text_reuse.png")); plt.close(fig)

print("fig4 row medians:", row.round(5).to_dict())
print("fig4 cve medians:", cve.round(5).to_dict())
print("figures written:", sorted(os.listdir(FIG)))

# ---- supporting tables
df.groupby("source").agg(posts=("cve","size"), unique_cves=("cve_base","nunique"),
    mean_epss=("epss_score","mean"), median_epss=("epss_score","median"),
    mean_cvss=("cvss_score","mean"),
    pct_enriched=("epss_status", lambda s:(s=="enriched").mean()*100),
    first_post=("date","min"), last_post=("date","max")).round(5).to_csv(os.path.join(TAB,"platform_profile.csv"))
cl.groupby("cve_year")["epss_score"].agg(["count","mean","median"]).round(5).to_csv(os.path.join(TAB,"epss_by_cve_year.csv"))
pd.crosstab(nt["date"].dt.to_period("M"), nt["source"]).to_csv(os.path.join(TAB,"monthly_volume_no_telegram.csv"))
print("tables written:", sorted(os.listdir(TAB)))
