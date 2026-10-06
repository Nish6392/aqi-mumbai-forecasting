import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"figure.dpi":150})
OUT="docs/img/"
C={"Persistence":"#1f77b4","Seasonal naive":"#ff7f0e","Ridge regression":"#2ca02c","Random forest":"#d62728","Gradient boosting":"#9467bd"}

df=pd.read_csv("data/mumbai_pm25.csv",parse_dates=["datetime"])
s=df.set_index("datetime")["pm25"].sort_index()
full=pd.date_range(s.index.min(),s.index.max(),freq="h"); s=s.reindex(full)
n=len(s); t1=full[int(n*.6)]; t2=full[int(n*.7)]
print(n, s.notna().mean(), t1, t2, s.max(), s.min(), s.mean())

# 1 hourly series with split
fig,ax=plt.subplots(figsize=(10,3.6))
ax.plot(s.index,s.values,lw=.8,color="#333")
ax.axvspan(s.index[0],t1,color="#2ca02c",alpha=.07,label="Train (60%)")
ax.axvspan(t1,t2,color="#ff7f0e",alpha=.10,label="Validation (10%)")
ax.axvspan(t2,s.index[-1],color="#d62728",alpha=.07,label="Test (30%)")
ax.axhline(30,color="#2e9e4f",ls="--",lw=1); ax.text(s.index[2],31,"CPCB 'Good' limit (30)",color="#2e9e4f",fontsize=8)
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title("Hourly PM2.5 at Mumbai (OpenAQ 8039) with chronological split"); ax.legend(ncol=3,loc="upper left",bbox_to_anchor=(0,.9),fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(OUT+"01_hourly_series.png"); plt.close()

# 2 hour of day + distribution
fig,axs=plt.subplots(1,2,figsize=(10,3.4))
h=s.groupby(s.index.hour).mean(); sd=s.groupby(s.index.hour).std()
axs[0].plot(h.index,h.values,marker="o",color="#1f77b4"); axs[0].fill_between(h.index,h-sd/2,h+sd/2,alpha=.15)
axs[0].set_xlabel("Hour of day (IST)"); axs[0].set_ylabel("Mean PM2.5 (µg/m³)"); axs[0].set_title("Typical day")
axs[1].hist(s.dropna(),bins=25,color="#888",edgecolor="white"); axs[1].axvline(s.mean(),color="#d62728",ls="--"); 
axs[1].set_xlabel("PM2.5 (µg/m³)"); axs[1].set_ylabel("Hours"); axs[1].set_title("Distribution of observed hours")
fig.tight_layout(); fig.savefig(OUT+"02_daily_profile_distribution.png"); plt.close()

# 3 MAE vs horizon
r=pd.read_csv("results/results.csv")
hs=[1,3,6,12,24]
fig,ax=plt.subplots(figsize=(7,4))
for _,row in r.iterrows():
    ax.plot(hs,[row[f"MAE@{x}h"] for x in hs],marker="o",lw=2.4 if row.model=="Ridge regression" else 1.4,color=C[row.model],label=row.model)
ax.set_xlabel("Forecast horizon (hours)"); ax.set_ylabel("Test MAE (µg/m³)"); ax.set_title("Error vs forecast horizon"); ax.legend(fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(OUT+"03_mae_vs_horizon.png"); plt.close()

# 4 mean MAE bar + skill
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
o=r.sort_values("MAE")
axs[0].barh(o.model,o.MAE,color=[C[m] for m in o.model]); axs[0].invert_yaxis()
for i,v in enumerate(o.MAE): axs[0].text(v+.08,i,f"{v:.2f}",va="center",fontsize=8)
axs[0].set_xlabel("Mean MAE over horizons (µg/m³)"); axs[0].set_title("Overall error (lower is better)")
sk=r.sort_values("Skill_vs_persistence",ascending=False)
axs[1].barh(sk.model,sk.Skill_vs_persistence,color=["#2ca02c" if v>0 else "#c0392b" for v in sk.Skill_vs_persistence]); axs[1].invert_yaxis(); axs[1].axvline(0,color="k",lw=.8)
axs[1].set_xlabel("Skill vs persistence"); axs[1].set_title("Skill score (>0 beats persistence)")
fig.tight_layout(); fig.savefig(OUT+"04_model_scoreboard.png"); plt.close()

# 5 bootstrap forest
b=pd.read_csv("results/bootstrap_focus_vs.csv")
fig,ax=plt.subplots(figsize=(7,3))
for i,row in b.iterrows():
    col="#2ca02c" if row.significant else "#888"
    ax.plot([row.ci95_lo,row.ci95_hi],[i,i],color=col,lw=3); ax.plot(row.mean_MAE_diff,i,"o",color=col)
ax.set_yticks(range(len(b))); ax.set_yticklabels("Ridge vs "+b.vs); ax.invert_yaxis(); ax.axvline(0,color="k",lw=.8)
ax.set_xlabel("Mean MAE difference (µg/m³); negative = Ridge better"); ax.set_title("Paired block-bootstrap 95% CI (green = excludes 0)")
fig.tight_layout(); fig.savefig(OUT+"05_bootstrap_ci.png"); plt.close()

# 6 example forecasts, all models
p=pd.read_csv("results/predictions_long.csv",parse_dates=["origin"])
orig=pd.Timestamp("2026-09-19 20:00")
if orig not in set(p.origin): orig=p.origin.iloc[len(p)//2]
sub=p[p.origin==orig]
fig,ax=plt.subplots(figsize=(8,3.8))
past=s[orig-pd.Timedelta(hours=48):orig]
ax.plot(past.index,past.values,color="k",lw=1.5,label="Observed")
fut=s[orig:orig+pd.Timedelta(hours=24)]; ax.plot(fut.index,fut.values,color="k",lw=1.5,ls=":")
for m,g in sub.groupby("model"):
    g=g.sort_values("horizon"); ax.plot(orig+pd.to_timedelta(g.horizon,unit="h"),g.y_pred,marker="o",ms=5,lw=1,color=C[m],label=m)
ax.axvline(orig,color="grey",ls="--"); ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title(f"Example forecast origin: {orig:%Y-%m-%d %H:%M}"); ax.legend(fontsize=7,ncol=3,frameon=False)
fig.tight_layout(); fig.savefig(OUT+"06_example_forecast.png"); plt.close()
print(sorted(set(p.origin))[:3], len(set(p.origin)), orig in set(p.origin), sub.shape)
