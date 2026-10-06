"""
Mumbai PM2.5 forecasting - interactive dashboard (Streamlit).

Run from the project root folder:
    python -m streamlit run dashboard.py

It reads whatever is in data/ and results/, so after you re-run the experiment with more models
(Ridge, Random forest, ...) they appear here automatically.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "mumbai_pm25.csv"
RES = ROOT / "results"

# CPCB PM2.5 categories (ug/m3)
BANDS = [(0, 30, "Good", "#2e9e4f"), (30, 60, "Satisfactory", "#a3c644"), (60, 90, "Moderate", "#f2c12e"),
         (90, 120, "Poor", "#f08a24"), (120, 250, "Very Poor", "#d9432f")]
MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]

st.set_page_config(page_title="Mumbai PM2.5 Forecasting", page_icon="🌫️", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_data
def load_hourly():
    if not DATA_CSV.exists():
        return None
    df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    s = df.set_index("datetime")["pm25"].sort_index()
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full)


@st.cache_data
def load_results():
    res = pd.read_csv(RES / "results.csv") if (RES / "results.csv").exists() else None
    pred = (pd.read_csv(RES / "predictions_long.csv", parse_dates=["origin"])
            if (RES / "predictions_long.csv").exists() else None)
    return res, pred


def shade_bands(ax, ymax):
    for lo, hi, _name, col in BANDS:
        if lo < ymax:
            ax.axhspan(lo, min(hi, ymax), color=col, alpha=0.10, lw=0)


def category_of(v):
    for lo, hi, name, _ in BANDS:
        if v <= hi:
            return name
    return "Severe"



@st.cache_data
def valid_origins(_series):
    """Hours t where the previous 48 h have >= 36 readings and the next 24 h exist in the file."""
    cnt = _series.notna().astype(int).rolling(48).sum().shift(1)
    ok = (cnt >= 36) & (pd.Series(_series.index, index=_series.index) <= _series.index[-1] - pd.Timedelta(hours=23))
    return list(_series.index[ok.fillna(False)])


def baseline_forecasts(hist):
    """hist: last 48 hourly values (oldest first). Returns dict name -> array of next 24 hourly values."""
    hist = np.asarray(hist, dtype=float)
    persistence = np.full(24, hist[-1])
    seasonal = hist[-24:].copy()              # same hour yesterday
    return {"Persistence": persistence, "Seasonal naive": seasonal, "Average of both": (persistence + seasonal) / 2}

series = load_hourly()
results, preds = load_results()

# ------------------------------------------------------------------ header
st.title("🌫️ Mumbai PM2.5 Forecasting")
st.caption("Can a model forecast the next 1-24 hours of PM2.5 at one Mumbai station "
           "(OpenAQ location 8039) better than the simplest baseline, 'persistence' (= same as now)?")

if series is None:
    st.error("data/mumbai_pm25.csv not found. Run `python prepare_data.py --raw raw_downloads/ "
             "--out data/mumbai_pm25.csv` first.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 The data", "🏆 Model comparison", "🔮 Forecast explorer",
                                        "🎯 Try it yourself", "ℹ️ About & honesty notes"])

# ------------------------------------------------------------------ tab 1: data
with tab1:
    obs = series.dropna()
    c = st.columns(5)
    c[0].metric("Hours covered", f"{len(series):,}")
    c[1].metric("Readings available", f"{len(obs):,}", f"{len(obs) / len(series):.0%} of hours")
    c[2].metric("Average PM2.5", f"{obs.mean():.1f} µg/m³")
    c[3].metric("Maximum", f"{obs.max():.0f} µg/m³")
    c[4].metric("Hours above 30 (not 'Good')", f"{(obs > 30).mean():.0%}")

    st.subheader("Hourly PM2.5 over time")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ymax = max(60, float(obs.max()) * 1.1)
    shade_bands(ax, ymax)
    ax.plot(series.index, series.values, color="#1f4e79", lw=1.1)
    # mark long outages (>= 12 h with no data)
    is_nan = series.isna()
    grp = (is_nan != is_nan.shift()).cumsum()
    for _, g in series[is_nan].groupby(grp[is_nan]):
        if len(g) >= 12:
            ax.axvspan(g.index[0], g.index[-1], color="grey", alpha=0.25, lw=0)
    ax.set_ylim(0, ymax); ax.set_ylabel("PM2.5 (µg/m³)")
    ax.margins(x=0.01)
    st.pyplot(fig); plt.close(fig)
    st.caption("Coloured bands = CPCB air-quality categories (green = Good ≤ 30, yellow-green = Satisfactory 30-60). "
               "Grey blocks = outages of 12 hours or more where the sensor sent no data.")

    left, right = st.columns(2)
    with left:
        st.subheader("Typical day (average by hour)")
        by_hour = obs.groupby(obs.index.hour).mean()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(by_hour.index, by_hour.values, color="#3b7dbd")
        ax.set_xlabel("Hour of day (IST)"); ax.set_ylabel("Mean PM2.5")
        st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("How often each air-quality level occurs")
        counts = pd.Series({name: int(((obs > lo) & (obs <= hi)).sum()) if lo > 0 else int((obs <= hi).sum())
                            for lo, hi, name, _ in BANDS})
        counts = counts[counts > 0]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        cols = {name: col for _, _, name, col in BANDS}
        ax.barh(counts.index[::-1], (counts.values[::-1] / len(obs)) * 100, color=[cols[n] for n in counts.index[::-1]])
        ax.set_xlabel("% of readings")
        st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ tab 2: models
with tab2:
    if results is None:
        st.warning("results/results.csv not found. Run `run_experiment.py` first.")
    else:
        st.subheader("Scoreboard")
        best = results.sort_values("MAE").iloc[0]
        persistence = results[results["model"] == "Persistence"]
        st.success(f"Lowest average error: **{best['model']}** (MAE {best['MAE']:.2f} µg/m³)")
        if len(persistence) and best["model"] != "Persistence":
            st.info("A trained model beat the persistence baseline on average error.")
        elif len(results) > 2:
            st.warning("No trained model beat persistence on average error. "
                       "That is a legitimate result with only one station and ~6 weeks of data.")

        show = results.rename(columns={"MAE": "Mean MAE", "CatAcc": "Category accuracy",
                                       "F1_event": "F1 (PM2.5 > 30)", "Skill_vs_persistence": "Skill vs persistence"})
        show = show.drop(columns=[c for c in ["MAE_sd"] if c in show.columns])
        st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns if c != "model"})
                     .highlight_min(subset=["Mean MAE", "RMSE"], color="#c8f0c8"), use_container_width=True, hide_index=True)
        st.caption("MAE = average absolute error in µg/m³ (lower is better). Skill vs persistence > 0 means better than the baseline.")

        hcols = [c for c in results.columns if c.startswith("MAE@")]
        hours = [int(c[4:-1]) for c in hcols]
        st.subheader("Error vs how far ahead we forecast")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for i, row in results.iterrows():
            ax.plot(hours, [row[c] for c in hcols], marker="o", lw=2, label=row["model"],
                    color=MODEL_COLORS[i % len(MODEL_COLORS)])
        ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("MAE (µg/m³)")
        ax.set_xticks(hours); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig); plt.close(fig)

        have = set(results["model"])
        missing = [m for m in ["Ridge", "Random forest", "Gradient boosting"] if not any(m.lower() in h.lower() for h in have)]
        if missing:
            st.caption(f"Not in the results yet: {', '.join(missing)}. Re-run `run_all.bat` to add them - they will show up here automatically.")

# ------------------------------------------------------------------ tab 3: forecast explorer
with tab3:
    if preds is None or preds.empty:
        st.warning("results/predictions_long.csv not found. Run `run_experiment.py` first.")
    else:
        origins = sorted(preds["origin"].unique())
        models = list(dict.fromkeys(preds["model"]))
        c1, c2 = st.columns([1, 2])
        origin = c1.selectbox("Forecast made at (origin time)", origins,
                              format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y, %H:%M"), index=len(origins) // 2)
        chosen = c2.multiselect("Models to show", models, default=models)
        origin = pd.Timestamp(origin)

        hist = series.loc[origin - pd.Timedelta(hours=48): origin - pd.Timedelta(hours=1)]
        future = series.loc[origin: origin + pd.Timedelta(hours=23)]
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot((hist.index - origin) / pd.Timedelta(hours=1), hist.values, color="black", lw=1.3, label="Observed (past 48 h)")
        ax.plot((future.index - origin) / pd.Timedelta(hours=1), future.values, color="black", lw=2.4, label="Observed (actual next 24 h)")
        sub = preds[preds["origin"] == origin]
        for i, m in enumerate(models):
            if m not in chosen:
                continue
            d = sub[sub["model"] == m].sort_values("horizon")
            ax.plot(d["horizon"] - 1, d["y_pred"], marker="o", ms=7, lw=1.6, ls="--",
                    color=MODEL_COLORS[i % len(MODEL_COLORS)], label=f"{m} forecast")
        ax.axvline(-0.5, color="grey", ls=":")
        ax.set_xlabel("Hours relative to forecast start"); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=2, fontsize=8)
        st.pyplot(fig); plt.close(fig)
        st.caption("Left of the dotted line = what the model was allowed to see. Right = the next 24 hours. "
                   "Dots are the model's predictions at 1, 3, 6, 12 and 24 hours ahead.")

        st.subheader("Numbers for this forecast")
        piv = sub[sub["model"].isin(chosen)].pivot_table(index="horizon", columns="model", values="y_pred")
        truth = sub.groupby("horizon")["y_true"].first().rename("Actual")
        table = pd.concat([truth, piv], axis=1)
        table.index = [f"{h} h ahead" for h in table.index]
        for m in chosen:
            if m in table.columns:
                table[f"{m} error"] = (table[m] - table["Actual"]).abs()
        st.dataframe(table.round(1), use_container_width=True)

# ------------------------------------------------------------------ tab 4: try it yourself
with tab4:
    st.subheader("Try it yourself: forecast the next 24 hours")
    st.write("Pick a real moment from the dataset, or type in your own recent readings, and get a 24-hour forecast "
             "from the simple baseline methods used in this project.")
    mode = st.radio("Where should the last 48 hours of data come from?",
                    ["A real moment from the dataset", "My own numbers"], horizontal=True)

    starts = valid_origins(series)
    hist = actual = origin_t = None

    if mode == "A real moment from the dataset":
        if not starts:
            st.warning("Not enough continuous data to build a forecast window.")
        else:
            t = st.select_slider("Forecast start time (the forecast covers the 24 hours from here)",
                                 options=starts, value=starts[len(starts) // 2],
                                 format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y, %H:%M"))
            origin_t = pd.Timestamp(t)
            hist = series.loc[origin_t - pd.Timedelta(hours=48): origin_t - pd.Timedelta(hours=1)].ffill().bfill().values
            actual = series.loc[origin_t: origin_t + pd.Timedelta(hours=23)].values
    else:
        default_vals = ""
        if starts:
            t0 = pd.Timestamp(starts[len(starts) // 2])
            w = series.loc[t0 - pd.Timedelta(hours=48): t0 - pd.Timedelta(hours=1)].ffill().bfill().values
            default_vals = ", ".join(f"{v:.0f}" for v in w)
        raw = st.text_area("Last 48 hourly PM2.5 values, oldest first, most recent last (µg/m³). "
                           "At least 24 values. Separate with commas, spaces or new lines.",
                           value=default_vals, height=110)
        try:
            vals = [float(x) for x in re.split(r"[,\s;]+", raw.strip()) if x]
        except ValueError:
            vals = []
            st.error("Could not read the numbers. Use only digits, commas and spaces.")
        if len(vals) < 24:
            st.info("Enter at least 24 numbers to see a forecast.")
        elif min(vals) < 0:
            st.error("PM2.5 cannot be negative.")
        else:
            hist = np.array(vals[-48:])

    if hist is not None:
        fc = baseline_forecasts(hist)
        blend = fc["Average of both"]
        now = float(hist[-1])

        m = st.columns(4)
        m[0].metric("Latest reading", f"{now:.0f} µg/m³", category_of(now))
        m[1].metric("Peak forecast (average method)", f"{blend.max():.0f} µg/m³", category_of(float(blend.max())))
        m[2].metric("Hours forecast above 30", f"{int((blend > 30).sum())} of 24")
        m[3].metric("Best guess for hour 24", f"{blend[-1]:.0f} µg/m³")
        if (blend > 30).any():
            st.warning("Alert: PM2.5 is forecast to rise above 30 µg/m³ (no longer 'Good') at some point in the next 24 hours.")
        else:
            st.success("No alert: PM2.5 is forecast to stay at or below 30 µg/m³ ('Good') for the next 24 hours.")

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ymax = max(60.0, float(np.nanmax(hist)) * 1.2, float(max(v.max() for v in fc.values())) * 1.2)
        shade_bands(ax, ymax)
        hx = np.arange(-len(hist), 0)
        ax.plot(hx, hist, color="black", lw=1.4, label="Your last readings")
        if actual is not None:
            ax.plot(np.arange(24), actual, color="black", lw=2.4, label="What actually happened")
        styles = {"Persistence": ("#1f77b4", "--"), "Seasonal naive": ("#d62728", "--"), "Average of both": ("#2ca02c", "-")}
        for name, y in fc.items():
            col, ls = styles[name]
            ax.plot(np.arange(24), y, color=col, ls=ls, lw=2, label=name)
        ax.axvline(-0.5, color="grey", ls=":")
        ax.set_ylim(0, ymax); ax.set_xlabel("Hours relative to forecast start"); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=2, fontsize=8)
        st.pyplot(fig); plt.close(fig)

        table = pd.DataFrame({"Hour ahead": np.arange(1, 25), **{k: np.round(v, 1) for k, v in fc.items()}})
        table["Air quality (average method)"] = [category_of(float(v)) for v in blend]
        if actual is not None:
            table["Actual"] = actual
            ok = ~np.isnan(actual)
            if ok.any():
                sc = st.columns(len(fc))
                for col, (name, y) in zip(sc, fc.items()):
                    col.metric(f"{name}: average error", f"{np.abs(y[ok] - actual[ok]).mean():.1f} µg/m³")
        with st.expander("Show the 24-hour table"):
            st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("How the methods work: Persistence = the latest reading stays the same. Seasonal naive = same hour "
                   "yesterday. Average = the mean of the two. These are the baselines in the project; trained models "
                   "did not have a proven advantage over them on this small dataset.")

# ------------------------------------------------------------------ tab 5: about
with tab5:
    st.markdown("""
### What this project does
Every 3 hours the models look at the **last 48 hours** of PM2.5 and forecast the **next 24 hours**.
Scores are computed at 1, 3, 6, 12 and 24 hours ahead, only where the real value was actually measured.
All models are tested on the same later period they never trained on (chronological 60 / 10 / 30 split).

### Models
| Model | Idea |
|---|---|
| Persistence | Tomorrow looks like right now (the baseline to beat) |
| Seasonal naive | Same hour yesterday |
| Ridge regression, Random forest, Gradient boosting | scikit-learn models trained on this station's history (lags, rolling means, hour of day) |

### Be honest about the limits
- **One station, PM2.5 only, ~6 weeks** (18 Aug - 28 Sep 2026). No weather or neighbouring stations.
- **A 147-hour outage** (22-28 Sep) falls in the test period, so the test set is small.
- **Distribution shift:** the test period is dirtier than the training period (monsoon ending).
- Air is mostly 'Good' here, so 'Poor' (> 90) never happens. Alerts use PM2.5 > 30 instead.
- Test forecasts overlap heavily, so small differences between models are **not** proof that one is better.
""")
