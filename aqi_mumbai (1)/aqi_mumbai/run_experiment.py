#!/usr/bin/env python
"""Run the full comparison and write tables + figures to --out.

  python run_experiment.py --data data/mumbai_pm25.csv --stations-csv data/stations.csv --train-frac 0.6 --val-frac 0.1 --event-cat 1
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aqi.data import load_long_csv, build_panel, make_task
from aqi import metrics as M
from aqi import models_classical as C

LABELS = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive", "ridge": "Ridge regression",
          "rf": "Random forest", "gbr": "Gradient boosting"}
ALL = list(LABELS)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--stations-csv");
    p.add_argument("--target", default="pm25")
    p.add_argument("--L", type=int, default=48); p.add_argument("--H", type=int, default=24)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    p.add_argument("--test-stride", type=int, help="default: 24 for several stations, 3 for a single station")
    p.add_argument("--train-frac", type=float, default=0.7); p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--min-input-cov", type=float, default=0.5, help="min observed share of the input window for val/test origins")
    p.add_argument("--event-cat", type=int, default=3, help="event class for F1: CPCB category >= this (3 = >90, 1 = >30 ug/m3)")
    p.add_argument("--focus", help="model key to test against the others in the bootstrap (default: best trained model by MAE)")
    p.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert a.data, "pass --data path/to/long.csv"
    frames, coords, names = load_long_csv(a.data, a.target, a.stations_csv)
    panel = build_panel(frames, coords, names, a.target, a.train_frac, a.val_frac)
    stride = a.test_stride or (24 if len(names) > 1 else 3)
    task = make_task(panel, a.L, a.H, a.horizons, test_stride=stride, min_input_cov=a.min_input_cov)
    print(f"[data] T={len(panel.index)} hours, N={len(names)} stations, features={panel.features}, "
          f"origins train/val/test = {len(task.origins['train'])}/{len(task.origins['val'])}/{len(task.origins['test'])}")

    o = task.origins["test"]
    assert len(o) > 0, "no valid test origins - lower --min-input-cov or --train-frac/--val-frac"
    true, mask = task.y_true(o), task.y_mask(o)
    preds, extras = {}, {}                                   # name -> list of (B,H,N) (one per seed)

    def run(name, fn):
        if name not in a.models:
            return
        try:
            r = fn()
        except ImportError as e:
            print(f"[skip] {LABELS[name]}: missing dependency ({e.name})"); return
        if isinstance(r, dict):
            extras.setdefault(name, []).append(r); r = r["preds"]
        preds.setdefault(name, []).append(r)

    run("persistence", lambda: C.persistence(task))
    run("seasonal_naive", lambda: C.seasonal_naive(task))
    run("ridge", lambda: C.ridge_model(task))
    run("rf", lambda: C.rf_model(task))
    run("gbr", lambda: C.gbr_model(task))

    # ------------------------------------------------ metrics
    rows = []
    for name, lst in preds.items():
        for seed, p in enumerate(lst):
            for r in M.score_model(p, true, mask, a.horizons, a.event_cat):
                rows.append(dict(model=LABELS[name], seed=seed, **r))
    long = pd.DataFrame(rows)
    long.to_csv(out / "metrics_long.csv", index=False)

    order = [LABELS[m] for m in ALL if LABELS[m] in set(long["model"])]
    per_seed = long.groupby(["model", "seed", "horizon"])[["MAE", "RMSE", "CatAcc", "F1_event"]].mean().reset_index()
    avg = per_seed.groupby(["model", "seed"])[["MAE", "RMSE", "CatAcc", "F1_event"]].mean().reset_index()
    summary = avg.groupby("model").agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"),
                                       CatAcc=("CatAcc", "mean"), F1_event=("F1_event", "mean")).reindex(order)
    base = summary.loc["Persistence", "MAE"] if "Persistence" in summary.index else np.nan
    summary["Skill_vs_persistence"] = 1 - summary["MAE"] / base
    summary["MAE_sd"] = summary["MAE_sd"].fillna(0.0)
    wide = per_seed.groupby(["model", "horizon"])["MAE"].mean().unstack().reindex(order)
    wide.columns = [f"MAE@{h}h" for h in wide.columns]
    table = wide.join(summary).reset_index()
    table.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(M.to_markdown(table))
    print("\n" + M.to_markdown(table))

    # ------------------------------------------------ long-format outputs for SQL
    o_time = panel.index[o]
    recs = []
    for name, lst in preds.items():
        pm = np.mean(lst, 0)
        for h in a.horizons:
            recs.append(pd.DataFrame({
                "origin": o_time.repeat(len(names)), "station": np.tile(names, len(o)), "horizon": h,
                "model": LABELS[name], "y_true": true[:, h - 1, :].reshape(-1),
                "y_pred": pm[:, h - 1, :].reshape(-1), "observed": mask[:, h - 1, :].reshape(-1)}))
    pl = pd.concat(recs)
    pl = pl[pl["observed"] & np.isfinite(pl["y_pred"])].drop(columns="observed")
    pl.to_csv(out / "predictions_long.csv", index=False)
    pd.DataFrame({"station": names, "lat": coords[:, 0], "lon": coords[:, 1]}).to_csv(out / "stations_used.csv", index=False)

    # ------------------------------------------------ significance (focus model vs the others)
    focus = a.focus if a.focus in preds else None
    if focus is None:
        cand = summary.drop(index=[LABELS["persistence"], LABELS["seasonal_naive"]], errors="ignore")["MAE"].dropna()
        focus = next((k for k in ALL if LABELS[k] == cand.idxmin()), None) if len(cand) else None
    if focus and len(preds) > 1:
        blk = max(7, 48 // stride)                           # ~2 days of consecutive origins per bootstrap block
        e_g = np.mean([M.per_origin_abs_err(p, true, mask, a.horizons) for p in preds[focus]], 0)
        rows = []
        for ref in ("ridge", "rf", "gbr", "seasonal_naive", "persistence"):
            if ref in preds and ref != focus:
                e_r = np.mean([M.per_origin_abs_err(p, true, mask, a.horizons) for p in preds[ref]], 0)
                d, lo, hi = M.paired_block_bootstrap(e_g, e_r, block=blk)
                rows.append(dict(focus=LABELS[focus], vs=LABELS[ref], mean_MAE_diff=d, ci95_lo=lo, ci95_hi=hi,
                                 significant=bool(hi < 0 or lo > 0)))
        if rows:
            b = pd.DataFrame(rows); b.to_csv(out / "bootstrap_focus_vs.csv", index=False)
            print(f"\n{LABELS[focus]} minus baseline MAE (negative = {LABELS[focus]} better), 95% block-bootstrap CI:\n"
                  + M.to_markdown(b, "{:.3f}"))

    # ------------------------------------------------ figures
    colors = plt.cm.tab10(np.arange(len(order)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, name in zip(colors, order):
        if name in wide.index:
            ax.plot(a.horizons, wide.loc[name].values, marker="o", label=name, color=c, lw=1.6)
    ax.set_xlabel("Forecast horizon (hours)"); ax.set_ylabel(f"MAE ({a.target}, ug/m3)")
    ax.set_title("Test error vs horizon"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "mae_vs_horizon.png", dpi=160); plt.close(fig)

    i0, n0 = len(o) // 2, int(np.argmax(np.nanmean(panel.X[:, :, 0], 0)))
    fig, ax = plt.subplots(figsize=(9, 4))
    t0 = o[i0]; hist = np.arange(-a.L, 0); fut = np.arange(a.H)
    ax.plot(hist, panel.X[t0 - a.L:t0, n0, 0], color="k", lw=1.2)
    ax.plot(fut, true[i0, :, n0], color="k", lw=2, label="Observed")
    for c, (key, lst) in zip(colors, [(k, preds[k]) for k in ALL if k in preds]):
        if key in ("persistence",):
            continue
        ax.plot(fut, lst[0][i0, :, n0], label=LABELS[key], lw=1.4)
    ax.axvline(0, color="grey", ls=":"); ax.set_xlabel("Hours relative to forecast origin"); ax.set_ylabel(a.target)
    ax.set_title(f"Example forecast - {names[n0]}, {panel.index[t0]:%Y-%m-%d %H:%M}"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "example_forecast.png", dpi=160); plt.close(fig)

    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
