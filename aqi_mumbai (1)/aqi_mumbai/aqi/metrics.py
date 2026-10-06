import numpy as np
import pandas as pd

# CPCB PM2.5 categories (ug/m3): Good, Satisfactory, Moderate, Poor, Very Poor, Severe
PM25_BINS = [30, 60, 90, 120, 250]


def category(x):
    return np.digitize(x, PM25_BINS, right=True)


def score_model(pred, true, mask, horizons, event_cat=3):
    """pred/true/mask: (B,H,N). Returns one dict per horizon.
    Event = CPCB category >= event_cat (3 = Poor or worse, >90; 1 = above Good, >30)."""
    rows = []
    for h in horizons:
        p, t, m = pred[:, h - 1], true[:, h - 1], mask[:, h - 1]
        m = m & np.isfinite(p)
        if m.sum() == 0:
            continue
        e = (p - t)[m]
        cp, ct = category(p[m]), category(t[m])
        tp = ((cp >= event_cat) & (ct >= event_cat)).sum(); fp = ((cp >= event_cat) & (ct < event_cat)).sum()
        fn = ((cp < event_cat) & (ct >= event_cat)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append(dict(horizon=h, MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                         CatAcc=(cp == ct).mean(), F1_event=f1, n_events=int((ct >= event_cat).sum()), n=int(m.sum())))
    return rows


def per_origin_abs_err(pred, true, mask, horizons):
    """Mean absolute error per forecast origin over reported horizons/stations -> (B,)."""
    idx = [h - 1 for h in horizons]
    p, t, m = pred[:, idx], true[:, idx], mask[:, idx] & np.isfinite(pred[:, idx])
    err = np.where(m, np.abs(p - t), 0.0)
    return err.sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)


def per_station_mae(pred, true, mask, horizons):
    idx = [h - 1 for h in horizons]
    p, t, m = pred[:, idx], true[:, idx], mask[:, idx] & np.isfinite(pred[:, idx])
    err = np.where(m, np.abs(p - t), 0.0)
    return err.sum((0, 1)) / np.maximum(m.sum((0, 1)), 1)


def paired_block_bootstrap(e_a, e_b, n=2000, block=7, seed=0):
    """Difference in mean error (a - b) with 95% CI. Negative => a is better.
    Blocks of consecutive origins (default 7 ~ a week at daily stride) keep autocorrelation."""
    rng = np.random.default_rng(seed)
    d = np.asarray(e_a) - np.asarray(e_b)
    B = len(d)
    nb = int(np.ceil(B / block))
    stats = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, max(B - block + 1, 1), nb)
        sample = np.concatenate([d[s:s + block] for s in starts])[:B]
        stats[i] = sample.mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return d.mean(), lo, hi


def to_markdown(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)
