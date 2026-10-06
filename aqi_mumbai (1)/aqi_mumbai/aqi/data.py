"""Data loading, imputation, scaling and chronological splitting.

Expected raw format (long CSV, one row per station-timestamp):
    station,datetime,pm25,pm10,no2,so2,co,o3,temp,rh,ws
Only `station`, `datetime` and the target column are required. Column names are
lower-cased; a few common aliases (PM2.5, timestamp, ...) are renamed.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .stations import STATIONS

ALIASES = {"pm2.5": "pm25", "pm2_5": "pm25", "pm 2.5": "pm25", "timestamp": "datetime", "date": "datetime",
           "from date": "datetime", "station_name": "station", "site": "station", "location": "station",
           "temperature": "temp", "at": "temp", "wind_speed": "ws", "humidity": "rh"}
CANDIDATE_FEATURES = ["pm10", "no2", "so2", "co", "o3", "temp", "rh", "ws"]


@dataclass
class Panel:
    index: pd.DatetimeIndex
    stations: list
    coords: np.ndarray      # (N,2) lat, lon
    features: list          # features[0] is the forecast target
    X_raw: np.ndarray       # (T,N,F) with NaN
    X: np.ndarray           # (T,N,F) imputed
    obs: np.ndarray         # (T,N) bool, target really observed
    tfeat: np.ndarray       # (T,4) hour/day-of-year sin/cos
    t_train: int
    t_val: int


def load_long_csv(path, target="pm25", stations_csv=None, min_coverage=0.5):
    df = pd.read_csv(path)
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    hourly = df.groupby(["station", pd.Grouper(key="datetime", freq="h")]).mean(numeric_only=True)
    feats = [c for c in [target] + CANDIDATE_FEATURES if c in hourly.columns]
    idx = pd.date_range(hourly.index.get_level_values(1).min(), hourly.index.get_level_values(1).max(), freq="h")
    frames = {f: hourly[f].unstack("station").reindex(idx) for f in feats}

    cov = frames[target].notna().mean()
    keep = [s for s in frames[target].columns if cov[s] >= min_coverage]
    dropped = sorted(set(frames[target].columns) - set(keep))
    if dropped:
        print(f"[data] dropping low-coverage stations: {dropped}")

    if stations_csv:
        c = pd.read_csv(stations_csv).set_index("station")[["lat", "lon"]]
        lookup = {name: (row["lat"], row["lon"]) for name, row in c.iterrows()}
    else:
        lookup = STATIONS
    coords, final = [], []
    for s in keep:
        hit = [k for k in lookup if k.lower() in s.lower() or s.lower() in k.lower()]
        if hit:
            coords.append(lookup[hit[0]]); final.append(s)
        else:
            print(f"[data] no coordinates for station '{s}' - dropped (add it to stations.csv)")
    frames = {f: v[final] for f, v in frames.items()}
    return frames, np.array(coords, dtype=float), final


def build_panel(frames, coords, stations, target="pm25", train_frac=0.7, val_frac=0.1):
    feats = [target] + [f for f in frames if f != target]
    idx = frames[target].index
    X_raw = np.stack([frames[f].reindex(idx)[stations].to_numpy(float) for f in feats], -1)
    T, N, F = X_raw.shape
    t_train, t_val = int(T * train_frac), int(T * (train_frac + val_frac))

    X = np.empty_like(X_raw)
    for f in range(F):
        d = pd.DataFrame(X_raw[:, :, f]).interpolate(limit=12, limit_direction="forward")
        d = d.fillna(d.iloc[:t_train].mean()).fillna(0.0)          # forward-only: no look-ahead; long gaps -> train mean
        X[:, :, f] = d.to_numpy()
    obs = ~np.isnan(X_raw[:, :, 0])

    h, doy = idx.hour.to_numpy(), idx.dayofyear.to_numpy()
    tfeat = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                      np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1)
    return Panel(idx, list(stations), np.asarray(coords, float), feats, X_raw, X, obs, tfeat, t_train, t_val)


@dataclass
class Task:
    panel: Panel
    L: int                  # input length (hours)
    H: int                  # forecast length (hours)
    horizons: list          # horizons reported in tables (1-indexed hours ahead)
    Xs: np.ndarray          # (T,N,F+4) scaled inputs, channel 0 = target
    mean: np.ndarray        # (N,F) train statistics
    std: np.ndarray
    origins: dict           # 'train' / 'val' / 'test' -> int arrays

    def y_true(self, o):
        """(B,H,N) target values in original units (imputed)."""
        return self.panel.X[o[:, None] + np.arange(self.H)[None], :, 0]

    def y_mask(self, o):
        return self.panel.obs[o[:, None] + np.arange(self.H)[None]]


def make_task(panel, L=48, H=24, horizons=(1, 3, 6, 12, 24), val_stride=3, test_stride=24, min_input_cov=0.5):
    """Forecast origin t: inputs are hours [t-L, t), targets are hours [t, t+H).
    Scaling uses train-period statistics only; splits are chronological.
    Val/test origins are kept only if at least `min_input_cov` of the last L hours of the target were really
    observed (mean-imputed inputs after long outages would otherwise be scored as if they were real history)."""
    T = panel.X.shape[0]
    with np.errstate(all="ignore"):
        mean = np.nan_to_num(np.nanmean(panel.X_raw[:panel.t_train], 0), nan=0.0)
        std = np.nan_to_num(np.nanstd(panel.X_raw[:panel.t_train], 0), nan=1.0)
    std[std < 1e-6] = 1.0
    Xs = (panel.X - mean) / std
    tf = np.broadcast_to(panel.tfeat[:, None, :], (T, Xs.shape[1], 4))
    Xs = np.concatenate([Xs, tf], -1).astype(np.float32)
    cs = np.vstack([np.zeros((1, panel.obs.shape[1])), np.cumsum(panel.obs, 0)])

    def ok(o):
        o = np.asarray(o)
        return o[(cs[o] - cs[o - L]).mean(1) / L >= min_input_cov] if len(o) else o

    origins = {
        "train": np.arange(L, panel.t_train - H + 1),
        "val": ok(np.arange(panel.t_train, panel.t_val - H + 1, val_stride)),
        "test": ok(np.arange(panel.t_val, T - H + 1, test_stride)),
    }
    return Task(panel, L, H, list(horizons), Xs, mean, std, origins)
