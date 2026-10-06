"""Classical models only (numpy / pandas / scikit-learn): persistence, seasonal naive, Ridge regression,
Random Forest, Gradient Boosting (all direct multi-horizon on lag / rolling / calendar features).
Every function returns predictions for task.origins['test'] with shape (B,H,N) in original units."""
import warnings
import numpy as np
from .stations import gaussian_kernel

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
LAG_MAX = max(LAGS)


def persistence(task):
    o = task.origins["test"]
    Y = task.panel.X[:, :, 0]
    return np.repeat(Y[o - 1][:, None, :], task.H, axis=1)


def seasonal_naive(task):
    """Value at the same hour of the most recent fully observed day."""
    o = task.origins["test"]
    Y = task.panel.X[:, :, 0]
    k = np.arange(task.H)
    back = 24 * np.ceil((k + 1) / 24).astype(int)
    return Y[o[:, None] + k[None] - back[None]]


# ------------------------------------------------------------------ scikit-learn models
def _cum(a):
    return np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, 0)])


def build_xgb_features(task, o, h, Wn=None, cache=None):
    """Feature matrix (B*N, F) for origins `o` and horizon h (1-indexed) - only information available at t.
    Includes the lagged target, rolling stats, latest co-pollutants/meteorology, a spatial-neighbour
    feature (distance-weighted neighbours' PM), station id and calendar features of the target hour."""
    P = task.panel
    Y = P.X[:, :, 0]
    B, N = len(o), Y.shape[1]
    if Wn is None:
        Wn = gaussian_kernel(P.coords)
        Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-9)
    cs, cs2 = (cache or (_cum(Y), _cum(Y ** 2)))
    F, names = [], []
    for l in LAGS:
        F.append(Y[o - l]); names.append(f"lag{l}")
    for w in (6, 24, 72):
        F.append((cs[o] - cs[o - w]) / w); names.append(f"mean{w}")
    m24 = (cs[o] - cs[o - 24]) / 24
    F.append(np.sqrt(np.maximum((cs2[o] - cs2[o - 24]) / 24 - m24 ** 2, 0))); names.append("std24")
    for f in range(1, P.X.shape[2]):
        F.append(P.X[o - 1, :, f]); names.append(P.features[f])
    F.append(Y[o - 1] @ Wn.T); names.append("nb_lag1")
    F.append(Y[o - 3] @ Wn.T); names.append("nb_lag3")
    F.append(Y[o - 1] - Y[o - 4]); names.append("delta3")
    tgt = P.index[o + h - 1]
    for nm, v in [("hsin", np.sin(2 * np.pi * tgt.hour / 24)), ("hcos", np.cos(2 * np.pi * tgt.hour / 24)),
                  ("dow", tgt.dayofweek.to_numpy()), ("month", tgt.month.to_numpy())]:
        F.append(np.repeat(np.asarray(v, float)[:, None], N, 1)); names.append(nm)
    F.append(np.tile(np.arange(N), (B, 1)).astype(float)); names.append("station")
    return np.stack(F, -1).reshape(B * N, -1), names


def sklearn_model(task, make_model, name):
    """One scikit-learn regressor per reported horizon (direct strategy), fitted on the training origins.
    Only rows whose target was really observed are used for fitting. Non-reported horizons stay NaN."""
    P = task.panel
    Y = P.X[:, :, 0]
    N = Y.shape[1]
    cache = (_cum(Y), _cum(Y ** 2))
    Wn = gaussian_kernel(P.coords); Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-9)
    tr = task.origins["train"]; tr = tr[tr >= LAG_MAX]
    te = task.origins["test"]
    out = np.full((len(te), task.H, N), np.nan)
    for h in task.horizons:
        Xtr, _ = build_xgb_features(task, tr, h, Wn, cache)
        ytr = Y[tr + h - 1].reshape(-1); mtr = P.obs[tr + h - 1].reshape(-1)
        Xte, _ = build_xgb_features(task, te, h, Wn, cache)
        mdl = make_model()
        mdl.fit(Xtr[mtr], ytr[mtr])
        out[:, h - 1, :] = mdl.predict(Xte).reshape(len(te), N)
        print(f"[{name}] horizon {h}h done")
    return np.clip(out, 0, None)


def ridge_model(task, alpha=10.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return sklearn_model(task, lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha)), "ridge")


def rf_model(task, n_estimators=300, min_samples_leaf=5):
    from sklearn.ensemble import RandomForestRegressor
    return sklearn_model(task, lambda: RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.5,
        random_state=0, n_jobs=-1), "random forest")


def gbr_model(task, max_iter=200, lr=0.05, max_depth=4):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return sklearn_model(task, lambda: HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=lr, max_depth=max_depth, min_samples_leaf=20,
        early_stopping=False, random_state=0), "gradient boosting")
