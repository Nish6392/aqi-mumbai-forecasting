"""Mumbai-region monitoring stations (CPCB / MPCB / SAFAR / IITM).

Coordinates are APPROXIMATE (rounded from memory of station locations).
Verify them against the station metadata that comes with your data download,
or override with a `stations.csv` (columns: station,lat,lon).
"""
import numpy as np

STATIONS = {
    "Bandra Kurla Complex": (19.0663, 72.8656),
    "Worli": (19.0176, 72.8172),
    "Colaba": (18.9067, 72.8147),
    "Mazgaon": (18.9667, 72.8442),
    "Chakala": (19.1110, 72.8600),
    "Malad West": (19.1860, 72.8480),
    "Borivali East": (19.2307, 72.8567),
    "Powai": (19.1176, 72.9060),
    "Kandivali East": (19.2050, 72.8700),
    "Navi Mumbai": (19.0330, 73.0297),
    "Vile Parle West": (19.1000, 72.8400),
    "Kurla": (19.0728, 72.8826),
    "Sion": (19.0390, 72.8619),
    "Chembur": (19.0522, 72.9005),
    "Deonar": (19.0500, 72.9200),
    "Byculla": (18.9791, 72.8330),
    "Bhandup West": (19.1450, 72.9370),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance_matrix(coords):
    """coords: (N,2) array of (lat, lon) -> (N,N) distances in km."""
    lat, lon = coords[:, 0], coords[:, 1]
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def gaussian_kernel(coords, sigma_km=6.0, eps=0.05, self_loops=False):
    """Thresholded Gaussian distance kernel (no self-loops by default)."""
    d = distance_matrix(coords)
    w = np.exp(-((d / sigma_km) ** 2))
    w[w < eps] = 0.0
    if not self_loops:
        np.fill_diagonal(w, 0.0)
    return w
