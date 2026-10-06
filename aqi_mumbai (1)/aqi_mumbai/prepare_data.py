#!/usr/bin/env python
"""Turn raw downloads (a folder of CSVs) into the long CSV the pipeline expects.

  python prepare_data.py --raw raw_downloads/ --out data/mumbai_pm25.csv

Handles two common layouts:
  (a) wide, one file per station: columns like 'From Date', 'PM2.5', 'PM10', 'NO2', 'Temp', 'RH', 'WS'.
      Station name = 'station' column if present, else the file name.
  (b) OpenAQ-style long (v3 measurement export): columns location_id, location_name, parameter, value,
      datetimeUtc, datetimeLocal, latitude, longitude. Station name = "<location_name> (<location_id>)".
      Local time (datetimeLocal, wall-clock, tz dropped) is used by default so hour-of-day features line up
      with Mumbai's daily cycle; pass --time utc to use UTC instead. Negative readings (sensor artefacts) -> NaN.
Also writes <out dir>/stations.csv (station,lat,lon) from the latitude/longitude columns, so the pipeline does not
depend on the hard-coded approximate coordinates in aqi/stations.py.
Check the printed summary and adapt ALIASES in aqi/data.py if your export uses other column names."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from aqi.data import ALIASES

PARAM = {"pm2.5": "pm25", "pm25": "pm25", "pm10": "pm10", "no2": "no2", "so2": "so2", "co": "co", "o3": "o3",
         "temperature": "temp", "temp": "temp", "relativehumidity": "rh", "rh": "rh", "humidity": "rh",
         "wind_speed": "ws", "ws": "ws", "windspeed": "ws"}
KEEP = ["pm25", "pm10", "no2", "so2", "co", "o3", "temp", "rh", "ws"]


def _openaq_time(df, mode):
    """datetimeLocal like 2026-08-18T03:00:00+05:30 -> naive wall-clock time; datetimeUtc -> naive UTC."""
    if mode == "local" and "datetimelocal" in df.columns:
        return pd.to_datetime(df["datetimelocal"].astype(str).str[:19], errors="coerce")
    for c in ("datetimeutc", "date.utc", "utc"):
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)
    return None


def read_one(path, time_mode="local"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "parameter" in df.columns and "value" in df.columns:          # layout (b): OpenAQ long
        t = _openaq_time(df, time_mode)
        if t is not None:
            df["datetime"] = t
        if "location_name" in df.columns and "location_id" in df.columns:
            df["station"] = df["location_name"].astype(str) + " (" + df["location_id"].astype(str) + ")"
        elif "location_name" in df.columns:
            df["station"] = df["location_name"]
        elif "station" not in df.columns and "location" in df.columns:
            df["station"] = df["location"]
        elif "station" not in df.columns:
            df["station"] = path.stem.replace("_", " ")
        coords = None
        if {"latitude", "longitude"} <= set(df.columns):
            coords = df.groupby("station")[["latitude", "longitude"]].first().rename(columns={"latitude": "lat", "longitude": "lon"})
        df["parameter"] = df["parameter"].astype(str).str.lower().str.replace(" ", "").map(PARAM)
        df = df.dropna(subset=["parameter"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.loc[df["value"] < 0, "value"] = np.nan                    # negative concentrations are sensor artefacts
        df = df.pivot_table(index=["station", "datetime"], columns="parameter", values="value", aggfunc="mean").reset_index()
        df.attrs["coords"] = coords
        return df
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})
    for c in ("datetime", "date.utc", "datetimeutc", "utc"):
        if c in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={c: "datetime"})
    if "station" not in df.columns:                                  # layout (a), station from file name
        df["station"] = path.stem.replace("_", " ")
    df.columns = [PARAM.get(c, c) for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/mumbai_pm25.csv")
    ap.add_argument("--time", choices=["local", "utc"], default="local")
    a = ap.parse_args()
    files = sorted(Path(a.raw).glob("*.csv"))
    assert files, f"no CSV files in {a.raw}"
    parts = [read_one(f, a.time) for f in files]
    coord_tables = [d.attrs["coords"] for d in parts if d.attrs.get("coords") is not None]
    df = pd.concat(parts, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["datetime"])
    cols = ["station", "datetime"] + [c for c in KEEP if c in df.columns]
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[cols].sort_values(["station", "datetime"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    if coord_tables:
        sc = pd.concat(coord_tables).reset_index()
        sc = sc[~sc["station"].duplicated()]
        sc.to_csv(Path(a.out).with_name("stations.csv"), index=False)
        print(f"wrote {Path(a.out).with_name('stations.csv')}")
    print(f"wrote {a.out}: {len(df)} rows, {df['station'].nunique()} stations, "
          f"{df['datetime'].min()} -> {df['datetime'].max()}")
    print("missing fraction per column:\n", df[cols[2:]].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
