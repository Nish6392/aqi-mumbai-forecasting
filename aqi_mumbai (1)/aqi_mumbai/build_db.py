#!/usr/bin/env python
"""Load readings + experiment outputs into SQLite and run the analysis queries.

  python build_db.py --data data/mumbai_pm25.csv --results results --db aqi.db --run-queries
"""
import argparse, re, sqlite3
from pathlib import Path
import pandas as pd
from aqi.data import ALIASES


def load_readings(path):
    df = pd.read_csv(path)
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    h = df.groupby(["station", pd.Grouper(key="datetime", freq="h")]).mean(numeric_only=True)
    full = pd.date_range(h.index.get_level_values(1).min(), h.index.get_level_values(1).max(), freq="h")
    # complete hourly grid per station so missing hours appear as NULL rows (coverage queries need them)
    h = h.reindex(pd.MultiIndex.from_product([h.index.get_level_values(0).unique(), full], names=["station", "datetime"])).reset_index()
    for c in ("pm25", "pm10", "ws", "temp"):          # optional columns: queries expect them, PM2.5-only data has NULLs
        if c not in h.columns:
            h[c] = float("nan")
    h["datetime"] = h["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return h


def run_queries(con, path="sql/queries.sql"):
    text = Path(path).read_text()
    for name, body in re.findall(r"-- name: (\w+)\n(.*?)(?=\n-- name:|\Z)", text, flags=re.S):
        try:
            print(f"\n### {name}\n" + pd.read_sql(body, con).head(15).to_string(index=False))
        except Exception as e:
            print(f"\n### {name}: skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--results", default="results")
    ap.add_argument("--db", default="aqi.db"); ap.add_argument("--run-queries", action="store_true")
    a = ap.parse_args()
    res = Path(a.results)
    con = sqlite3.connect(a.db)
    load_readings(a.data).to_sql("readings", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_read ON readings(station, datetime)")
    if (res / "stations_used.csv").exists():
        pd.read_csv(res / "stations_used.csv").to_sql("stations", con, if_exists="replace", index=False)
    if (res / "predictions_long.csv").exists():
        pl = pd.read_csv(res / "predictions_long.csv"); pl["origin"] = pd.to_datetime(pl["origin"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        pl.to_sql("predictions", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_pred ON predictions(model, horizon, station)")
    if (res / "results.csv").exists():
        pd.read_csv(res / "results.csv").to_sql("metrics", con, if_exists="replace", index=False)
    con.commit()
    print(f"[db] wrote {a.db}: " + ", ".join(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")))
    if a.run_queries: run_queries(con)


if __name__ == "__main__":
    main()
