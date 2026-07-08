#!/usr/bin/env python3
"""GitHub Actionsが集めた gha_data/raw/ をローカルSQLiteへ取り込む。

git pull 後に実行すると、GHA収集分とローカルlaunchd収集分が1つのDBに合流する。
主キー(局ID+測定時刻)で重複排除されるため何度実行しても安全。
使い方: python3 tools/sync_from_github.py
"""
import csv
import gzip
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ramis_core as rc

GHA = HERE / "gha_data"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    conn = rc.open_db(HERE / "data" / "ramis.sqlite3")
    n_meas = n_rad = n_miss = 0
    for f in sorted(GHA.glob("raw/*/*.csv.gz")):
        with gzip.open(f, "rt") as fh:
            rows = list(csv.DictReader(fh))
        if f.name.endswith("_radar.csv.gz"):
            conn.execute("""CREATE TABLE IF NOT EXISTS radar_rain (
                station_id TEXT NOT NULL, obs_time TEXT NOT NULL, mmh REAL,
                PRIMARY KEY (station_id, obs_time))""")
            for r in rows:
                cur = conn.execute("INSERT OR IGNORE INTO radar_rain VALUES (?,?,?)",
                                   (r["station_id"], r["obs_time"], num(r["mmh"])))
                n_rad += cur.rowcount
        elif f.name.endswith("_missing.csv.gz"):
            fetched = f"{f.parent.name}T{f.name[:2]}:{f.name[2:4]}:00+09:00"
            for r in rows:
                cur = conn.execute("INSERT OR IGNORE INTO missing_log VALUES (?,?,?,?)",
                                   (r["station_id"], fetched, r["missing_status"],
                                    r["last_meas"]))
                n_miss += cur.rowcount
        else:
            fetched = f"{f.parent.name}T{f.name[:2]}:{f.name[2:4]}:00+09:00"
            for r in rows:
                cur = conn.execute("""INSERT OR IGNORE INTO measurements
                    (station_id, meas_datetime, air_dose_rate, counting_rate,
                     dust_beta_conc, wind_direction, wind_speed, precipitation,
                     solar_amount, missing_status, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (r["station_id"], r["meas_datetime"], num(r["air_dose_rate"]),
                     num(r["counting_rate"]), num(r["dust_beta_conc"]),
                     r["wind_direction"] or None, num(r["wind_speed"]),
                     num(r["precipitation"]), num(r["solar_amount"]),
                     r["missing_status"] or None, fetched))
                n_meas += cur.rowcount
    conn.commit()
    print(f"取り込み: 測定+{n_meas} レーダー+{n_rad} 欠測+{n_miss}")


if __name__ == "__main__":
    main()
