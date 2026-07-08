#!/usr/bin/env python3
"""全局の標高(m)を取得して stations.elevation に保存する(一度だけ実行)。

ソース: Open-Meteo Elevation API (Copernicus DEM GLO-90ベース、無料)
100点ずつバッチ取得、1秒間隔でアクセス。
使い方: python3 tools/fetch_elevation.py
"""
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "ramis.sqlite3"
API = "https://api.open-meteo.com/v1/elevation"
BATCH = 100


def main():
    conn = sqlite3.connect(DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(stations)")]
    if "elevation" not in cols:
        conn.execute("ALTER TABLE stations ADD COLUMN elevation REAL")
        conn.commit()

    rows = conn.execute("""SELECT id, latitude, longitude FROM stations
                           WHERE latitude IS NOT NULL AND elevation IS NULL""").fetchall()
    print(f"標高未取得: {len(rows)}局")
    done = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        lats = ",".join(f"{r[1]:.5f}" for r in chunk)
        lons = ",".join(f"{r[2]:.5f}" for r in chunk)
        url = f"{API}?latitude={lats}&longitude={lons}"
        req = urllib.request.Request(url, headers={"User-Agent": "ramis-monitor/0.1"})
        elev = None
        for attempt in range(6):   # レート制限(429)は待てば解除される
            try:
                with urllib.request.urlopen(req, timeout=30) as res:
                    elev = json.loads(res.read())["elevation"]
                break
            except Exception as e:
                wait = 60 if "429" in str(e) else 10
                print(f"  batch {i//BATCH}: {e} — {wait}秒待って再試行({attempt+1}/5)", flush=True)
                time.sleep(wait)
        if elev is None:
            sys.exit(f"batch {i//BATCH} が5回失敗。時間を置いて再実行すれば続きから取得する")
        conn.executemany("UPDATE stations SET elevation=? WHERE id=?",
                         [(float(v), r[0]) for v, r in zip(elev, chunk)])
        conn.commit()
        done += len(chunk)
        if (i // BATCH) % 10 == 0:
            print(f"  {done}/{len(rows)}", flush=True)
        time.sleep(3)

    n, lo, hi = conn.execute(
        "SELECT COUNT(elevation), MIN(elevation), MAX(elevation) FROM stations").fetchone()
    print(f"完了: 標高あり{n}局  範囲 {lo:.0f}〜{hi:.0f} m")


if __name__ == "__main__":
    main()
