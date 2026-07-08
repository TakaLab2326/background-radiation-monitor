#!/usr/bin/env python3
"""蓄積DBの最新値から自己完結HTMLマップを生成する。

使い方: python3 make_map.py [--db data/ramis.sqlite3] [--out out/ramis_map.html]
"""
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def latest_snapshot(conn):
    """各局の最新測定行(線量率があるもの)を返す。"""
    rows = conn.execute("""
        SELECT m.station_id, MAX(m.meas_datetime) AS mt, m.air_dose_rate,
               m.wind_direction, m.wind_speed, m.precipitation,
               s.display_name, s.latitude, s.longitude, s.data_type
        FROM measurements m
        JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate IS NOT NULL
        GROUP BY m.station_id
    """).fetchall()
    out = []
    for sid, mt, dose, wdir, wspd, prec, name, lat, lon, dtype in rows:
        if lat is None or lon is None:
            continue
        out.append([
            round(lon, 5), round(lat, 5), round(dose, 5), dtype or 1,
            (name or sid).replace("　", " "),
            mt[5:16].replace("T", " "),          # "MM-DD HH:MM"
            wdir or "",
            wspd if wspd is not None else -1,
            prec if prec is not None else -1,
        ])
    return out


def missing_stations(conn):
    """最新の収集回で欠測扱いだった局(座標つき)。"""
    last = conn.execute("SELECT MAX(fetched_at) FROM missing_log").fetchone()[0]
    if last is None:
        return []
    rows = conn.execute("""
        SELECT s.longitude, s.latitude, s.display_name, ml.last_meas_datetime
        FROM missing_log ml JOIN stations s ON s.id = ml.station_id
        WHERE ml.fetched_at = ? AND s.longitude IS NOT NULL
    """, (last,)).fetchall()
    return [[round(lon,5), round(lat,5), (nm or "").replace("　"," "),
             (lm or "")[:16].replace("T"," ")] for lon, lat, nm, lm in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(HERE / "data" / "ramis.sqlite3"))
    ap.add_argument("--out", default=str(HERE / "out" / "ramis_map.html"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    stations = latest_snapshot(conn)
    missing = missing_stations(conn)
    coast = json.loads((HERE / "assets" / "japan_outline.json").read_text())
    assert stations, "DBに測定データがない。先に collect.py を実行すること"

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "latest_meas": max(s[5] for s in stations),
        "stations": stations,
        "missing": missing,
        "coast": coast,
    }
    tpl = (HERE / "map_template.html").read_text()
    marker = "/*__PAYLOAD__*/null"
    assert marker in tpl, "テンプレートのプレースホルダが見つからない"
    html = tpl.replace(marker, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"OK: 局{len(stations)} 欠測{len(missing)} -> {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
