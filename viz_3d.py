#!/usr/bin/env python3
"""標高つき3D地図マップのHTML生成。

各局を(経度, 緯度, 標高)の3D空間に置き、線量率で色分けする。
実行: python3 viz_3d.py  (標準ライブラリのみ)
出力: out/ramis_3d.html
前提: tools/fetch_elevation.py 実行済み(stations.elevationあり)
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    conn = sqlite3.connect(HERE / "data" / "ramis.sqlite3")
    rows = conn.execute("""
        SELECT m.station_id, MAX(m.meas_datetime), m.air_dose_rate,
               s.display_name, s.latitude, s.longitude, s.elevation
        FROM measurements m JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate > 0 AND s.latitude IS NOT NULL
        GROUP BY m.station_id
    """).fetchall()
    pts, n_noelev = [], 0
    for sid, mt, dose, name, lat, lon, elev in rows:
        if elev is None:
            n_noelev += 1
            continue
        pts.append([round(lon, 5), round(lat, 5), round(elev, 1), round(dose, 5),
                    (name or sid).replace("　", " "), mt[5:16].replace("T", " ")])
    assert pts, "標高つきの局がない。tools/fetch_elevation.py を先に実行すること"
    if n_noelev:
        print(f"注意: 標高未取得のため除外 {n_noelev}局")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "latest_meas": max(p[5] for p in pts),
        "points": pts,
        "coast": json.loads((HERE / "assets" / "japan_outline.json").read_text()),
    }
    tpl = (HERE / "map3d_template.html").read_text()
    marker = "/*__PAYLOAD__*/null"
    assert marker in tpl
    out = HERE / "out" / "ramis_3d.html"
    out.write_text(tpl.replace(marker, json.dumps(payload, ensure_ascii=False,
                                                  separators=(",", ":"))))
    print(f"OK: {len(pts)}局 -> {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
