#!/usr/bin/env python3
"""日本の海岸線・県境ポリラインを assets/japan_outline.json に生成する(一度だけ実行)。

ソース: dataofjapan/land の japan.topojson (Natural Earth由来)
TopoJSONのarc(共有境界線)をそのまま折れ線として描けば、
ポリゴン組み立てなしで海岸線+県境が重複なく描ける。
使い方: python3 tools/build_outline.py [ローカルtopojsonパス]
"""
import json
import sys
import urllib.request
from pathlib import Path

SRC_URL = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.topojson"
OUT = Path(__file__).resolve().parent.parent / "assets" / "japan_outline.json"
DECIMATE = 3      # 3点に1点残す(表示用には十分)
ROUND = 3         # 小数3桁 ≈ 100m精度


def main():
    if len(sys.argv) > 1:
        topo = json.load(open(sys.argv[1]))
    else:
        req = urllib.request.Request(SRC_URL, headers={"User-Agent": "ramis-monitor/0.1"})
        with urllib.request.urlopen(req, timeout=60) as res:
            topo = json.loads(res.read())

    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    lines = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for i, (dx, dy) in enumerate(arc):
            x += dx
            y += dy
            if i % DECIMATE == 0 or i == len(arc) - 1:
                pts.append([round(x * sx + tx, ROUND), round(y * sy + ty, ROUND)])
        if len(pts) >= 2:
            lines.append(pts)

    OUT.write_text(json.dumps(lines, separators=(",", ":")))
    n_pts = sum(len(l) for l in lines)
    print(f"OK: {len(lines)} lines, {n_pts} points -> {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
