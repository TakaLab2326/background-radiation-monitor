#!/usr/bin/env python3
"""GitHub Pages用サイトを gha_data の最新ファイルから生成する(標準ライブラリのみ)。

GitHub Actions上で毎時実行される想定。ローカルでも動く。
  入力: gha_data/raw/ の最新スナップショット + gha_data/stations.csv.gz
        site/learn_viz.html (学習タブ: 別途生成した静的スナップショット)
  出力: _site/index.html (タブ統合ダッシュボード)
"""
import csv
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from viz_hub import build_hub

HERE = Path(__file__).resolve().parent
GHA = HERE / "gha_data"
SITE = HERE / "_site"
JST = timezone(timedelta(hours=9))


def latest_run():
    """最新の測定スナップショットと対応する欠測ファイルのパスを返す。"""
    files = sorted(GHA.glob("raw/*/[0-9]*.csv.gz"))
    files = [f for f in files if "_" not in f.name]
    assert files, "gha_data/raw にデータがない"
    f = files[-1]
    return f, f.with_name(f.stem.replace(".csv", "") + "_missing.csv.gz")


def read_gz(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    meas_path, miss_path = latest_run()
    stations = {r["id"]: r for r in read_gz(GHA / "stations.csv.gz")}
    coast = json.loads((HERE / "assets" / "japan_outline.json").read_text())
    now_iso = datetime.now(JST).isoformat(timespec="minutes")

    # 局ごとの最新行(1回分スナップショット内で最大の測定時刻)
    latest = {}
    for r in read_gz(meas_path):
        sid = r["station_id"]
        if sid not in latest or r["meas_datetime"] > latest[sid]["meas_datetime"]:
            latest[sid] = r

    map_pts, pts3d = [], []
    for sid, r in latest.items():
        s = stations.get(sid)
        dose = fnum(r["air_dose_rate"])
        if not s or not s.get("latitude") or dose is None or dose <= 0:
            continue
        lon, lat = round(float(s["longitude"]), 5), round(float(s["latitude"]), 5)
        name = (s.get("display_name") or sid).replace("　", " ")
        mt = r["meas_datetime"][5:16].replace("T", " ")
        map_pts.append([lon, lat, round(dose, 5), int(r["data_type"]), name, mt,
                        r.get("wind_direction") or "",
                        fnum(r.get("wind_speed"), -1), fnum(r.get("precipitation"), -1)])
        elev = fnum(s.get("elevation"))
        if elev is not None:
            pts3d.append([lon, lat, round(elev, 1), round(dose, 5), name, mt])

    missing = []
    for r in read_gz(miss_path):
        s = stations.get(r["station_id"])
        if s and s.get("latitude"):
            missing.append([round(float(s["longitude"]), 5), round(float(s["latitude"]), 5),
                            (s.get("display_name") or "").replace("　", " "),
                            (r.get("last_meas") or "")[:16].replace("T", " ")])

    def render(tpl_name, payload):
        tpl = (HERE / tpl_name).read_text()
        return tpl.replace("/*__PAYLOAD__*/null",
                           json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    latest_meas = max(p[5] for p in map_pts)
    map_html = render("map_template.html", {
        "generated_at": now_iso, "latest_meas": latest_meas,
        "stations": map_pts, "missing": missing, "coast": coast})
    html3d = render("map3d_template.html", {
        "generated_at": now_iso, "latest_meas": latest_meas,
        "points": pts3d, "coast": coast})
    learn_path = HERE / "site" / "learn_viz.html"
    learn_html = learn_path.read_text() if learn_path.exists() else \
        "<meta charset='utf-8'><p style='padding:2em'>学習タブは準備中</p>"

    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(build_hub([
        ("全国マップ", map_html), ("学習・精度", learn_html), ("3D 標高", html3d)]))
    print(f"OK: 局{len(map_pts)} 欠測{len(missing)} 3D{len(pts3d)} "
          f"({meas_path.name}) -> _site/index.html "
          f"({(SITE/'index.html').stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
