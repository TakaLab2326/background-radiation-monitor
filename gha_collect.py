#!/usr/bin/env python3
"""GitHub Actions用の収集: 1回分をファイル(csv.gz)として gha_data/ に追記する。

ローカル(collect.py=SQLite)と同じ取得コアを使い、出力先だけをファイルにした版。
Python標準ライブラリのみ。出力:
  gha_data/raw/YYYY-MM-DD/HHMM.csv.gz          測定スナップショット(JST)
  gha_data/raw/YYYY-MM-DD/HHMM_missing.csv.gz  欠測局一覧
  gha_data/raw/YYYY-MM-DD/HHMM_radar.csv.gz    レーダー雨(>0の局のみ)
  gha_data/stations.csv.gz                     局マスタ(内容変化時のみ更新、標高列は保持)
ローカルDBへの取り込みは tools/sync_from_github.py。
"""
import csv
import gzip
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rain_radar
import ramis_core as rc

HERE = Path(__file__).resolve().parent
GHA = HERE / "gha_data"
JST = timezone(timedelta(hours=9))

MEAS_COLS = ["station_id", "data_type", "meas_datetime", "air_dose_rate",
             "counting_rate", "dust_beta_conc", "wind_direction", "wind_speed",
             "precipitation", "solar_amount", "missing_status"]
ST_COLS = ["id", "data_type", "display_name", "display_name_roman", "pref_code",
           "latitude", "longitude", "weather_sensor_flg", "riamoni_flg", "elevation"]


def gz_write(path, header, rows):
    """決定的なgzip(タイムスタンプ0)でCSVを書く。内容が同じなら再実行してもバイト一致。"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
            g.write(buf.getvalue().encode())


def load_stations():
    p = GHA / "stations.csv.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main():
    now = datetime.now(JST)
    day, hm = now.strftime("%Y-%m-%d"), now.strftime("%H%M")
    stations = load_stations()
    meas, missing, parts = [], [], []

    for dt in (1, 2, 3):
        try:
            recs = rc.fetch_snapshot(dt)
            for r in recs:
                if r.get("meas_datetime"):
                    meas.append([r["id"], dt, r["meas_datetime"], r.get("air_dose_rate"),
                                 r.get("counting_rate"), r.get("dust_beta_conc"),
                                 r.get("wind_direction_name"), r.get("wind_speed"),
                                 r.get("precipitation"), r.get("solar_amount"),
                                 r.get("missing_status")])
                old = stations.get(r["id"], {})
                stations[r["id"]] = {
                    "id": r["id"], "data_type": old.get("data_type") or dt,
                    "display_name": r.get("display_name") or old.get("display_name"),
                    "display_name_roman": r.get("display_name_roman") or old.get("display_name_roman"),
                    "pref_code": r.get("pref_code") or old.get("pref_code"),
                    "latitude": r.get("latitude") or old.get("latitude"),
                    "longitude": r.get("longitude") or old.get("longitude"),
                    "weather_sensor_flg": r.get("weather_sensor_flg") or old.get("weather_sensor_flg"),
                    "riamoni_flg": r.get("riamoni_flg") or old.get("riamoni_flg"),
                    "elevation": old.get("elevation", ""),   # 標高はローカルで付与した値を保持
                }
            parts.append(f"type{dt}:{len(recs)}")
        except Exception as e:
            parts.append(f"type{dt}:ERROR {e}")
        time.sleep(2)

    try:
        for r in rc.fetch_snapshot(0):
            missing.append([r["id"], r.get("missing_status"), r.get("meas_datetime")])
        parts.append(f"欠測:{len(missing)}")
    except Exception as e:
        parts.append(f"欠測:ERROR {e}")

    radar_rows = []
    try:
        coords = [(s["id"], float(s["latitude"]), float(s["longitude"]))
                  for s in stations.values() if s.get("latitude")]
        obs_iso, rows, _ = rain_radar.sample_stations(coords)
        radar_rows = [[sid, t, mmh] for sid, t, mmh in rows if mmh > 0]
        parts.append(f"レーダー雨:{len(radar_rows)}/{len(rows)}")
    except Exception as e:
        parts.append(f"レーダー雨:ERROR {e}")

    if not meas:
        sys.exit("全系統の取得に失敗")
    raw = GHA / "raw" / day
    gz_write(raw / f"{hm}.csv.gz", MEAS_COLS, meas)
    gz_write(raw / f"{hm}_missing.csv.gz", ["station_id", "missing_status", "last_meas"], missing)
    gz_write(raw / f"{hm}_radar.csv.gz", ["station_id", "obs_time", "mmh"], radar_rows)

    st_rows = [[s.get(c, "") for c in ST_COLS] for s in
               sorted(stations.values(), key=lambda x: x["id"])]
    tmp = GHA / "stations.new.csv.gz"
    gz_write(tmp, ST_COLS, st_rows)
    dst = GHA / "stations.csv.gz"
    if not dst.exists() or tmp.read_bytes() != dst.read_bytes():
        tmp.replace(dst)
    else:
        tmp.unlink()

    print(f"{now.isoformat(timespec='seconds')} {' '.join(parts)} -> raw/{day}/{hm}.csv.gz")


if __name__ == "__main__":
    main()
