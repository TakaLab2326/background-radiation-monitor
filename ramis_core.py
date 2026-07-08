#!/usr/bin/env python3
"""RAMIS(放射線モニタリング情報共有・公表システム) 公開APIクライアント + SQLite蓄積コア。

Python標準ライブラリのみで動く(conda環境不問)。
APIは公式ドキュメントの無い内部APIのため、仕様変更で壊れる可能性がある。
その場合はまず fetch_snapshot() の生レスポンスを確認すること。

データ出典: 放射線モニタリング情報共有・公表システム(原子力規制委員会)
https://www.ramis.nra.go.jp/
"""
import gzip
import io
import json
import sqlite3
import time
import urllib.request
from datetime import datetime

API_BASE = "https://www.ramis.nra.go.jp/api/v1"
MAP_MEANS_URL = API_BASE + "/map/map-means-data-public"

# data_typeの意味は実データからの推定(公式定義は未公開)
DATA_TYPES = {
    0: "欠測局一覧",              # 全測定値がnull。故障・停止中の局リスト
    1: "モニタリングポスト",       # 固定局。気象センサ付きが約756局
    2: "リアルタイム線量測定システム",  # 福島中心のリアモニ
    3: "その他(排気筒モニタ等か)",   # 少数局
}

USER_AGENT = "ramis-monitor/0.1 (personal research; contact via NRA site form if issue)"


def fetch_snapshot(data_type: int, timeout: int = 60, retries: int = 1):
    """指定data_typeの現在スナップショットを取得して局レコードのリストを返す。"""
    url = f"{MAP_MEANS_URL}?data_type={data_type}"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    })
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                raw = res.read()
                if res.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
            payload = json.loads(raw)
            return payload["data"]
        except Exception as e:  # ネットワーク断・一時エラーは1回だけ再試行
            last_err = e
            if attempt < retries:
                time.sleep(5)
    raise RuntimeError(f"fetch failed data_type={data_type}: {last_err}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    id                 TEXT PRIMARY KEY,
    data_type          INTEGER,
    display_name       TEXT,
    display_name_roman TEXT,
    pref_code          TEXT,
    latitude           REAL,
    longitude          REAL,
    weather_sensor_flg TEXT,
    riamoni_flg        TEXT,
    elevation          REAL,
    first_seen         TEXT,
    last_seen          TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
    station_id      TEXT NOT NULL,
    meas_datetime   TEXT NOT NULL,
    air_dose_rate   REAL,
    counting_rate   REAL,
    dust_beta_conc  REAL,
    wind_direction  TEXT,
    wind_speed      REAL,
    precipitation   REAL,
    solar_amount    REAL,
    missing_status  TEXT,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (station_id, meas_datetime)
);
CREATE TABLE IF NOT EXISTS missing_log (
    station_id     TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    missing_status TEXT,
    last_meas_datetime TEXT,
    PRIMARY KEY (station_id, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_meas_time ON measurements (meas_datetime);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_snapshot(conn: sqlite3.Connection, records, data_type: int, fetched_at: str):
    """測定値スナップショットを保存。戻り値 (局数, 新規測定行数)。

    10分ごとに実行しても、meas_datetimeが変わっていない局は
    INSERT OR IGNORE により重複しない。
    """
    new_rows = 0
    for r in records:
        conn.execute(
            """INSERT INTO stations (id, data_type, display_name, display_name_roman,
                   pref_code, latitude, longitude, weather_sensor_flg, riamoni_flg,
                   first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen""",
            (r["id"], data_type, r.get("display_name"), r.get("display_name_roman"),
             r.get("pref_code"), r.get("latitude"), r.get("longitude"),
             r.get("weather_sensor_flg"), r.get("riamoni_flg"), fetched_at, fetched_at))
        if r.get("meas_datetime") is None:
            continue
        cur = conn.execute(
            """INSERT OR IGNORE INTO measurements
               (station_id, meas_datetime, air_dose_rate, counting_rate, dust_beta_conc,
                wind_direction, wind_speed, precipitation, solar_amount,
                missing_status, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (r["id"], r["meas_datetime"], r.get("air_dose_rate"), r.get("counting_rate"),
             r.get("dust_beta_conc"), r.get("wind_direction_name"), r.get("wind_speed"),
             r.get("precipitation"), r.get("solar_amount"), r.get("missing_status"),
             fetched_at))
        new_rows += cur.rowcount
    conn.commit()
    return len(records), new_rows


def save_missing(conn: sqlite3.Connection, records, fetched_at: str) -> int:
    """data_type=0(欠測局一覧)を故障ラベルとして記録する。

    欠測局は他のdata_typeに現れないことがあるため、
    座標をマップに出せるよう局マスタにも登録する(既存行は上書きしない)。
    """
    for r in records:
        conn.execute(
            """INSERT INTO stations (id, data_type, display_name, display_name_roman,
                   pref_code, latitude, longitude, weather_sensor_flg, riamoni_flg,
                   first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen""",
            (r["id"], 0, r.get("display_name"), r.get("display_name_roman"),
             r.get("pref_code"), r.get("latitude"), r.get("longitude"),
             r.get("weather_sensor_flg"), r.get("riamoni_flg"), fetched_at, fetched_at))
        conn.execute(
            "INSERT OR IGNORE INTO missing_log VALUES (?,?,?,?)",
            (r["id"], fetched_at, r.get("missing_status"), r.get("meas_datetime")))
    conn.commit()
    return len(records)


def _selftest():
    """小さいdata_type=3で取得→インメモリ保存の自己テスト。"""
    recs = fetch_snapshot(3)
    assert isinstance(recs, list) and len(recs) > 0, "empty response"
    assert "id" in recs[0] and "latitude" in recs[0], "unexpected schema"
    conn = open_db(":memory:")
    n_st, n_new = save_snapshot(conn, recs, 3, now_iso())
    got = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    print(f"selftest OK: fetched={n_st} inserted={n_new} in_db={got}")


if __name__ == "__main__":
    _selftest()
