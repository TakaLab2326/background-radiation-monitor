#!/usr/bin/env python3
"""欠測イベント台帳: missing_log(生ログ)を「欠測1件=1行」に変換する。

生ログは「収集時点で欠測だった局の名簿」の積み重ね。これを局ごとに
連続する収集セッションのまとまり=1イベントとして再構築する。
  開始   = イベント最初の記録の last_meas_datetime + 10分 (RAMIS由来・高精度)
  復旧確認 = 名簿から消えた収集時刻 (収集間隔ぶんの誤差を含む上限側の値)
継続中イベントには最新スロットのクリギング推定(σつき)を添える。

実行: conda activate base && python3 tools/build_missing_ledger.py
出力: data/missing_events.csv + 要約表示
"""
import csv
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import impute_eval as ie
from kriging_eval import fit_gp
from rise_watch import judge_rain, load_rain_index, load_sessions

OUT = HERE / "data" / "missing_events.csv"


def parse(ts):
    return datetime.fromisoformat(ts)


def build_events(conn):
    """局ごとに連続セッションをまとめてイベント化する。"""
    sessions = [r[0] for r in conn.execute(
        "SELECT DISTINCT fetched_at FROM missing_log ORDER BY fetched_at")]
    s_index = {ts: i for i, ts in enumerate(sessions)}
    rows = conn.execute("""SELECT station_id, fetched_at, last_meas_datetime
                           FROM missing_log ORDER BY station_id, fetched_at""").fetchall()
    events = []
    cur = None
    for sid, fetched, last_meas in rows:
        i = s_index[fetched]
        if cur is not None and (sid != cur["station_id"] or i > cur["end_i"] + 1):
            events.append(cur)
            cur = None
        if cur is None:
            cur = {"station_id": sid, "start_i": i, "end_i": i,
                   "last_meas": last_meas, "n_confirm": 1}
        else:
            cur["end_i"] = i
            cur["n_confirm"] += 1
            # last_measはイベント中に更新されない想定だが、より古い値を正とする
            if last_meas and (not cur["last_meas"] or last_meas < cur["last_meas"]):
                cur["last_meas"] = last_meas
    if cur is not None:
        events.append(cur)

    last_session_i = len(sessions) - 1
    for e in events:
        e["first_confirmed"] = sessions[e["start_i"]]
        if e["end_i"] < last_session_i:
            e["recovered_at"] = sessions[e["end_i"] + 1]   # 次セッションで名簿から消えた
        else:
            e["recovered_at"] = None                        # 最新セッションでも欠測=継続中
        if e["last_meas"]:
            e["start_est"] = (parse(e["last_meas"])
                              + timedelta(minutes=10)).isoformat()
        else:
            e["start_est"] = None
    return events, sessions


def add_station_info(conn, events):
    st = {r[0]: r[1:] for r in conn.execute(
        "SELECT id, display_name, pref_code, latitude, longitude FROM stations")}
    for e in events:
        name, pref, lat, lon = st.get(e["station_id"], (None, None, None, None))
        e.update(name=name, pref=pref, lat=lat, lon=lon)


def add_neighbors(conn, events):
    """イベント局の周辺10km/20kmの局数(同時欠測局を除く稼働局)を数える。"""
    st = conn.execute("SELECT id, latitude, longitude FROM stations "
                      "WHERE latitude IS NOT NULL").fetchall()
    ids = [r[0] for r in st]
    xy = np.c_[[r[2] * 111.32 * np.cos(np.radians(36)) for r in st],
               [r[1] * 110.57 for r in st]]
    tree = cKDTree(xy)
    pos = {i: xy[k] for k, i in enumerate(ids)}

    # セッションごとの欠測集合(そのイベント時点で同時に欠測だった局を除外するため)
    sess_missing = {}
    for sid, fetched in conn.execute("SELECT station_id, fetched_at FROM missing_log"):
        sess_missing.setdefault(fetched, set()).add(sid)

    for e in events:
        p = pos.get(e["station_id"])
        if p is None:
            e["n_10km"] = e["n_20km"] = None
            continue
        down = sess_missing.get(e["first_confirmed"], set())
        for r_km, key in ((10, "n_10km"), (20, "n_20km")):
            near = tree.query_ball_point(p, r_km)
            alive = [k for k in near if ids[k] != e["station_id"]
                     and ids[k] not in down]
            e[key] = len(alive)


def add_gp_estimates(events):
    """継続中イベントに最新スロットのクリギング推定(σつき)を付ける。"""
    ongoing = [e for e in events if e["recovered_at"] is None and e["lat"]]
    if not ongoing:
        return None
    m = ie.load()
    counts = m.groupby("meas_datetime").size()
    ts = counts[counts >= 2000].sort_index().index[-1]
    df = ie.slot_frame(m, ts)
    gp, _ = fit_gp(ie.xy_km(df), df.logv.values, np.random.RandomState(0))
    xy = np.c_[[e["lon"] * 111.32 * np.cos(np.radians(36)) for e in ongoing],
               [e["lat"] * 110.57 for e in ongoing]]
    mu, sd = gp.predict(xy, return_std=True)
    for e, m_, s_ in zip(ongoing, mu, sd):
        e["est_usvh"] = round(float(np.exp(m_)), 4)
        e["sigma_rel_pct"] = round(float((np.exp(s_) - 1) * 100), 1)
    return ts


def add_rain(conn, events):
    """欠測開始時に頭上の雨があったか(天候由来欠測の仮説検証用)。"""
    import pandas as pd
    rain_idx = load_rain_index(conn)
    sessions = load_sessions(conn)
    for e in events:
        if e["start_est"]:
            e["rain_mm_3h"], e["rain_flag"] = judge_rain(
                rain_idx, sessions, e["station_id"],
                pd.Timestamp(e["start_est"]).tz_convert("UTC"))
        else:
            e["rain_mm_3h"], e["rain_flag"] = None, "不明"


def main():
    conn = sqlite3.connect(HERE / "data" / "ramis.sqlite3")
    events, sessions = build_events(conn)
    add_station_info(conn, events)
    add_neighbors(conn, events)
    add_rain(conn, events)
    slot = add_gp_estimates(events)

    now = datetime.now().astimezone()
    cols = ["station_id", "name", "pref", "lat", "lon", "start_est",
            "first_confirmed", "recovered_at", "duration_h", "status",
            "n_confirm", "n_10km", "n_20km", "rain_mm_3h", "rain_flag",
            "est_usvh", "sigma_rel_pct"]
    for e in events:
        e["status"] = "継続中" if e["recovered_at"] is None else "復旧"
        ref = parse(e["recovered_at"]) if e["recovered_at"] else now
        e["duration_h"] = (round((ref - parse(e["start_est"])).total_seconds() / 3600, 1)
                           if e["start_est"] else None)
    events.sort(key=lambda e: (e["start_est"] or ""), reverse=True)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(events)

    ongoing = [e for e in events if e["status"] == "継続中"]
    rec = [e for e in events if e["status"] == "復旧" and e["duration_h"] is not None]
    print(f"セッション数: {len(sessions)} ({sessions[0][:16]} 〜 {sessions[-1][:16]})")
    print(f"イベント総数: {len(events)} (継続中 {len(ongoing)} / 復旧 {len(rec)})")
    if rec:
        d = np.array([e["duration_h"] for e in rec])
        print(f"復旧イベントの継続時間: 中央値 {np.median(d):.1f}h / "
              f"90%タイル {np.percentile(d, 90):.1f}h / 最大 {d.max():.1f}h")
        multi = {}
        for e in rec:
            multi[e["station_id"]] = multi.get(e["station_id"], 0) + 1
        flick = sum(1 for v in multi.values() if v >= 3)
        print(f"チラつき型(復旧イベント3回以上): {flick}局")
    if slot:
        print(f"継続中イベントへのσつき推定: スロット {slot} で付与")
    flags = {}
    for e in events:
        flags[e["rain_flag"]] = flags.get(e["rain_flag"], 0) + 1
    print(f"欠測開始時の雨: {flags}")
    print(f"出力: {OUT}")


if __name__ == "__main__":
    main()
