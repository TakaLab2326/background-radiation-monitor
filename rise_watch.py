#!/usr/bin/env python3
"""上昇イベントの検出と雨照合: 「雨で説明できる上昇か」を全時系列で自動ラベルする。

背景: 平常時の線量率上昇のほぼ全ては降雨によるラドン子孫核種の洗い落とし
(雨と同期して上がり、止むと半減期20〜27分で戻る)。よって
「雨が無いのに上がった」イベントだけが点検・注視の対象になる。

判定(1エピソード = 同一局の連続した上昇行のまとまり):
  上昇行 = 自局履歴のロバストz(中央値/MAD基準) >= Z_RISE
  雨あり = エピソード開始前 WINDOW_H 時間の頭上レーダー雨 > 0
  雨なし = 同窓内に雨の観測機会があり、雨ゼロ
  不明   = 同窓内に収集が無く雨情報が欠けている(収集間引きの正直な扱い)

制約: 収集間隔(1〜3時間)ぶんの見逃しがあるため事後検出用。リアルタイム警報ではない。
実行: conda activate base && python3 rise_watch.py
出力: 画面サマリ + out/rise_watch.csv(エピソード一覧)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE / "data" / "ramis.sqlite3"
OUT = HERE / "out" / "rise_watch.csv"

Z_RISE = 4.0       # ロバストz閾値(正規近似で誤検知率~3e-5)
MAD_FLOOR = 0.002  # µSv/h。表示分解能0.001の2目盛=統計ゆらぎ未満のMADを底上げ
MIN_HIST = 10      # 履歴がこれ未満の局はベースラインを推定しない
WINDOW_H = 3       # 雨照合窓。半減期20〜27分×6以上で雨性上昇は減衰し切る
TOL_MIN = 20       # 測定スロット→雨観測のずれ許容。収集cronはスロットの最大10分後+
                   # 実行1〜2分+ナウキャスト5分丸め ≈ 17分までずれるため
GAP_H = 1          # 上昇行の間隔がこれを超えたら別エピソード


def load_sessions(conn):
    """収集セッション時刻(UTC)。雨の「観測機会」があった時刻の一覧。"""
    ts = [r[0] for r in conn.execute("SELECT DISTINCT fetched_at FROM measurements")]
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values()


def load_rain_index(conn):
    """局ID → 雨>0の観測(UTC時刻, mm/h)テーブル。radar_rain未作成なら空。"""
    try:
        df = pd.read_sql(
            "SELECT station_id, obs_time, mmh FROM radar_rain WHERE mmh > 0", conn)
    except pd.errors.DatabaseError:
        return {}
    df["t"] = pd.to_datetime(df.obs_time, utc=True)
    return {sid: g[["t", "mmh"]] for sid, g in df.groupby("station_id")}


def judge_rain(rain_idx, sessions, sid, t_utc, t_end_utc=None, window_h=WINDOW_H):
    """(窓内最大雨量mm/h, 判定)。判定 = 雨あり / 雨なし / 不明。

    窓は「開始のwindow_h時間前 〜 終了+許容」。雨性上昇は雨と同時進行するため、
    エピソード継続中の雨も「説明できる雨」に数える。
    """
    lo = t_utc - pd.Timedelta(hours=window_h)
    hi = (t_end_utc if t_end_utc is not None else t_utc) + pd.Timedelta(minutes=TOL_MIN)
    if not ((sessions > lo) & (sessions <= hi)).any():
        return None, "不明"
    g = rain_idx.get(sid)
    if g is not None:
        hit = g[(g.t > lo) & (g.t <= hi)]
        if len(hit):
            return float(hit.mmh.max()), "雨あり"
    return 0.0, "雨なし"


def find_episodes(m):
    """上昇行を局ごとに連結してエピソード化する。"""
    med = m.groupby("station_id").air_dose_rate.transform("median")
    mad = (m.air_dose_rate - med).abs().groupby(m.station_id).transform("median")
    # 量子化幅(局の最小表示刻み)をσの下限にする。表示が0.01刻みの粗い局は
    # MADが0になりやすく、±1〜2目盛の揺らぎを異常と誤検知するため
    quant = m.groupby("station_id").air_dose_rate.transform(
        lambda v: np.diff(np.unique(v)).min() if v.nunique() > 1 else MAD_FLOOR)
    sigma = np.maximum.reduce([mad.values * 1.4826, quant.values,
                               np.full(len(m), MAD_FLOOR)])
    n = m.groupby("station_id").air_dose_rate.transform("size")
    m = m.assign(base=med, z=(m.air_dose_rate - med) / sigma)
    rises = m[(m.z >= Z_RISE) & (n >= MIN_HIST)].sort_values(["station_id", "t"])

    episodes = []
    for sid, g in rises.groupby("station_id"):
        new = g.t.diff() > pd.Timedelta(hours=GAP_H)
        for _, ep in g.groupby(new.cumsum()):
            peak = ep.loc[ep.z.idxmax()]
            episodes.append({
                "station_id": sid, "name": peak.display_name, "pref": peak.pref_code,
                "start": ep.t.min(), "end": ep.t.max(), "n_rows": len(ep),
                "baseline": round(float(peak.base), 4),
                "peak_usvh": float(peak.air_dose_rate), "peak_z": round(float(peak.z), 1),
            })
    return episodes


def main():
    conn = sqlite3.connect(DB)
    m = pd.read_sql("""
        SELECT m.station_id, m.meas_datetime, m.air_dose_rate,
               s.display_name, s.pref_code
        FROM measurements m JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate > 0
    """, conn)
    m["t"] = pd.to_datetime(m.meas_datetime, utc=True)
    print(f"対象: {len(m):,}行 / {m.station_id.nunique():,}局")

    episodes = find_episodes(m)
    sessions = load_sessions(conn)
    rain_idx = load_rain_index(conn)
    for e in episodes:
        e["rain_mm_3h"], e["rain_flag"] = judge_rain(
            rain_idx, sessions, e["station_id"], e["start"], e["end"])
        e["start"] = e["start"].tz_convert("Asia/Tokyo").isoformat()
        e["end"] = e["end"].tz_convert("Asia/Tokyo").isoformat()

    df = pd.DataFrame(episodes)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"上昇エピソード: {len(df)}件")
    if len(df):
        print(df.rain_flag.value_counts().to_string())
        watch = df[df.rain_flag == "雨なし"].nlargest(10, "peak_z")
        if len(watch):
            print("\n=== 雨で説明できない上昇(要注視候補・上位) ===")
            for _, r in watch.iterrows():
                print(f"  {r['name']}({r.pref}) {r.start[5:16]} "
                      f"{r.baseline}→{r.peak_usvh} µSv/h (z={r.peak_z})")
    print(f"出力: {OUT}")


if __name__ == "__main__":
    main()
