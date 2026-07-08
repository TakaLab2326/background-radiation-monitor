#!/usr/bin/env python3
"""本物の故障期間での答え合わせ(バックフィル評価)。

RAMISは局が復旧すると欠測期間の過去データをまとめて返す(バックフィル)。
そこで「遅れて届いた行」を本物の欠測エピソードとみなし、
  ・その測定時刻に「オンタイムで届いていた行」だけで推定(=当時知り得た情報のみ)
  ・後から届いた真値と比較
することで、人工マスクではない実運用条件の成績を測る。

判定基準:
  遅配行  = fetched_at が meas_datetime より LATE_H 時間以上遅い行
  オンタイム = fetched_at - meas_datetime < ONTIME_H 時間
  評価可能 = 同時刻のオンタイム近傍局が MIN_NEIGHBORS 局以上ある遅配行
  (自分の収集が止まっていた時間帯は近傍もオンタイムでないため自動的に除外される)

実行: conda activate ramis_ml && python backfill_eval.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import impute_eval as ie

LATE_H = 1.0
ONTIME_H = 1.0
MIN_NEIGHBORS = 500
K = ie.K


def load_all():
    conn = sqlite3.connect(ie.DB)
    df = pd.read_sql("""
        SELECT m.station_id, m.meas_datetime, m.fetched_at, m.air_dose_rate,
               s.latitude AS lat, s.longitude AS lon, s.display_name
        FROM measurements m JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate > 0 AND s.latitude IS NOT NULL
    """, conn)
    df["meas_dt"] = pd.to_datetime(df.meas_datetime)
    df["fetch_dt"] = pd.to_datetime(df.fetched_at)
    df["lag_h"] = (df.fetch_dt - df.meas_dt).dt.total_seconds() / 3600
    df["logv"] = np.log(df.air_dose_rate)
    return df


def main():
    df = load_all()
    late = df[df.lag_h >= LATE_H]
    print(f"遅配行(≥{LATE_H}h遅れ): {len(late)}行 / {late.station_id.nunique()}局")

    ontime = df[df.lag_h < ONTIME_H]
    # 局ごとの平常値(オンタイム行のみ・全期間平均で近似)
    base = ontime.groupby("station_id").logv.mean()

    rows = []
    for ts, grp in late.groupby("meas_datetime"):
        on = ontime[ontime.meas_datetime == ts]
        on = on[~on.station_id.isin(grp.station_id)]      # 遅配局自身は除外
        if len(on) < MIN_NEIGHBORS:
            continue
        oxy = ie.xy_km(on)
        tree = cKDTree(oxy)
        on_logv = on.logv.values
        on_base = base.reindex(on.station_id).values       # 近傍の平常値(無ければNaN)
        for _, r in grp.iterrows():
            if r.station_id not in base.index:
                continue                                    # 自局の平常値が無ければ対象外
            pm = base[r.station_id]
            xy = np.array([[r.lon * 111.32 * np.cos(np.radians(36)), r.lat * 110.57]])
            d, idx = tree.query(xy, k=min(K, len(on)))
            d, idx = d[0], idx[0]
            w = 1.0 / np.maximum(d, 0.1)
            idw = (on_logv[idx] * w).sum() / w.sum()
            ratio = on_logv[idx] - on_base[idx]
            ok = ~np.isnan(ratio)
            idw_ratio = (ratio[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else 0.0
            rows.append({
                "station": r.display_name or r.station_id, "ts": ts,
                "lag_h": r.lag_h, "y": r.air_dose_rate,
                "過去平均そのまま": np.exp(pm),
                f"IDW近傍{K}局": np.exp(idw),
                "過去平均×近傍変動比": np.exp(pm + idw_ratio),
            })

    if not rows:
        print("評価可能なエピソードがまだ無い(収集の蓄積とともに増える)")
        return
    ev = pd.DataFrame(rows)
    print(f"評価可能: {len(ev)}行 / {ev.station.nunique()}局 / "
          f"{ev.ts.nunique()}時刻  遅配 中央値{ev.lag_h.median():.1f}h 最大{ev.lag_h.max():.1f}h\n")
    print("=== 本物の欠測期間での成績(人工マスクではない) ===")
    y = ev.y.values
    for mth in ["過去平均そのまま", f"IDW近傍{K}局", "過去平均×近傍変動比"]:
        e = np.abs(ev[mth].values - y)
        print(f"  {mth:<14} MAE {e.mean():.4f}  中央値誤差 {np.median(e/y)*100:.2f}%")
    worst = ev.assign(err=np.abs(ev["過去平均×近傍変動比"] - ev.y)).nlargest(3, "err")
    print("\n外した例(過去平均×近傍変動比):")
    for _, r in worst.iterrows():
        print(f"  {r.station[:24]} {r.ts[5:16]} 真値{r.y:.3f} 推定{r['過去平均×近傍変動比']:.3f}")


if __name__ == "__main__":
    main()
