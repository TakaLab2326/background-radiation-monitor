#!/usr/bin/env python3
"""欠測補完の精度評価: 周辺局・地理・風・雨を特徴に、人工マスクで実測する。

シナリオA: 履歴のない局の空間補完(局分割8:2、テスト局は学習に不使用)
シナリオB: 故障局の時間外挿(自局の過去履歴 → 数時間後を予測、時間前向き分割)

実行: conda activate base && python impute_eval.py
依存: numpy / pandas / scikit-learn / scipy (Anaconda base標準搭載)
注意: 降水特徴は配管済みだが、雨天データが蓄積されるまで効果は出ない。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DB = HERE / "data" / "ramis.sqlite3"
K = 5          # 近傍局数
SEEDS = range(5)

DIR16 = {n: i * 22.5 for i, n in enumerate(
    ["北","北北東","北東","東北東","東","東南東","南東","南南東",
     "南","南南西","南西","西南西","西","西北西","北西","北北西"])}


def load():
    conn = sqlite3.connect(DB)
    m = pd.read_sql("""
        SELECT m.station_id, m.meas_datetime, m.air_dose_rate,
               m.wind_direction, m.wind_speed, m.precipitation,
               s.latitude AS lat, s.longitude AS lon, s.display_name, s.elevation
        FROM measurements m JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate IS NOT NULL AND s.latitude IS NOT NULL
    """, conn)
    return m


def slot_frame(m, ts):
    """指定時刻の断面(1局1行)。値0以下(検出下限等)は評価対象外。"""
    df = m[(m.meas_datetime == ts) & (m.air_dose_rate > 0)]
    df = df.drop_duplicates("station_id").reset_index(drop=True)
    df["logv"] = np.log(df.air_dose_rate)
    return df


def xy_km(df):
    """近傍探索用の平面近似座標(km)。日本スケールでは十分な精度。"""
    lat0 = 36.0
    return np.c_[df.lon.values * 111.32 * np.cos(np.radians(lat0)),
                 df.lat.values * 110.57]


def neighbor_features(df):
    """各局について近傍K局の値・距離と気象特徴を作る(自局は除外)。"""
    xy = xy_km(df)
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=K + 1)          # 先頭は自分
    d, idx = d[:, 1:], idx[:, 1:]
    v = df.logv.values[idx]                    # (n, K) 近傍のlog線量
    w = 1.0 / np.maximum(d, 0.1)
    idw = (v * w).sum(1) / w.sum(1)

    # 風・雨: センサのある局の値を、無い局へは最寄りセンサ局から引き当て
    met = df[df.wind_speed.notna()].reset_index(drop=True)
    if len(met) >= 3:
        mt = cKDTree(xy_km(met))
        md, mi = mt.query(xy, k=1)
        ws = met.wind_speed.values[mi]
        ang = np.radians([DIR16.get(x, np.nan) for x in met.wind_direction.values[mi]])
        pr = met.precipitation.fillna(0).values[mi]
    else:
        md = np.full(len(df), 999.0); ws = np.zeros(len(df))
        ang = np.full(len(df), np.nan); pr = np.zeros(len(df))

    # 福島第一からの距離(km)。事故由来のセシウム沈着場の急勾配を捉える(実測で有効確認済み)
    dist_f1 = np.hypot((df.lon.values - 141.0328) * 111.32 * np.cos(np.radians(36)),
                       (df.lat.values - 37.4214) * 110.57)

    # 標高: 自局の値と近傍K局平均との差(谷底=負、尾根=正)。未取得局は中央値で埋める
    elev = pd.to_numeric(df.elevation, errors="coerce").values
    med = np.nanmedian(elev)
    elev_f = np.where(np.isnan(elev), med if np.isfinite(med) else 0.0, elev)
    elev_rel = elev_f - elev_f[idx].mean(1)

    hour = pd.to_datetime(df.meas_datetime.iloc[0]).hour
    F = pd.DataFrame({
        "elev": elev_f, "elev_rel": elev_rel, "dist_f1": dist_f1,
        "idw": idw,
        "n_std": v.std(1),
        "lat": df.lat.values, "lon": df.lon.values,
        "wind_speed": ws, "wind_sin": np.sin(ang), "wind_cos": np.cos(ang),
        "precip": pr, "met_dist": md,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
    })
    for k in range(K):
        F[f"v{k}"] = v[:, k]
        F[f"d{k}"] = d[:, k]
    F["nn_value"] = v[:, 0]                    # 最近傍コピー(ベースライン用)
    return F


def metrics(y, p):
    e = np.abs(p - y)
    return {"MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "中央値誤差%": np.median(e / y) * 100}


def scenario_A(df):
    """履歴なし局の空間補完。局8:2分割×5シード。"""
    F = neighbor_features(df)
    y = df.air_dose_rate.values
    feats = [c for c in F.columns if c != "nn_value"]
    rows = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        test = rng.rand(len(df)) < 0.2
        model = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        model.fit(F.loc[~test, feats], df.logv[~test])
        preds = {
            "全国中央値(参考)": np.full(test.sum(), np.median(y[~test])),
            "最近傍コピー": np.exp(F.nn_value[test]),
            f"IDW近傍{K}局": np.exp(F.idw[test]),
            "GBM(周辺+地理+風)": np.exp(model.predict(F.loc[test, feats])),
        }
        for name, p in preds.items():
            rows.setdefault(name, []).append(metrics(y[test], p))
    return {n: pd.DataFrame(r).mean() for n, r in rows.items()}


def scenario_B(m, hist_end, target_ts):
    """故障局の時間外挿: hist_end までの自局平均で target_ts を予測。"""
    hist = m[m.meas_datetime <= hist_end]
    past = hist.groupby("station_id").agg(
        past_mean=("air_dose_rate", "mean"), n_hist=("air_dose_rate", "size"),
        lat=("lat", "first"), lon=("lon", "first"))
    now = slot_frame(m, target_ts).set_index("station_id")
    both = past.join(now[["air_dose_rate"]], how="inner").dropna()
    both = both[both.n_hist >= 2]

    # 近傍の「現在値/過去平均」比で自局過去平均を補正 (ratio-IDW)
    xy = np.c_[both.lon * 111.32 * np.cos(np.radians(36)), both.lat * 110.57]
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=K + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    ratio = (both.air_dose_rate.values / both.past_mean.values)[idx]
    w = 1.0 / np.maximum(d, 0.1)
    idw_ratio = (ratio * w).sum(1) / w.sum(1)

    y = both.air_dose_rate.values
    res = {
        "自局過去平均そのまま": metrics(y, both.past_mean.values),
        "過去平均×近傍変動比": metrics(y, both.past_mean.values * idw_ratio),
    }
    return pd.DataFrame(res).T, len(both)


def main():
    m = load()
    counts = m.groupby("meas_datetime").size()
    big = counts[counts >= 2000].sort_index()
    assert len(big) >= 2, f"共通スロットが不足。collect.pyの蓄積を増やすこと: {counts.tail()}"
    target_ts = big.index[-1]
    print(f"共通スロット(2000局以上): {len(big)}個  評価対象: {target_ts} ({big.iloc[-1]}局)\n")

    print("=== シナリオA: 履歴のない局を周辺+地理+風で補完(テスト局は学習に不使用, 5試行平均) ===")
    A = scenario_A(slot_frame(m, target_ts))
    print(pd.DataFrame(A).T.round(4).to_string(), "\n")

    hist_end = big.index[-2]
    B, n = scenario_B(m, hist_end, target_ts)
    dt_h = (pd.to_datetime(target_ts) - pd.to_datetime(hist_end)).total_seconds() / 3600
    print(f"=== シナリオB: 故障局の外挿({hist_end} までの履歴 → {dt_h:.1f}時間後, {n}局) ===")
    print(B.round(4).to_string())
    print("\n注: 現データに降水>0が無いため雨特徴は未検証(配管済み)。雨天蓄積後に再評価する。")


if __name__ == "__main__":
    main()
