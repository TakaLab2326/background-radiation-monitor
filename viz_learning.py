#!/usr/bin/env python3
"""欠測補完の学習を可視化するためのデータ生成 → 自己完結HTMLを出力。

impute_eval.py と同じ特徴量・同じ評価設計(局分割)を使い、
5-fold の out-of-fold 予測で「全局が一度はテストになる」形にする。

実行: conda activate base && python viz_learning.py
出力: out/learn_viz.html
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

import impute_eval as ie

HERE = Path(__file__).resolve().parent

FEAT_JP = {
    "idw": "近傍IDW平均", "n_std": "近傍のばらつき",
    "lat": "緯度", "lon": "経度",
    "wind_speed": "風速", "wind_sin": "風向(南北成分)", "wind_cos": "風向(東西成分)",
    "precip": "降水量", "met_dist": "最寄り気象局までの距離",
    "elev": "標高", "elev_rel": "近傍との標高差", "dist_f1": "福島第一からの距離",
    "hour_sin": "時刻(周期成分1)", "hour_cos": "時刻(周期成分2)",
    **{f"v{k}": f"近傍{k+1}位の値" for k in range(ie.K)},
    **{f"d{k}": f"近傍{k+1}位までの距離" for k in range(ie.K)},
}


def main():
    m = ie.load()
    counts = m.groupby("meas_datetime").size()
    target_ts = counts[counts >= 2000].sort_index().index[-1]
    df = ie.slot_frame(m, target_ts)
    F = ie.neighbor_features(df)
    feats = [c for c in F.columns if c != "nn_value"]
    y = df.air_dose_rate.values

    # --- 5-fold OOF (局単位で分割: 各局は必ず「学習に不使用」の状態で予測される)
    oof = np.zeros(len(df))
    last_model, last_test = None, None
    for tr, te in KFold(5, shuffle=True, random_state=0).split(df):
        model = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        model.fit(F.iloc[tr][feats], df.logv.iloc[tr])
        oof[te] = model.predict(F.iloc[te][feats])
        last_model, last_test = model, te
    pred_gbm = np.exp(oof)
    pred_idw = np.exp(F.idw.values)

    mae = {
        "idw": float(np.abs(pred_idw - y).mean()),
        "gbm": float(np.abs(pred_gbm - y).mean()),
        "idw_medpct": float(np.median(np.abs(pred_idw - y) / y) * 100),
        "gbm_medpct": float(np.median(np.abs(pred_gbm - y) / y) * 100),
    }

    # --- 特徴量重要度 (最終foldのテスト局で並べ替え検証)
    r = permutation_importance(last_model, F.iloc[last_test][feats],
                               df.logv.iloc[last_test], n_repeats=5, random_state=0)
    imp = sorted(zip(feats, r.importances_mean, r.importances_std),
                 key=lambda t: -t[1])[:10]
    importance = [[FEAT_JP.get(f, f), round(float(mu), 5), round(float(sd), 5)]
                  for f, mu, sd in imp]

    # --- 本物の欠測局をIDW(現時点の最良手法)で補完
    conn = sqlite3.connect(ie.DB)
    last_fetch = conn.execute("SELECT MAX(fetched_at) FROM missing_log").fetchone()[0]
    miss = pd.read_sql("""
        SELECT s.id, s.display_name AS name, s.latitude AS lat, s.longitude AS lon,
               ml.last_meas_datetime AS last_meas
        FROM missing_log ml JOIN stations s ON s.id = ml.station_id
        WHERE ml.fetched_at = ? AND s.latitude IS NOT NULL
    """, conn, params=(last_fetch,))
    live_xy = ie.xy_km(df)
    tree = cKDTree(live_xy)
    d, idx = tree.query(ie.xy_km(miss), k=ie.K)
    w = 1.0 / np.maximum(d, 0.1)
    miss_pred = np.exp((df.logv.values[idx] * w).sum(1) / w.sum(1))

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "slot": target_ts[5:16].replace("T", " "),
        "n": len(df),
        "mae": {k: round(v, 4) for k, v in mae.items()},
        "importance": importance,
        # [lon, lat, 実測, IDW予測, GBM予測, 局名]
        "points": [[round(df.lon[i], 5), round(df.lat[i], 5), round(float(y[i]), 5),
                    round(float(pred_idw[i]), 5), round(float(pred_gbm[i]), 5),
                    df.display_name[i].replace("　", " ") if df.display_name[i] else df.station_id[i]]
                   for i in range(len(df))],
        # [lon, lat, 予測値, 局名, 最終実測, 最寄り生存局km]
        "missing_pred": [[round(miss.lon[j], 5), round(miss.lat[j], 5),
                          round(float(miss_pred[j]), 5),
                          (miss.name[j] or miss.id[j]).replace("　", " "),
                          (miss.last_meas[j] or "")[:16].replace("T", " "),
                          round(float(d[j, 0]), 1)]
                         for j in range(len(miss))],
        "coast": json.loads((HERE / "assets" / "japan_outline.json").read_text()),
    }

    tpl = (HERE / "learn_viz_template.html").read_text()
    marker = "/*__PAYLOAD__*/null"
    assert marker in tpl
    out = HERE / "out" / "learn_viz.html"
    out.write_text(tpl.replace(marker, json.dumps(payload, ensure_ascii=False,
                                                  separators=(",", ":"))))
    print(f"OK: 評価{len(df)}局 欠測補完{len(miss)}局 スロット{target_ts}")
    print(f"    MAE idw={mae['idw']:.4f} gbm={mae['gbm']:.4f} -> {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
