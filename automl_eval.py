#!/usr/bin/env python3
"""AutoML(FLAML)を既存手法と同じ物差しで比較する評価。データが貯まるほど強くなる。

- 蓄積された全スロット(断面)からデータセットを構築(多断面=学習データ増)
- 分割は局単位(GroupShuffleSplit)。同じ局の行が学習とテストに跨るリークを防ぐ
- 自局の過去平均は「そのスロットより前」だけから計算(時間リークなし)
- 降水>0の行が存在すれば「雨天時のみ」の成績を自動で分けて表示

実行: conda activate ramis_ml && python automl_eval.py
      (時間をかけるなら --budget 600 など。既定はFLAML 1モードあたり120秒)
依存: environment_ml.yml から `conda env create -f environment_ml.yml` で環境を再現可能
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

import impute_eval as ie

try:
    from flaml import AutoML
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False


def _load_radar():
    """radar_rainテーブル(気象庁レーダー実況)を読む。無ければNone。"""
    import sqlite3
    try:
        conn = sqlite3.connect(ie.DB)
        r = pd.read_sql("SELECT station_id, obs_time, mmh FROM radar_rain", conn)
    except Exception:
        return None
    if r.empty:
        return None
    r["t"] = pd.to_datetime(r.obs_time, utc=True)
    return r


def build_dataset(m, min_stations=1500, max_slots=None):
    """全共通スロットを積み上げた学習テーブルを作る。

    各行 = (局, スロット)。特徴 = 近傍・地理・気象・時刻 + 自局の過去平均。
    過去平均は必ず「それより前のスロット」だけから計算する。
    """
    counts = m.groupby("meas_datetime").size()
    slots = counts[counts >= min_stations].sort_index().index.tolist()
    if max_slots:
        slots = slots[-max_slots:]
    assert slots, "共通スロットがない。collect.pyの蓄積を確認すること"

    from scipy.spatial import cKDTree

    radar = _load_radar()
    hist = defaultdict(lambda: [0.0, 0])   # station -> [logyの和, 件数]
    frames = []
    for ts in slots:
        df = ie.slot_frame(m, ts)
        F = ie.neighbor_features(df)
        lpm = np.array([hist[s][0] / hist[s][1] if hist[s][1] > 0 else np.nan
                        for s in df.station_id])
        F["log_past_mean"] = lpm
        F["n_hist"] = [hist[s][1] for s in df.station_id]

        # 基準値比の特徴: 近傍K局の「今/自分の平常値」(対数比)。
        # 局ごとの絶対レベル差が消え、モデルは変動の伝播だけを学べる
        xy = ie.xy_km(df)
        d, idx = cKDTree(xy).query(xy, k=ie.K + 1)
        d, idx = d[:, 1:], idx[:, 1:]
        nb_ratio = df.logv.values[idx] - lpm[idx]       # 履歴のない近傍はNaN
        w = (1.0 / np.maximum(d, 0.1)) * ~np.isnan(nb_ratio)
        wsum = w.sum(1)
        F["idw_ratio"] = np.where(wsum > 0,
                                  np.nansum(nb_ratio * w, axis=1) / np.where(wsum == 0, 1, wsum),
                                  np.nan)
        for k in range(ie.K):
            F[f"r{k}"] = nb_ratio[:, k]

        # レーダー雨(±15分以内の最寄り実況)。無い期間はNaN
        F["radar_mmh"] = np.nan
        if radar is not None:
            tsdt = pd.to_datetime(ts).tz_convert("UTC")
            dt_sec = np.abs((radar["t"] - tsdt).dt.total_seconds())
            near = radar.loc[dt_sec <= 900]
            if len(near):
                best = near.loc[dt_sec[near.index].idxmin(), "obs_time"]
                sub = near[near.obs_time == best]
                F["radar_mmh"] = df.station_id.map(
                    dict(zip(sub.station_id, sub.mmh))).values
        F["station_id"] = df.station_id.values
        F["slot"] = ts
        F["y"] = df.air_dose_rate.values
        F["logy"] = df.logv.values
        frames.append(F)
        for s, lv in zip(df.station_id, df.logv):   # 特徴を作った後に履歴を更新
            hist[s][0] += lv
            hist[s][1] += 1
    data = pd.concat(frames, ignore_index=True)
    return data, slots


def metrics(y, p):
    e = np.abs(p - y)
    return {"MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "中央値誤差%": np.median(e / y) * 100}


def eval_split(data, feats, budget, seed, use_flaml, residual_base="idw",
               ratio_feats=None):
    """局単位で8:2に分け、各手法のテスト予測を返す。

    residual_base: 残差学習の基準列。モデルは「基準からのズレ(対数比)」だけを学び、
    予測 = 基準 + 学習したズレ。基準がNaNの行は学習から除外(予測もNaNになる)。
    """
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                          .split(data, groups=data.station_id))
    tr, te = data.iloc[tr_idx], data.iloc[te_idx]
    preds = {"IDW近傍5局": np.exp(te.idw.values)}

    gbm = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    gbm.fit(tr[feats], tr.logy)
    preds["GBM(直接)"] = np.exp(gbm.predict(te[feats]))

    trb = tr[tr[residual_base].notna()]
    gres = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    gres.fit(trb[feats], trb.logy - trb[residual_base])
    preds[f"GBM残差({residual_base}基準)"] = np.exp(te[residual_base].values
                                                + gres.predict(te[feats]))

    if ratio_feats:   # 基準値比: 目的変数も特徴も「平常値からの変動比」
        grat = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        grat.fit(trb[ratio_feats], trb.logy - trb[residual_base])
        preds["GBM基準値比(近傍も比)"] = np.exp(te[residual_base].values
                                          + grat.predict(te[ratio_feats]))

    if use_flaml:
        automl = AutoML()
        automl.fit(tr[feats], tr.logy, task="regression", metric="mae",
                   time_budget=budget, eval_method="cv", split_type="group",
                   groups=tr.station_id, seed=seed, verbose=0)
        preds[f"FLAML({automl.best_estimator})"] = np.exp(automl.predict(te[feats]))
    return te, preds


def report(te, preds, label):
    print(f"\n--- {label} (テスト{len(te)}行/{te.station_id.nunique()}局) ---")
    rows = {name: metrics(te.y.values, p) for name, p in preds.items()}
    print(pd.DataFrame(rows).T.round(4).to_string())
    radar = np.nan_to_num(te.radar_mmh.values, nan=0.0) if "radar_mmh" in te else 0
    rain = (te.precip.values > 0) | (radar > 0)
    if rain.any():
        rows = {name: metrics(te.y.values[rain], p[rain]) for name, p in preds.items()}
        print(f"  ▼ 雨天時のみ ({rain.sum()}行, センサ+レーダー判定)")
        print(pd.DataFrame(rows).T.round(4).to_string())
    else:
        print("  (雨天行なし: 降水>0のデータが貯まるとここに雨天時の成績が出る)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=120, help="FLAMLの探索秒数/モード")
    ap.add_argument("--max-slots", type=int, default=200, help="使う断面数の上限(直近から)")
    ap.add_argument("--min-stations", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-flaml", action="store_true", help="FLAML抜き(ベースラインのみ)")
    args = ap.parse_args()

    use_flaml = HAS_FLAML and not args.no_flaml
    if not HAS_FLAML:
        print("注意: flamlが見つからない。`conda activate ramis_ml` で実行すること。"
              "環境が無ければ `conda env create -f environment_ml.yml`。今回はFLAML抜きで続行。")

    m = ie.load()
    data, slots = build_dataset(m, args.min_stations, args.max_slots)
    n_rain = int((data.precip > 0).sum())
    print(f"データセット: {len(data):,}行 / {data.station_id.nunique():,}局 / "
          f"{len(slots)}断面 ({slots[0][:16]} 〜 {slots[-1][:16]}) / 雨天行 {n_rain:,}")

    ratio_cols = ["idw_ratio"] + [f"r{k}" for k in range(ie.K)]
    base_feats = [c for c in data.columns
                  if c not in ("station_id", "slot", "y", "logy", "nn_value",
                               "log_past_mean", "n_hist", *ratio_cols)]
    # 基準値比モデルの特徴: 近傍の変動比 + 気象・時刻・地理(変動の伝播に効きうるもの)
    ratio_feats = ratio_cols + ["wind_speed", "wind_sin", "wind_cos", "precip",
                                "radar_mmh", "met_dist", "hour_sin", "hour_cos",
                                "dist_f1", "elev", "n_hist"]
    # 全行NaNの列(例: レーダー蓄積前のradar_mmh)は学習に使えないため除外
    usable = [c for c in data.columns if data[c].notna().any()]
    dropped = [c for c in base_feats + ratio_feats if c not in usable]
    base_feats = [c for c in base_feats if c in usable]
    ratio_feats = [c for c in ratio_feats if c in usable]
    if dropped:
        print(f"注: 全行NaNのため今回は特徴から除外: {sorted(set(dropped))}")

    # モード1: 履歴なし(新設局・完全に未知の局を想定)。残差の基準はIDW
    te, preds = eval_split(data, base_feats, args.budget, args.seed, use_flaml,
                           residual_base="idw")
    report(te, preds, "履歴なし局の補完(周辺+地理+気象のみ)")

    # モード2: 履歴あり(故障局を想定)。残差の基準は自局の過去平均
    # 比較は「履歴のある行」に限定する(無い行では過去平均ベースラインが定義できない)
    te, preds = eval_split(data, base_feats + ["log_past_mean", "n_hist"],
                           args.budget, args.seed, use_flaml,
                           residual_base="log_past_mean", ratio_feats=ratio_feats)
    mask = te.n_hist.values > 0
    te2 = te[mask]
    preds2 = {k: v[mask] for k, v in preds.items()}
    preds2["自局過去平均そのまま"] = np.exp(te2.log_past_mean.values)  # 参照ベースライン
    report(te2, preds2, "故障局の補完(+自局の過去履歴, 履歴のある行のみ)")

    print("\n※ 分割は常に局単位。FLAMLの内部CVも局グループで実施(リークなし)。")


if __name__ == "__main__":
    main()
