#!/usr/bin/env python3
"""観測網の品質監視: 補完モデルの残差を「局の健康診断」として読む。

見つけるもの(いずれも故障判定ではなく点検候補のスクリーニング):
  ・ズレ(ドリフト)候補 — 期待値(自局の平常値×近傍の変動比)から一方向に外れ続ける局。
    値は出ているのに実はおかしい「静かな故障」の候補
  ・張り付き候補 — 値が長時間まったく変化しない局。放射線計測は統計ゆらぎが必ずあるため、
    完全一定は「固まっている」可能性(ただし表示桁が粗い局は誤検知しうる)

実行: conda activate ramis_ml && python quality_watch.py
出力: 画面ランキング + out/quality_watch.csv(全局の指標)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import impute_eval as ie
from automl_eval import build_dataset

HERE = Path(__file__).resolve().parent
SE_FLOOR = 0.005     # 統計的に意味のない微小ズレをt値で過大評価しないための下限


def main():
    m = ie.load()
    data, slots = build_dataset(m, 1500, None)
    names = dict(sqlite3.connect(ie.DB).execute(
        "SELECT id, display_name FROM stations").fetchall())

    # --- ズレ(ドリフト)候補: 履歴のある行のみ
    d = data[data.n_hist >= 2].copy()
    d["expected"] = d.log_past_mean + d.idw_ratio.fillna(0.0)
    d["resid"] = d.logy - d.expected
    g = d.groupby("station_id").resid.agg(["count", "mean", "std"]).fillna(0)
    g = g[g["count"] >= 3]
    se = np.maximum(g["std"] / np.sqrt(g["count"]), SE_FLOOR)
    g["t"] = g["mean"] / se
    g["ズレ%"] = (np.exp(g["mean"]) - 1) * 100

    # --- 張り付き候補: 全期間で値が完全一定
    v = data.groupby("station_id").agg(n=("logy", "size"), nunique=("y", "nunique"),
                                       val=("y", "first"))
    stuck = v[(v.n >= 5) & (v.nunique == 1)]

    out = g.assign(name=[names.get(s, s) for s in g.index]).sort_values(
        "t", key=abs, ascending=False)
    out.to_csv(HERE / "out" / "quality_watch.csv", encoding="utf-8-sig")

    print(f"対象: {len(g)}局 × {len(slots)}断面 (履歴つき行で診断)\n")
    print("=== ズレ(ドリフト)候補 上位10 — 期待値から外れ続ける局 ===")
    print(f"{'局名':<28}{'観測n':>4}{'ズレ%':>8}{'t値':>7}")
    for sid, r in out.head(10).iterrows():
        print(f"{(r['name'] or sid)[:26]:<28}{int(r['count']):>4}{r['ズレ%']:>8.1f}{r['t']:>7.1f}")
    print(f"\n=== 張り付き候補: {len(stuck)}局 (値が全期間で完全一定) ===")
    for sid, r in stuck.head(8).iterrows():
        print(f"  {(names.get(sid) or sid)[:30]:<32} {r.val} µSv/h × {int(r.n)}回")
    print(f"\n全指標: out/quality_watch.csv  "
          f"(注: 断面{len(slots)}枚での暫定診断。蓄積とともに信頼度が上がる)")


if __name__ == "__main__":
    main()
