#!/usr/bin/env python3
"""クリギング(ガウス過程回帰)による空間補完の評価。

売り: 予測値に不確かさ(±σ)が付く。欠測局の推定に「どれくらい信用できるか」を添えられる。
- カーネル: 定数×Matérn(ν=1.5) + ホワイトノイズ。入力は平面近似km座標(x, y)
- 速度対策: ハイパーパラメータは1,000局のサブサンプルで学習し、全局には固定して適用
- 検証: 局8:2分割×3シード×全共通スロット。MAEに加え「±1σ/±2σに実測が入る率」(較正)を報告
  (理想: 1σ≈68%, 2σ≈95%。これが大きく外れるならσは信用できない)

実行: conda activate ramis_ml && python kriging_eval.py
"""
import argparse

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

import impute_eval as ie

SEEDS = range(3)


def fit_gp(x_tr, y_tr, rng):
    """サブサンプルでカーネル学習 → 全学習点に固定カーネルで適用したGPを返す。"""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=20.0, length_scale_bounds=(0.5, 500.0), nu=1.5)
              + WhiteKernel(0.05, (1e-4, 2.0)))
    sub = rng.choice(len(x_tr), size=min(1000, len(x_tr)), replace=False)
    gp0 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp0.fit(x_tr[sub], y_tr[sub])
    gp = GaussianProcessRegressor(kernel=gp0.kernel_, optimizer=None,
                                  normalize_y=True, random_state=0)
    gp.fit(x_tr, y_tr)
    return gp, gp0.kernel_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stations", type=int, default=1500)
    args = ap.parse_args()

    m = ie.load()
    counts = m.groupby("meas_datetime").size()
    slots = counts[counts >= args.min_stations].sort_index().index.tolist()
    assert slots, "共通スロットがない"

    rows = {"IDW近傍5局": [], "クリギング(GP)": []}
    cover1, cover2, kernels = [], [], []
    for ts in slots:
        df = ie.slot_frame(m, ts)
        F = ie.neighbor_features(df)
        xy = ie.xy_km(df)
        y = df.air_dose_rate.values
        logy = df.logv.values
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            te = rng.rand(len(df)) < 0.2
            gp, learned = fit_gp(xy[~te], logy[~te], rng)
            mu, sd = gp.predict(xy[te], return_std=True)
            err_log = np.abs(mu - logy[te])
            cover1.append((err_log <= sd).mean())
            cover2.append((err_log <= 2 * sd).mean())
            kernels.append(learned)
            e_gp = np.abs(np.exp(mu) - y[te])
            e_idw = np.abs(np.exp(F.idw.values[te]) - y[te])
            rows["クリギング(GP)"].append((e_gp.mean(), np.median(e_gp / y[te]) * 100))
            rows["IDW近傍5局"].append((e_idw.mean(), np.median(e_idw / y[te]) * 100))

    print(f"評価: {len(slots)}スロット × {len(SEEDS)}シード (局8:2分割)")
    print(f"{'手法':<14}{'MAE µSv/h':>12}{'中央値誤差%':>10}")
    for name, r in rows.items():
        a = np.array(r)
        print(f"{name:<14}{a[:,0].mean():>12.4f}{a[:,1].mean():>10.2f}")
    print(f"\nσの較正: ±1σ内 {np.mean(cover1)*100:.1f}% (理想68%) / "
          f"±2σ内 {np.mean(cover2)*100:.1f}% (理想95%)")
    print(f"学習されたカーネル例: {kernels[-1]}")

    # 本物の欠測局へ適用: 最新スロット全局で学習し、σつき推定を出す
    df = ie.slot_frame(m, slots[-1])
    gp, _ = fit_gp(ie.xy_km(df), df.logv.values, np.random.RandomState(0))
    import sqlite3
    conn = sqlite3.connect(ie.DB)
    last = conn.execute("SELECT MAX(fetched_at) FROM missing_log").fetchone()[0]
    miss = conn.execute("""SELECT s.longitude, s.latitude, s.display_name
                           FROM missing_log ml JOIN stations s ON s.id=ml.station_id
                           WHERE ml.fetched_at=? AND s.longitude IS NOT NULL""",
                        (last,)).fetchall()
    mxy = np.c_[[lo * 111.32 * np.cos(np.radians(36)) for lo, la, n in miss],
                [la * 110.57 for lo, la, n in miss]]
    mu, sd = gp.predict(mxy, return_std=True)
    rel = np.exp(sd) - 1     # 対数σ → 「×(1±rel)」のおおよその相対幅
    print(f"\n欠測局{len(miss)}局へのσつき推定(最新スロット{slots[-1][5:16]}):")
    print(f"  相対不確かさ(1σ): 中央値 ±{np.median(rel)*100:.0f}% / 最大 ±{rel.max()*100:.0f}%")
    worst = int(np.argmax(sd))
    print(f"  最も不確かな局: {miss[worst][2]} (推定 {np.exp(mu[worst]):.3f} µSv/h ±{rel[worst]*100:.0f}%)")


if __name__ == "__main__":
    main()
