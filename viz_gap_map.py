#!/usr/bin/env python3
"""情報空白マップ: クリギング不確実性(σ)×人口 = 「推定が不確かな場所に何人住むか」。

各局の最新測定(直近3時間窓)でガウス過程を学習し、約5kmグリッドの
相対不確実性(1σ)を推定。WorldPop 1km人口を同グリッドに合算して
  空白度 = 相対不確実性(%) × 居住人口
を地図化する。欠測補完の不確実性を「住民への情報提供リスク」として見る試作。

実行: conda activate base && python3 viz_gap_map.py
入力: data/ramis.sqlite3 / data/worldpop_jpn_2020_1km.tif / assets/japan_outline.json
出力: out/gap_map.html (自己完結・ライト/ダーク対応)
人口データ出典: WorldPop 2020 UN-adjusted 1km (CC BY 4.0) https://www.worldpop.org/
"""
import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

import impute_eval as ie
from kriging_eval import fit_gp

HERE = Path(__file__).resolve().parent
POP_TIF = HERE / "data" / "worldpop_jpn_2020_1km.tif"
OUT = HERE / "out" / "gap_map.html"
BLOCK = 6            # 1kmピクセル×6 = 0.05°グリッド
MIN_POP = 100        # これ未満のセルは表示しない(描画とGP予測の削減)
WINDOW_H = 3         # 「最新値」とみなす時間窓
MAX_TRAIN = 4000     # GP学習局数の上限(計算コスト)


def load_pop_cells():
    """WorldPop 1kmを0.05°セルに合算し (lon, lat, pop) を返す。"""
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(POP_TIF)
    sx, sy, _ = im.tag_v2[33550]
    _, _, _, x0, y0, _ = im.tag_v2[33922]
    a = np.array(im)
    a = np.where(a < 0, 0, a)                      # nodata(-99999)→0
    h = (a.shape[0] // BLOCK) * BLOCK
    w = (a.shape[1] // BLOCK) * BLOCK
    pop = a[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum((1, 3))
    ii, jj = np.nonzero(pop >= MIN_POP)
    lon = x0 + (jj * BLOCK + BLOCK / 2) * sx
    lat = y0 - (ii * BLOCK + BLOCK / 2) * sy
    return lon, lat, pop[ii, jj]


def latest_field(conn):
    """各局の最新測定(全体最新から3時間以内)を1局1行で返す。"""
    m = pd.read_sql("""
        SELECT m.station_id, m.meas_datetime, m.air_dose_rate,
               s.latitude AS lat, s.longitude AS lon, s.display_name, s.pref_code
        FROM measurements m JOIN stations s ON s.id = m.station_id
        WHERE m.air_dose_rate > 0 AND s.latitude IS NOT NULL
    """, conn)
    t = pd.to_datetime(m.meas_datetime)
    win = t >= (t.max() - timedelta(hours=WINDOW_H))
    m = (m[win].sort_values("meas_datetime")
         .drop_duplicates("station_id", keep="last").reset_index(drop=True))
    m["logv"] = np.log(m.air_dose_rate)
    return m, str(t.max())


def main():
    conn = sqlite3.connect(HERE / "data" / "ramis.sqlite3")
    st, t_max = latest_field(conn)
    print(f"学習局数: {len(st)} (最新測定 {t_max} から{WINDOW_H}時間窓)")

    tr = st
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(MAX_TRAIN, random_state=0).reset_index(drop=True)
        print(f"GP学習コストのため {MAX_TRAIN} 局にサブサンプル")
    # σの地域較正: 福島(沈着場の不均一が大きい)とその他でハイパーパラメータを
    # 別学習し、予測の条件付けは全局で行う。latent=Trueで機器ノイズを除いた場のσ。
    xy_tr = ie.xy_km(tr)
    gps = {}
    for key, sub in (("その他", st[st.pref_code != "07"]),
                     ("福島", st[st.pref_code == "07"])):
        gps[key], k = fit_gp(xy_tr, tr.logv.values, np.random.RandomState(0),
                             hyper_xy=ie.xy_km(sub), hyper_y=sub.logv.values,
                             latent=True)
        print(f"カーネル({key}, {len(sub)}局で学習): {k}")

    lon, lat, pop = load_pop_cells()
    print(f"人口セル(0.05°, {MIN_POP}人以上): {len(lon):,} / 合計 {pop.sum()/1e6:.1f}百万人")
    cxy = np.c_[lon * 111.32 * np.cos(np.radians(36)), lat * 110.57]
    _, nn = cKDTree(ie.xy_km(st)).query(cxy)      # セルの属する地域=最寄り局の県
    cell_f = st.pref_code.values[nn] == "07"
    sd = np.empty(len(cxy))
    for mask, gp in ((~cell_f, gps["その他"]), (cell_f, gps["福島"])):
        idx = np.nonzero(mask)[0]
        for i in range(0, len(idx), 2000):
            j = idx[i:i + 2000]
            _, sd[j] = gp.predict(cxy[j], return_std=True)
    rel = np.exp(sd) - 1                          # 対数σ→相対幅
    score = rel * pop

    # 上位セルに最寄り局名を付ける(場所の手がかり)
    tree = cKDTree(ie.xy_km(st))
    order = np.argsort(score)[::-1]
    top = []
    for k in order[:20]:
        d, j = tree.query(cxy[k])
        top.append({"lon": round(float(lon[k]), 3), "lat": round(float(lat[k]), 3),
                    "pop": int(pop[k]), "rel": round(float(rel[k]) * 100, 1),
                    "score": int(score[k]),
                    "near": st.display_name.iloc[j] or st.station_id.iloc[j],
                    "km": round(float(d), 1)})

    q = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    cells = [[round(float(lon[k]), 3), round(float(lat[k]), 3), int(pop[k]),
              round(float(rel[k]) * 100, 1), int(score[k])] for k in range(len(lon))]
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()[:16],
        "t_max": t_max[:16], "window_h": WINDOW_H,
        "n_train": len(tr), "n_cells": len(cells),
        "edges": [float(x) for x in q],
        # σは大半が飽和し分位点が縮退するため等間隔ビン(近傍の情報勾配を見せる)
        "edges_rel": [float(x) for x in
                      np.linspace(rel.min(), rel.max(), 6)[1:5] * 100],
        "cells": cells, "top": top,
        "stations": [[round(float(r.lon), 3), round(float(r.lat), 3)]
                     for r in st.itertuples()],
        "coast": json.loads((HERE / "assets" / "japan_outline.json").read_text()),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.replace("/*__PAYLOAD__*/null",
                                    json.dumps(payload, ensure_ascii=False,
                                               separators=(",", ":"))))
    covered = pop[score >= q[3]].sum()
    print(f"空白度上位20%セルの人口: {covered/1e6:.1f}百万人")
    print(f"OK -> {OUT} ({OUT.stat().st_size // 1024} KB)")


TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>情報空白マップ（試作・非公式）</title>
<style>
  :root { --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --hair:rgba(11,11,11,.1); --coast:#c3c2b7;
    --seq:#86b6ef,#5598e7,#2a78d6,#1c5cab,#0d366b; }
  @media (prefers-color-scheme: dark) { :root { --page:#0d0d0d; --surface:#1a1a19;
    --ink:#fff; --ink2:#c3c2b7; --hair:rgba(255,255,255,.1); --coast:#4a4a46;
    --seq:#274a78,#3a6aa8,#5590d0,#86b8ea,#bcd9f7; } }
  :root[data-theme="dark"] { --page:#0d0d0d; --surface:#1a1a19; --ink:#fff;
    --ink2:#c3c2b7; --hair:rgba(255,255,255,.1); --coast:#4a4a46;
    --seq:#274a78,#3a6aa8,#5590d0,#86b8ea,#bcd9f7; }
  :root[data-theme="light"] { --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b;
    --ink2:#52514e; --hair:rgba(11,11,11,.1); --coast:#c3c2b7;
    --seq:#86b6ef,#5598e7,#2a78d6,#1c5cab,#0d366b; }
  * { box-sizing:border-box } body { margin:0; background:var(--page); color:var(--ink);
    font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif; }
  header { padding:14px 18px 6px } h1 { font-size:17px; margin:0 }
  .sub { font-size:12px; color:var(--ink2); margin-top:4px }
  .wrap { padding:0 18px 18px; max-width:1100px; margin:0 auto }
  #map { width:100%; height:62vh; min-height:380px; background:var(--surface);
    border:1px solid var(--hair); border-radius:8px; cursor:grab; touch-action:none }
  .legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
    font-size:11.5px; color:var(--ink2); margin:8px 0 }
  .mode { display:flex; gap:6px; margin:0 0 8px }
  .mode button { font:inherit; font-size:12.5px; color:var(--ink2); cursor:pointer;
    background:var(--surface); border:1px solid var(--hair); border-radius:6px;
    padding:5px 12px }
  .mode button.on { color:var(--ink); font-weight:700; border-color:var(--ink2) }
  .sw { display:inline-block; width:14px; height:14px; border-radius:3px;
    vertical-align:-2px; margin-right:4px }
  #tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
    border:1px solid var(--hair); border-radius:6px; padding:6px 9px; font-size:12px;
    display:none; box-shadow:0 2px 10px rgba(0,0,0,.18); z-index:9 }
  table { border-collapse:collapse; font-size:12.5px; margin-top:6px; width:100% }
  th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--hair) }
  th { color:var(--ink2); font-weight:600 } td.num,th.num { text-align:right }
  h2 { font-size:14px; margin:20px 0 2px }
  footer { font-size:11px; color:var(--muted); margin-top:16px; line-height:1.7 }
  a { color:inherit }
</style>
<header>
  <h1>情報空白マップ — 推定が不確かな場所に、どれだけ人が住んでいるか（試作・非公式）</h1>
  <div class="sub" id="sub"></div>
</header>
<div class="wrap">
  <div class="mode" id="mode">
    <button class="on">空白度（σ×人口）</button><button>不確かさ（σのみ）</button>
  </div>
  <canvas id="map"></canvas>
  <div class="legend" id="legend"></div>
  <h2>空白度 上位20セル（表ビュー）</h2>
  <table id="toptable"></table>
  <footer>
    空白度 = クリギング推定の相対不確かさ(1σ, %) × セル居住人口(0.05°≒5kmメッシュ)。
    σは福島／その他で別々に較正し(地域の統計性質が異なるため)、機器ノイズを除いた場の不確かさ。
    各局の「直近3時間窓の最新値」を混合して学習した近似であり、防災判断には使えません。<br>
    データ出典: <a href="https://www.ramis.nra.go.jp/">放射線モニタリング情報共有・公表システム（原子力規制委員会）</a>を加工して作成（非公式）。
    人口: <a href="https://www.worldpop.org/">WorldPop 2020 UN-adjusted 1km</a> (CC BY 4.0)。
    海岸線: Natural Earth。
  </footer>
</div>
<div id="tip"></div>
<script>
"use strict";
const P = /*__PAYLOAD__*/null;
const canvas = document.getElementById("map"), tip = document.getElementById("tip");
const ctx = canvas.getContext("2d");
const mercY = lat => Math.log(Math.tan(Math.PI/4 + lat*Math.PI/360)) * 180/Math.PI;
let view = {scale:1, cx:137, cy:mercY(38)};
const colors = () => getComputedStyle(document.documentElement)
  .getPropertyValue("--seq").split(",").map(s => s.trim());
let mode = 0;                                    // 0=σ×人口 1=σのみ
const valOf = c => mode ? c[3] : c[4];
const edgesOf = () => mode ? P.edges_rel : P.edges;
const binOf = s => { let b = 0; for (const e of edgesOf()) if (s >= e) b++; return b; };

function fit() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  view.scale = Math.min(w/(149-127), h/(mercY(45.8)-mercY(30.5))) * 0.96;
  view.cx = 138; view.cy = mercY(38.2);
}
const sx = lon => canvas.clientWidth/2 + (lon - view.cx) * view.scale;
const sy = lat => canvas.clientHeight/2 - (mercY(lat) - view.cy) * view.scale;

function draw() {
  const dpr = devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cs = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.strokeStyle = cs.getPropertyValue("--coast").trim(); ctx.lineWidth = 0.7;
  for (const line of P.coast) {
    ctx.beginPath();
    for (let i = 0; i < line.length; i++) {
      const x = sx(line[i][0]), y = sy(line[i][1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  const cols = colors(), s = Math.max(1.4, 0.05 * view.scale);
  for (const c of P.cells) {
    ctx.fillStyle = cols[binOf(valOf(c))];
    ctx.fillRect(sx(c[0]) - s/2, sy(c[1]) - s/2, s, s);
  }
}

function legend() {
  const E = edgesOf(), cols = colors();
  const f = mode ? (v => "±" + Math.round(v) + "%")
                 : (v => v >= 1000 ? Math.round(v/1000) + "k" : Math.round(v));
  const lab = [`〜${f(E[0])}`,
    ...E.slice(1).map((e, i) => `${f(E[i])}〜${f(e)}`),
    `${f(E[3])}〜`];
  document.getElementById("legend").innerHTML =
    (mode ? "不確かさ(1σ, 地域較正済み): " : "空白度(σ%×人): ") + lab.map((t, i) =>
      `<span><span class="sw" style="background:${cols[i]}"></span>${t}</span>`).join(" ");
  document.getElementById("sub").textContent =
    `局の最新値 ${P.t_max}(直近${P.window_h}h窓・${P.n_train}局)で学習 / ` +
    `人口セル ${P.n_cells.toLocaleString()}個 / 生成 ${P.generated_at}`;
}

function table() {
  const rows = P.top.map((t, i) => `<tr><td>${i+1}</td>` +
    `<td>${t.near} から ${t.km}km (${t.lat}, ${t.lon})</td>` +
    `<td class="num">${t.pop.toLocaleString()}</td><td class="num">±${t.rel}%</td>` +
    `<td class="num">${t.score.toLocaleString()}</td></tr>`).join("");
  document.getElementById("toptable").innerHTML =
    "<tr><th>#</th><th>場所（最寄り局から）</th><th class=num>人口</th>" +
    "<th class=num>不確かさ1σ</th><th class=num>空白度</th></tr>" + rows;
}

canvas.addEventListener("mousemove", ev => {
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = 8 * 8;
  for (const c of P.cells) {
    const dx = sx(c[0]) - mx, dy = sy(c[1]) - my, d = dx*dx + dy*dy;
    if (d < bd) { bd = d; best = c; }
  }
  if (best) {
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY + 14) + "px";
    tip.innerHTML = `人口 ${best[2].toLocaleString()}人<br>不確かさ ±${best[3]}%` +
      `<br>空白度 ${best[4].toLocaleString()}`;
  } else tip.style.display = "none";
});
canvas.addEventListener("mouseleave", () => tip.style.display = "none");
document.querySelectorAll("#mode button").forEach((b, i) =>
  b.addEventListener("click", () => {
    mode = i;
    document.querySelectorAll("#mode button").forEach((x, j) =>
      x.classList.toggle("on", j === i));
    draw(); legend();
  }));
canvas.addEventListener("wheel", ev => {
  ev.preventDefault();
  view.scale *= ev.deltaY < 0 ? 1.15 : 1/1.15;
  draw();
}, {passive: false});
let drag = null;
canvas.addEventListener("pointerdown", ev => {
  drag = {x: ev.clientX, y: ev.clientY}; canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener("pointermove", ev => {
  if (!drag) return;
  view.cx -= (ev.clientX - drag.x) / view.scale;
  view.cy += (ev.clientY - drag.y) / view.scale;
  drag = {x: ev.clientX, y: ev.clientY}; draw();
});
canvas.addEventListener("pointerup", () => drag = null);
addEventListener("resize", () => { draw(); });
new MutationObserver(() => { draw(); legend(); }).observe(
  document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",
  () => { draw(); legend(); });
fit(); draw(); legend(); table();
</script>
"""


if __name__ == "__main__":
    main()
