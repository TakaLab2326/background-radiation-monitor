#!/usr/bin/env python3
"""RAMIS 放射線モニタリング — 選んで・見て・学習する (Streamlit)

データを選ぶ → 可視化 → ボタンで機械学習(欠測補完の練習問題)まで、
専門知識なしで体験できるアプリ。

デプロイ: Streamlit Community Cloud (リポジトリの app ブランチ)
データ: 同リポジトリ main ブランチの gha_data/ を実行時に取得(10分キャッシュ)
出典: 放射線モニタリング情報共有・公表システム(原子力規制委員会) / 気象庁
"""
import gzip
import io
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

REPO = "TakaLab2326/background-radiation-monitor"
TARBALL = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"
K = 5  # 近傍局数

PREF = {f"{i:02d}": n for i, n in enumerate(
    ["北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県",
     "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県",
     "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県", "三重県",
     "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県", "島根県",
     "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県",
     "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"], start=1)}
TYPE_NAMES = {1: "モニタリングポスト", 2: "リアルタイム線量計", 3: "その他"}
VARS = {"空間線量率 (µSv/h)": "air_dose_rate", "風速 (m/s)": "wind_speed",
        "降水量 (mm)": "precipitation", "日射量": "solar_amount",
        "計数率 (cpm)": "counting_rate", "標高 (m)": "elevation"}

# 線量率5ビンの色(検証済みパレット・ライト用)
BIN_EDGES = [0.05, 0.1, 0.3, 1.0]
BIN_COLORS = [[134, 182, 239], [85, 152, 231], [42, 120, 214], [28, 92, 171], [13, 54, 107]]
BIN_LABELS = ["< 0.05", "0.05–0.1", "0.1–0.3", "0.3–1", "≥ 1"]

st.set_page_config(page_title="RAMIS 選んで学ぶ放射線モニタリング", page_icon="📡",
                   layout="wide")


@st.cache_data(ttl=600, show_spinner="GitHubから最新データを取得中…")
def load_data():
    """mainブランチのtarballを一括取得して測定/局マスタ/レーダーを展開する。"""
    req = urllib.request.Request(TARBALL, headers={"User-Agent": "ramis-app"})
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = io.BytesIO(r.read())
    meas_parts, radar_parts, stations = [], [], None
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for m in tar.getmembers():
            if "/gha_data/" not in m.name or not m.name.endswith(".csv.gz"):
                continue
            data = gzip.decompress(tar.extractfile(m).read())
            df = pd.read_csv(io.BytesIO(data), dtype=str)   # 先頭ゼロ保護のため全て文字列で読む
            if m.name.endswith("stations.csv.gz"):
                stations = df
            elif m.name.endswith("_radar.csv.gz"):
                radar_parts.append(df)
            elif not m.name.endswith("_missing.csv.gz"):
                meas_parts.append(df)
    meas = pd.concat(meas_parts, ignore_index=True).drop_duplicates(
        ["station_id", "meas_datetime"])
    radar = (pd.concat(radar_parts, ignore_index=True).drop_duplicates()
             if radar_parts else pd.DataFrame(columns=["station_id", "obs_time", "mmh"]))
    stations = stations.set_index("id")
    for c in ("latitude", "longitude", "elevation"):
        stations[c] = pd.to_numeric(stations[c], errors="coerce")
    meas = meas.join(stations[["display_name", "pref_code", "latitude", "longitude",
                               "elevation"]], on="station_id")
    for c in ("air_dose_rate", "wind_speed", "precipitation", "counting_rate",
              "solar_amount"):
        meas[c] = pd.to_numeric(meas[c], errors="coerce")
    meas["data_type"] = pd.to_numeric(meas.data_type, errors="coerce").astype("Int64")
    meas.loc[meas.solar_amount < 0, "solar_amount"] = np.nan   # -99等は欠測記号
    meas = meas.dropna(subset=["latitude", "air_dose_rate"])
    meas = meas[meas.air_dose_rate > 0]
    meas["t"] = pd.to_datetime(meas.meas_datetime)
    return meas, stations, radar


def dose_bin(v):
    return int(np.searchsorted(BIN_EDGES, v, side="right"))


# ---------------- サイドバー: データ選択 ----------------
meas, stations, radar = load_data()

st.sidebar.title("① データを選ぶ")
prefs = st.sidebar.multiselect(
    "都道府県(空欄=全国)", options=list(PREF), format_func=lambda c: PREF[c])
types = [t for t in (1, 2, 3)
         if st.sidebar.checkbox(TYPE_NAMES[t], value=(t != 3), key=f"ty{t}")]
times = sorted(meas.meas_datetime.unique())
if len(times) > 1:
    t0, t1 = st.sidebar.select_slider(
        "期間(実在する測定時刻が目盛り)", options=times, value=(times[0], times[-1]),
        format_func=lambda v: v[5:16].replace("T", " "))
else:
    t0, t1 = times[0], times[-1]

sel = meas[meas.data_type.isin(types)
           & meas.meas_datetime.between(t0, t1)
           & (meas.pref_code.isin(prefs) if prefs else True)]
st.sidebar.markdown(
    f"**選択中: {sel.station_id.nunique():,}局 / {len(sel):,}行**\n\n"
    f"({sel.meas_datetime.nunique()}時点)")

st.title("📡 選んで学ぶ 放射線モニタリング")
st.caption("全国の空間線量率(10分値)を選んで可視化し、そのまま機械学習を体験できます。"
           "データは10分ごとに自動収集(GitHub Actions)。")

PAGES = ["② 可視化", "②' 関係と外れ値", "③ 機械学習を体験", "仕組みと出典"]
page = st.radio("ページ", PAGES, horizontal=True, key="page",
                label_visibility="collapsed")

# ---------------- 可視化 ----------------
if page == PAGES[0]:
    if sel.empty:
        st.warning("選択条件に合うデータがありません。条件を広げてください。")
        st.stop()

    vlabel = st.selectbox("表示する項目(モニタリング値のほか気象・地形も選べます)",
                          list(VARS), key="var")
    vcol = VARS[vlabel]

    latest = sel.sort_values("meas_datetime").groupby("station_id").tail(1).copy()
    latest["val"] = latest[vcol]
    shown = latest.dropna(subset=["val"])
    if shown.empty:
        st.warning(f"「{vlabel}」の値を持つ局が選択範囲にありません。"
                   "気象の項目はセンサ付きの局(全体の1〜2割)だけが持っています。")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("値のある局", f"{len(shown):,} / {len(latest):,}")
    c2.metric("中央値", f"{shown.val.median():.3g}")
    c3.metric("最大", f"{shown.val.max():.3g}")
    c4.metric("最新測定", shown.meas_datetime.max()[5:16].replace("T", " "))

    # 色分け: 線量率は固定しきい値、その他は五分位(データに合わせて自動)
    if vcol == "air_dose_rate":
        edges = list(BIN_EDGES)
    elif shown.val.nunique() <= 1:
        edges = []
    else:
        edges = sorted(set(float(q) for q in
                           np.quantile(shown.val, [0.2, 0.4, 0.6, 0.8])))
    colors = BIN_COLORS[:len(edges) + 1] if edges else [BIN_COLORS[2]]

    def color_of(v):
        return colors[min(int(np.searchsorted(edges, v, side="right")),
                          len(colors) - 1)] if edges else colors[0]

    pts = shown.assign(color=shown.val.map(color_of),
                       v=shown.val.map(lambda x: f"{x:.4g}"),
                       name=shown.display_name.fillna(shown.station_id))
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=float(pts.latitude.mean()), longitude=float(pts.longitude.mean()),
            zoom=4.2 if not prefs else 6.5),
        layers=[pdk.Layer("ScatterplotLayer", data=pts[
            ["longitude", "latitude", "color", "v", "name"]],
            get_position=["longitude", "latitude"], get_fill_color="color",
            get_radius=2500, pickable=True)],
        tooltip={"text": "{name}\n{v}"}))
    if edges:
        lg = ([f"〜{edges[0]:.3g}"]
              + [f"{a:.3g}〜{b:.3g}" for a, b in zip(edges[:-1], edges[1:])]
              + [f"{edges[-1]:.3g}〜"])
        st.caption(f"凡例({vlabel}): " + " / ".join(lg) + " — 薄い青→濃い青の順に大きい")
    else:
        st.caption(f"凡例({vlabel}): 選択範囲では全局ほぼ同じ値です")

    if sel.meas_datetime.nunique() >= 3 and vcol != "elevation":
        st.subheader(f"時間変化 — {vlabel}(選択局の中央値)")
        tsv = sel.dropna(subset=[vcol]).groupby("meas_datetime")[vcol].median()
        tsv.index = tsv.index.str.slice(5, 16)
        st.line_chart(tsv, height=200)

        st.subheader("局を選んで時間変化を比べる")
        opts = shown.sort_values("val", ascending=False)
        id2name = dict(zip(opts.station_id, opts.display_name.fillna(opts.station_id)))
        picks = st.multiselect(f"局({vlabel}の大きい順に並んでいます)",
                               options=list(opts.station_id),
                               default=list(opts.station_id[:3]),
                               format_func=lambda i: id2name.get(i, i))
        if picks:
            wide = (sel[sel.station_id.isin(picks)]
                    .pivot_table(index="meas_datetime", columns="station_id",
                                 values=vcol)
                    .rename(columns=id2name))
            wide.index = wide.index.str.slice(5, 16)
            st.line_chart(wide, height=260)
            st.caption(f"{vlabel}。雨の通過や昼夜の変化が見えることがあります。")

# ---------------- 関係と外れ値 ----------------
elif page == PAGES[1]:
    st.markdown("**特徴量どうしの関係を調べる** — 軸を選ぶと局ごとの最新値で散布図を描きます。"
                "関係が見える組み合わせ=学習に効く特徴、ポツンと離れた点=外れ値候補です。")
    if sel.empty:
        st.warning("選択条件に合うデータがありません。")
        st.stop()
    latest = sel.sort_values("meas_datetime").groupby("station_id").tail(1).copy()

    cx, cy = st.columns(2)
    xlabel = cx.selectbox("横軸", list(VARS), index=1, key="relx")
    ylabel = cy.selectbox("縦軸", list(VARS), index=0, key="rely")
    xcol, ycol = VARS[xlabel], VARS[ylabel]
    df = latest.dropna(subset=[xcol, ycol]).copy()
    df["name"] = df.display_name.fillna(df.station_id)
    if len(df) < 10 or xcol == ycol:
        st.warning("この組み合わせで両方の値を持つ局が10局未満か、同じ項目です。"
                   "軸を変えてください(気象はセンサ付き局のみが持ちます)。")
        st.stop()

    def robust_z(v):
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        return (v - med) / mad if mad > 0 else v * 0.0

    thr = st.slider("外れ値のきびしさ(ロバストzスコア。小さいほど多く検出)",
                    2.0, 6.0, 3.5, 0.5)
    zx, zy = robust_z(df[xcol]), robust_z(df[ycol])
    df["判定"] = np.where((zx.abs() > thr) | (zy.abs() > thr), "外れ値候補", "通常")
    n_out = int((df["判定"] == "外れ値候補").sum())

    logy = st.checkbox("縦軸を対数にする(線量率のように裾が長い項目で見やすい)",
                       value=(ycol == "air_dose_rate"))
    plot = pd.DataFrame({xlabel: df[xcol],
                         ylabel: np.log10(df[ycol]) if logy and (df[ycol] > 0).all()
                         else df[ycol], "判定": df["判定"]})
    if logy and (df[ycol] > 0).all():
        plot = plot.rename(columns={ylabel: f"log10 {ylabel}"})
    st.scatter_chart(plot, x=xlabel, y=plot.columns[1], color="判定", height=420)

    r = df[[xcol, ycol]].corr(method="spearman").iloc[0, 1]
    m1, m2 = st.columns(2)
    m1.metric("関係の強さ(スピアマン相関)", f"{r:+.2f}")
    m2.metric("外れ値候補", f"{n_out}局 / {len(df)}局")
    st.caption("相関の目安: ±0.7以上=強い / ±0.4=中くらい / 0付近=ほぼ無関係。"
               "相関が強い特徴は学習に効きやすい(ただし偶然の相関もある)。")

    if n_out:
        worst = df.assign(zmax=np.maximum(zx.abs(), zy.abs())).nlargest(
            min(10, n_out), "zmax")
        st.dataframe(worst[["name", xcol, ycol, "zmax"]].rename(columns={
            "name": "局名", xcol: xlabel, ycol: ylabel, "zmax": "zスコア"}).round(3),
            width="stretch", hide_index=True)
        st.session_state["outlier_ids"] = set(df.loc[df["判定"] == "外れ値候補",
                                                     "station_id"])
        st.checkbox(f"⬇ この外れ値候補 {n_out}局 を「③機械学習」の学習から除外する",
                    key="exclude_outliers")
    with st.expander("⚠ 外れ値は外せばよいとは限らない(重要)"):
        st.markdown(
            "外れ値には2種類あります。**(1) 機器の異常や入力ミス**(例: ありえない負の値)"
            "は外すべきですが、**(2) 本当にそういう値の場所**(例: 福島の帰還困難区域の高い"
            "線量率)は本物のデータなので、外すと「高い場所を知らないモデル」になって"
            "しまいます。地図やニュースと照らして、どちらのタイプか考えてから除外するのが"
            "正しい手順です。除外して精度がどう変わるか比べるのも良い実験です。")

# ---------------- 機械学習 ----------------
elif page == PAGES[2]:
    st.markdown("""
**やること: 「もしこの観測局が故障したら、周りの局から値を当てられる?」という実験**

モニタリングポストは時々故障して値が取れなくなります。そんなとき周りの局から
値を推測できると便利ですが、本当に故障した局には「正解」が無いので、
推測が当たっているか確かめられません。そこでこのアプリは:

1. 🙈 **正常に動いている局の一部を、わざと「故障したフリ」にします**(測定値を隠す)
2. 🤖 残りの局のデータだけを使って、コンピュータが隠した局の値を予測します
3. ✅ 隠しておいた**本当の測定値**と比べて答え合わせ(採点)します

正解を知ったうえで隠しているので、予測の実力を正確に測れます。
下のボタンでこの「目隠しテスト」が動きます。
""")
    with st.expander("採点でズルをさせない工夫(なぜ「局ごと」に分けるの?)"):
        st.markdown(
            "テストに出る局のデータが1つでも練習に混ざっていると、コンピュータは答えを"
            "丸暗記できてしまい、実力以上の点数が出ます(カンニングと同じ)。このアプリは"
            "**テスト用に選んだ局のデータを練習から完全に除外**してから採点しています。")

    n_st = sel.station_id.nunique()
    if n_st < 100:
        st.warning(f"選択中の局が{n_st}局です。学習には100局以上を推奨します"
                   "(都道府県の選択を広げてみてください)。")

    excl_ids = (st.session_state.get("outlier_ids", set())
                if st.session_state.get("exclude_outliers") else set())
    if excl_ids:
        st.info(f"「関係と外れ値」ページで指定した外れ値候補 **{len(excl_ids)}局** を"
                "除外して学習します(チェックを外せば戻ります)。")

    ML_KINDS = ["勾配ブースティング(おすすめ)", "ランダムフォレスト", "リッジ回帰(線形)",
                "k近傍回帰", "かんたんAutoML(4種類を自動比較して勝者を採用)"]
    kind = st.selectbox("学習方法を選ぶ", ML_KINDS,
                        help="AutoMLは4種類のモデルを検証用の局で比較し、一番良いものを自動採用します")

    if st.button("🎯 学習を実行(数十秒〜1分)", type="primary", disabled=sel.empty):
        from scipy.spatial import cKDTree
        from sklearn.ensemble import (HistGradientBoostingRegressor,
                                      RandomForestRegressor)
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def make_model(name):
            if name.startswith("勾配"):
                return HistGradientBoostingRegressor(max_iter=300, random_state=0)
            if name.startswith("ランダム"):
                return make_pipeline(SimpleImputer(), RandomForestRegressor(
                    n_estimators=150, n_jobs=-1, random_state=0))
            if name.startswith("リッジ"):
                return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))
            return make_pipeline(SimpleImputer(), StandardScaler(),
                                 KNeighborsRegressor(n_neighbors=10))

        sel_ml = sel[~sel.station_id.isin(excl_ids)] if excl_ids else sel
        with st.spinner("特徴量を作成中(各局の近傍5局を探索)…"):
            frames = []
            for ts_key, g in sel_ml.groupby("meas_datetime"):
                g = g.drop_duplicates("station_id")
                if len(g) < 30:
                    continue
                xy = np.c_[g.longitude * 91.2, g.latitude * 110.6]
                d, idx = cKDTree(xy).query(xy, k=min(K, len(g) - 1) + 1)
                d, idx = d[:, 1:], idx[:, 1:]
                logv = np.log(g.air_dose_rate.values)
                w = 1 / np.maximum(d, 0.1)
                F = pd.DataFrame({
                    "近傍IDW平均": (logv[idx] * w).sum(1) / w.sum(1),
                    "近傍1位の値": logv[idx[:, 0]], "近傍1位までの距離": d[:, 0],
                    "近傍のばらつき": logv[idx].std(1),
                    "緯度": g.latitude.values, "経度": g.longitude.values,
                    "標高": pd.to_numeric(g.elevation, errors="coerce").values,
                    "風速": pd.to_numeric(g.wind_speed, errors="coerce").values,
                    "降水量": pd.to_numeric(g.precipitation, errors="coerce").values,
                })
                F["station_id"], F["logy"], F["y"] = \
                    g.station_id.values, logv, g.air_dose_rate.values
                F["name"] = g.display_name.fillna(g.station_id).values
                frames.append(F)
            if not frames:
                st.error(
                    "選択されたデータが少なすぎて学習できません(1時点あたり30局以上が必要)。"
                    "都道府県の選択を増やす・空欄(全国)にする・期間を広げる、のいずれかを"
                    "試してください。")
                st.stop()
            data = pd.concat(frames, ignore_index=True)

        feats = [c for c in data.columns if c not in
                 ("station_id", "logy", "y", "name") and data[c].notna().any()]
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                      .split(data, groups=data.station_id))
        TR, TE = data.iloc[tr], data.iloc[te]

        if kind.startswith("かんたんAutoML"):
            # 学習用の局をさらに2つに分け、検証側の成績で4モデルを比較(テスト局は不使用)
            tr2, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=1)
                           .split(TR, groups=TR.station_id))
            TR2, VA = TR.iloc[tr2], TR.iloc[va]
            scores = {}
            for name in ML_KINDS[:4]:
                with st.spinner(f"AutoML: {name} を試験中…"):
                    m = make_model(name)
                    m.fit(TR2[feats], TR2.logy)
                    pv = np.exp(m.predict(VA[feats]))
                    scores[name] = np.abs(pv - VA.y.values).mean()
            kind_won = min(scores, key=scores.get)
            st.info("**AutoMLの比較結果**(検証用の局での平均誤差 µSv/h) → "
                    f"勝者: **{kind_won}**")
            st.dataframe(pd.Series(scores, name="検証誤差").round(4).to_frame(),
                         width="stretch")
            model_name = kind_won
        else:
            model_name = kind

        with st.spinner(f"学習中({model_name})…"):
            model = make_model(model_name)
            model.fit(TR[feats], TR.logy)
        y, p_idw = TE.y.values, np.exp(TE["近傍IDW平均"].values)
        p_ml = np.exp(model.predict(TE[feats]))

        st.success(f"完了! 学習 {TR.station_id.nunique():,}局 / "
                   f"試験 {TE.station_id.nunique():,}局(学習には未使用) / "
                   f"採用モデル: {model_name}")
        res = pd.DataFrame({
            "平均誤差 µSv/h": [np.abs(p_idw - y).mean(), np.abs(p_ml - y).mean()],
            "誤差の中央値 %": [np.median(np.abs(p_idw - y) / y) * 100,
                          np.median(np.abs(p_ml - y) / y) * 100],
        }, index=["近くの局の平均(IDW)", f"機械学習({model_name})"]).round(4)
        st.dataframe(res, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**予測 vs 実測**(対角線上=完全一致)")
            st.scatter_chart(pd.DataFrame({
                "実測 µSv/h": y, "予測 µSv/h": p_ml}),
                x="実測 µSv/h", y="予測 µSv/h", height=320)
        with col2:
            st.markdown("**モデルは何を重視した?**")
            sub = TE.sample(min(800, len(TE)), random_state=0)
            imp = permutation_importance(model, sub[feats], sub.logy,
                                         n_repeats=3, random_state=0)
            st.bar_chart(pd.Series(imp.importances_mean, index=feats)
                         .sort_values(), height=320, horizontal=True)
        worst = TE.assign(err=np.abs(p_ml - y) / y).nlargest(5, "err")
        st.markdown("**外した局トップ5**(なぜ外れたか考えてみよう — 周りに局が少ない?"
                    "線量の変化が急な地域?)")
        st.dataframe(worst[["name", "y", "err"]].assign(
            err=lambda d: (d.err * 100).round(1)).rename(columns={
                "name": "局名", "y": "実測 µSv/h", "err": "誤差 %"}),
            width="stretch", hide_index=True)

# ---------------- 仕組み ----------------
else:
    st.markdown(f"""
### 仕組み
- **データ**: [放射線モニタリング情報共有・公表システム(原子力規制委員会)](https://www.ramis.nra.go.jp/)
  の公開データを GitHub Actions が10分ごとに自動収集。雨は気象庁の高解像度降水ナウキャスト。
- **このアプリ**: [公開リポジトリ]({f"https://github.com/{REPO}"}) からデータを取得して動作。
  リアルタイムの[全国ダッシュボードはこちら](https://takalab2326.github.io/background-radiation-monitor/)。
- **機械学習**: 勾配ブースティング決定木(scikit-learn)。評価は局単位分割で情報リークを防止。
- 本アプリは学習・研究用の非公式プロジェクトです。防災判断には必ず公式情報を使用してください。
- 単位 µSv/h = マイクロシーベルト毎時。日本の平常時はおおむね 0.02〜0.1 µSv/h。雨で一時的に
  上がるのは自然現象(大気中のラドン子孫核種が雨で地表に運ばれるため)です。
""")
