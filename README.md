# background-radiation-monitor — 放射線モニタリングデータの収集・可視化・欠測補完

RAMIS(放射線モニタリング情報共有・公表システム、原子力規制委員会)の公開データを
10分ごとに収集して蓄積し、可視化・欠測補完AIにつなげるプロジェクト。
学習・研究用の非公式プロジェクトです。防災判断には必ず公式情報を使用してください。

## 公開運用(GitHub Actions + Pages)

- `.github/workflows/collect.yml` — 10分ごとに全国スナップショット+レーダー雨を `gha_data/raw/` へ追記
- `.github/workflows/pages.yml` — 毎時ダッシュボードを再生成しGitHub Pagesへ公開(`build_site.py`)
- `tools/sync_from_github.py` — git pull後にGHA収集分をローカルSQLiteへ合流(ローカル収集と重複しない)
- コードはMIT、データは出典明示で利用可(詳細は `LICENSE`)

- データ出典: [放射線モニタリング情報共有・公表システム(原子力規制委員会)](https://www.ramis.nra.go.jp/)
- 利用条件: NRAサイトは出典記載を条件に利用可(政府標準利用規約系)。
  公表物には上の出典と利用日を記載すること。
- 注意: 使用しているのは画面用の内部API(公式ドキュメント無し)。仕様変更で壊れる可能性あり。

## 構成(すべてPython標準ライブラリのみ・conda環境不問)

| ファイル | 役割 |
|---|---|
| `ramis_core.py` | APIクライアント + SQLite保存のコア。`python3 ramis_core.py` で自己テスト |
| `collect.py` | 1回分の収集(手動 or launchd/cronで10分ごと) |
| `make_map.py` | 蓄積DBの最新値から自己完結HTMLマップを生成 → `out/ramis_map.html` |
| `tools/build_outline.py` | 日本の海岸線データを一度だけ生成(assets/japan_outline.json) |
| `com.taka.ramis-collect.plist` | launchd用テンプレート(未登録) |
| `data/ramis.sqlite3` | 蓄積DB(stations / measurements / missing_log) |
| `impute_eval.py` | 欠測補完の評価(人工マスク、局分割) ※conda base で実行 |
| `viz_learning.py` | 学習可視化HTML生成 → `out/learn_viz.html` ※conda base で実行 |
| `automl_eval.py` | IDW/GBM/FLAMLの同条件比較 ※conda env `ramis_ml` で実行 |
| `environment_ml.yml` | ramis_ml環境の再現用(`conda env create -f environment_ml.yml`) |
| `tools/fetch_elevation.py` | 全局の標高を取得しstations.elevationに保存(一度だけ) |
| `viz_3d.py` | 標高×線量率の3Dマップ生成 → `out/ramis_3d.html` |

## データが貯まったら(雨の日を1回以上含む1〜2週間後)

```bash
conda activate ramis_ml
cd ~/Desktop/ramis_monitor
python automl_eval.py --budget 300   # 雨天時のみの成績表が自動で出る
conda activate base
python impute_eval.py                 # 従来物差しの再評価
python viz_learning.py                # 可視化を更新
```

## 使い方

```bash
# 単発収集(まず1回)
python3 ~/Desktop/ramis_monitor/collect.py

# 10分ごとの自動収集を開始する場合(登録は任意・いつでも解除可)
cp ~/Desktop/ramis_monitor/com.taka.ramis-collect.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.taka.ramis-collect.plist
# 解除:
launchctl unload ~/Library/LaunchAgents/com.taka.ramis-collect.plist

# マップ生成
python3 ~/Desktop/ramis_monitor/make_map.py && open out/ramis_map.html
```

## データの意味(2026-07-07時点の実測による推定)

- APIエンドポイント: `GET https://www.ramis.nra.go.jp/api/v1/map/map-means-data-public?data_type=N`
  (gzip圧縮JSON)
- data_type: 0=欠測局一覧(236局・故障ラベルに使える) / 1=モニタリングポスト(2,551局・
  うち約756局に風向風速降水量あり) / 2=リアルタイム線量測定システム(3,582局) / 3=その他(35局)
- 値: air_dose_rate はµSv/h、meas_datetime は10分値(ほぼリアルタイム)

## ロードマップ

1. 収集を1〜2週間走らせてデータセット化
2. 地域を絞って欠測補完のベースライン(近傍局の距離重み平均)→LightGBM
   (正常データを人工マスクして正解付き評価)
3. Streamlitで時系列アニメーション・雨と線量率の重ね合わせ・異常検知マップ

`kriging_eval.py` — クリギング(ガウス過程)の精度とσ較正の評価。`conda activate ramis_ml && python kriging_eval.py`

`backfill_eval.py` — 本物の欠測期間での答え合わせ(復旧バックフィル利用)。`ramis_ml`で実行
`rain_radar.py` — 気象庁ナウキャスト実況→全局の頭上の雨(collect.pyから自動実行、単体実行も可)
`quality_watch.py` — 観測網の品質監視(ドリフト/張り付きの点検候補ランキング)。`ramis_ml`で実行
