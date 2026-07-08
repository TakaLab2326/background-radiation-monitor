# PROGRESS

## 2026-07-08 未明
- 済: 収集コア(ramis_core/collect)・試作マップ(make_map→Artifact公開)・欠測補完の評価1回目(impute_eval.py)
- 結果: 故障局3.3h外挿=自局過去平均でMAE 0.005 µSv/h / 履歴なし局=IDWでMAE 0.024(中央値誤差13%)
- 方針: GBMはIDWと同等(データ不足)。雨イベント蓄積後に雨特徴の効果を再評価する
- 残り: launchd常時収集が未開始(ユーザー操作待ち)。雨天後にimpute_eval再実行。Streamlit時系列は未着手
- 環境: 収集系=/usr/bin/python3(標準ライブラリのみ) / ML系=conda base(sklearn 1.5.1既存、追加インストールなし)

## 2026-07-08 未明 (2)
- 済: 学習可視化(viz_learning.py + learn_viz_template.html → Artifact公開)。5-fold OOFで全3,640局を評価
- 結果: MAE idw=0.0181 / gbm=0.0187。重要度1位=近傍IDW平均(0.642)で他を圧倒
- 修正: 線量率0の局(ニセコヘリポート)を slot_frame で除外(ゼロ除算対処)
- 残り: launchd常時収集は依然未開始。雨蓄積後に impute_eval / viz_learning を再実行

## 2026-07-08 未明 (3)
- 済: AutoML比較基盤(automl_eval.py) + 専用env ramis_ml(conda-forge: flaml/lgbm/xgb) + environment_ml.yml
- スモーク(30秒/モード): 履歴なし=FLAML(rf) 0.0211でIDWと同着首位 / 故障局=過去平均0.0053が依然最強(乾燥夜間なので想定通り)
- 雨天行が貯まると自動で「雨天時のみ」表が出る設計。本番は --budget 300 以上を推奨
- 残り: launchd常時収集の開始(ユーザー操作)。雨イベント後に automl_eval.py を再実行

## 2026-07-08 早朝 (4) 標高
- 済: 全5,338局の標高取得(Open-Meteo/Copernicus DEM, tools/fetch_elevation.py, 429バックオフ対応)
- 済: elev/elev_rel特徴を impute_eval.neighbor_features に追加。3D地図(viz_3d.py)をArtifact公開
- 結果(正直): アブレーションで標高は精度に効かず(MAE 0.0241→0.0244, ノイズ内)。重要度6位=見てはいるが近傍値と冗長
- 期待: 雨イベント時の「地形×降水」の相互作用で効く可能性 → 雨蓄積後に再アブレーション

## 2026-07-08 早朝 (5) 残差学習+過去データ調査
- 済: 残差学習をautoml_evalへ正式組込(モード1=IDW基準/モード2=過去平均基準)。故障局モードでGBM誤差半減(0.0111→0.0058)
- 知見: エリア/標高のハード分割は無効。福島距離特徴(D案)が最良だが未組込(ユーザー指示待ち)
- 過去データ: 有料販売は無い。RAMIS UIにCSV DL(無料)、県ポータル、JAEA EMDB(2011年〜)が入手先。一括APIは自治体向け認証
- RAMIS APIメモ: /view/map-table-display-moni-public は400(パラメータ次第で公開の可能性、未解明)

## 2026-07-08 早朝 (6) 福島距離+EMDB
- 済: dist_f1(福島第一からの距離)を neighbor_features に正式組込。GBM直接 0.0240→0.0212(IDW同着、RMSEは明確に上回る)。重要度3位
- 済: 学習可視化Artifactを同URLで更新
- EMDB: サイト全体が500エラーで停止中(全URL→/emdbリダイレクト→Server Error)。復旧後に再調査。狙いは①過去の雨イベント時系列②Cs-137土壌沈着量マップ(dist_f1の上位互換特徴)

## 2026-07-08 早朝 (7) EMDB調査(ユーザー提供URL)
- 判明: EMDBの正URLは radioactivity.nra.go.jp/emdb/。Accept-Languageヘッダ無しだと500(前回「停止中」判定は誤り)
- カタログ163項目取得済(scratchpadのemdb_items.json)。本命: item 449=全国モニタリングポスト時系列 2011/03〜2026/06・85万件 / content 12=Cs沈着量9項目(dist_f1の上位互換特徴候補)
- 利用条件: 出典明示で利用可【確認済み】
- DL経路: /emdb/download/zip/{item}/{年度}/{文字}/{言語} と /emdb/search/data。ただし探索中にタイムアウト化(IP絞りの可能性)→深追い中止
- 次: 時間を置いて、ブラウザで実際の検索リクエストを1回観察(Chrome拡張のread_network_requests)→それを模倣する礼儀正しいfetch_emdb.pyを作る

## 2026-07-08 早朝 (8) 基準値比+クリギング
- 済: 基準値比モデル(目的も特徴も平常値からの変動比)をautoml_eval故障局モードに追加 → ML最良(中央値誤差4.76%、過去平均4.50%に肉薄)
- 済: kriging_eval.py 新規(Matérn GP、サブサンプルでカーネル学習→全局固定適用)
- 結果: GPの精度はIDWと同等(MAE 0.0207 vs 0.0199)。価値はσ: ±2σ被覆95.6%(理想95%)とほぼ完璧な較正
- 欠測局216局のσつき推定: 相対不確かさ中央値±27%、最悪は延岡局±71%(孤立局)
- 次: 欠測局マップの◆にσを載せる可視化統合(未着手)。雨蓄積後に基準値比の真価を再評価

## 2026-07-08 朝 (9) 推奨3案の実装
- A1済: backfill_eval.py — 本物の欠測3行で初評価(過去平均 誤差1.3%)。fetched_atで「当時知り得た情報」を再現する設計
- A2済: rain_radar.py — 気象庁hrpnsタイル(z6・4bitパレットPNG自前デコード)→全5,340局のradar_mmh。菊川市の0.5mm/hを検出し色マップ検証済み。collect.pyに統合、automl_evalの特徴にも追加(全NaN列は自動除外)
- B1済: quality_watch.py — ズレ候補ランキング(福島圏±6-12%が上位、断面5枚の暫定)+張り付き検知(現在0局)
- 最新評価(5断面14,384行): 履歴なし GBM直接0.0200=IDW同着 / 故障局 基準値比0.0055(過去平均0.0052に肉薄)
- 未決: 収集の常時稼働(launchd or GitHub Actions)。全部これ待ち

## 2026-07-08 朝 (10) 常時収集の開始
- 決定: 「今はlaunchdのみ」(GHA privateはPro3,000分/月でも10分間隔=4,320分に不足。Publicなら無料、必要なら月1,600円で私有可、と説明済み)
- 実測で確定: RAMIS APIのlatest_get_datetimeは履歴を再生しない(復旧局バックフィルのみ)→収集の穴=恒久欠損。間引き運用は不可
- ハマり: DesktopはTCC保護でlaunchdから読めない → 実体を ~/ramis_monitor へ移動、Desktopにはシンボリックリンク
- 稼働開始: com.taka.ramis-collect (10分間隔+RunAtLoad) 08:26初回成功、レーダー雨込み
- EMDB: アクセス規制解除を確認(200)。次回=年次zip/検索APIの経路解明→過去15年分の取り込み

## 2026-07-08 朝 (11) EMDB解読完了+データの正体判明
- 検索API解読: POST /emdb/search/data (CSRF=csrftokenクッキー+X-CSRFToken、term=月インデックス=(年-2000)*12+月-1、content/types/items/prefecture、jtable頁指定)
- CSV: #form-download へ ids をPOST(検索応答のids文字列を使用)
- 判明: item 449 = 「環境放射能水準調査」= **日次値×約50-80地点(都道府県代表局・測定高2種)**。RAMISの10分値×6,400局とは別事業・別粒度
- 帰結: 雨スパイク(30-120分)の検証には日次は粗い。EMDBの用途=長期ベースライン/ドリフト基準/季節変動。10分値の過去アーカイブは公開されていない可能性が高い→自前収集が唯一の道(稼働中で正解)

## 2026-07-08 朝 (12) 統合ダッシュボード
- 済: viz_hub.py — 3可視化を<template>+iframe(srcdoc遅延展開)でタブ統合。名前空間分離で既存3ページ無改造
- テーマはハブ→iframeへ転送、タブ切替時にresizeを送って再描画。2.1MB
- 統合Artifact: https://claude.ai/code/artifact/85c720e3-4da2-476e-88db-4fc26236a489
- 更新手順: 3つの生成スクリプト→viz_hub.py→同URLへ再公開
