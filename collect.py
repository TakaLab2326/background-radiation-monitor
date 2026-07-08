#!/usr/bin/env python3
"""RAMISの現在値を1回取得してSQLiteに追記する収集スクリプト。

10分ごとの定期実行を想定(launchd/cron)。単発で手動実行してもよい。
使い方:
    python3 collect.py                # data/ramis.sqlite3 に追記
    python3 collect.py --db 別パス.sqlite3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ramis_core as rc

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "ramis.sqlite3"
LOG_PATH = HERE / "data" / "collect.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = rc.open_db(args.db)
    fetched_at = rc.now_iso()
    parts, failures = [], 0

    for dt in (1, 2, 3):  # 測定値のある系統
        try:
            recs = rc.fetch_snapshot(dt)
            n_st, n_new = rc.save_snapshot(conn, recs, dt, fetched_at)
            parts.append(f"type{dt}:{n_st}局/新規{n_new}")
        except Exception as e:
            failures += 1
            parts.append(f"type{dt}:ERROR {e}")
        time.sleep(2)  # 連続アクセスの間隔を空ける(サーバへの配慮)

    try:  # 欠測局一覧 = 故障ラベル
        recs = rc.fetch_snapshot(0)
        n = rc.save_missing(conn, recs, fetched_at)
        parts.append(f"欠測:{n}局")
    except Exception as e:
        failures += 1
        parts.append(f"欠測:ERROR {e}")

    try:  # 気象庁レーダー実況(全局の頭上の雨)。失敗しても収集は続行
        import rain_radar
        _, n_rain, n_all, _ = rain_radar.update_radar_rain(args.db)
        parts.append(f"レーダー雨:{n_rain}/{n_all}局")
    except Exception as e:
        parts.append(f"レーダー雨:ERROR {e}")

    line = f"{fetched_at} " + " ".join(parts)
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    sys.exit(1 if failures == 4 else 0)  # 全滅時のみ異常終了


if __name__ == "__main__":
    main()
