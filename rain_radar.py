#!/usr/bin/env python3
"""気象庁 高解像度降水ナウキャスト(実況)から全局の頭上の雨強度を取得する。

- タイル: https://www.jma.go.jp/bosai/jmatile/data/nowc/{bt}/none/{vt}/surf/hrpns/{z}/{x}/{y}.png
- z=6 で日本全域を約20タイルでカバー(1ピクセル≈2.4km。RAMIS局の雨特徴には十分)
- PNGは標準ライブラリのみでデコード(zlib+struct。パレット/RGBA・8bit・非インターレースに対応)
- 色→mm/h は気象庁の「雨雲の動き」凡例に基づく(下のPALETTE)。未知色が出たら警告して0扱い

collect.py から毎回呼ばれ、radar_rain テーブルに (station_id, obs_time, mmh) を追記する。
出典: 気象庁「雨雲の動き(高解像度降水ナウキャスト)」
"""
import json
import sqlite3
import struct
import time
import urllib.request
import zlib

BASE = "https://www.jma.go.jp/bosai/jmatile/data/nowc"
UA = {"User-Agent": "ramis-monitor/0.1 (research)", "Accept-Language": "ja"}
Z = 6
X_RANGE = range(53, 60)   # 経度 ≈ 118〜158°E
Y_RANGE = range(22, 29)   # 緯度 ≈ 21〜48°N

# 気象庁 降水強度の凡例色(RGB) → 強度の代表値 mm/h
PALETTE = {
    (242, 242, 255): 0.5,   # 0.1–1
    (160, 210, 255): 3.0,   # 1–5
    (33, 140, 255): 7.5,    # 5–10
    (0, 65, 255): 15.0,     # 10–20
    (250, 245, 0): 25.0,    # 20–30
    (255, 153, 0): 40.0,    # 30–50
    (255, 40, 0): 65.0,     # 50–80
    (180, 0, 104): 90.0,    # 80–
}


def _get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def latest_time():
    d = json.loads(_get(f"{BASE}/targetTimes_N1.json"))
    t = d[0]
    return t["basetime"], t["validtime"]


def decode_png(data):
    """8bit・非インターレースのパレット/RGB/RGBA PNGを (h, w) のRGBタプル配列に展開。"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "PNGではない"
    pos, w = 8, None
    plte, trns, idat = b"", b"", b""
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
            assert interlace == 0, "インターレースPNGは未対応"
            assert (ctype == 3 and depth in (1, 2, 4, 8)) or (ctype in (0, 2, 6) and depth == 8), \
                f"未対応PNG(depth={depth},ctype={ctype})"
        elif typ == b"PLTE":
            plte = chunk
        elif typ == b"tRNS":
            trns = chunk
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + ln
    raw = zlib.decompress(idat)
    nch = {0: 1, 2: 3, 3: 1, 6: 4}[ctype]
    stride = (w * nch * depth + 7) // 8   # 行のバイト数(4bitパレットなら2画素/byte)
    fdist = max(1, (nch * depth) // 8)    # フィルタの参照距離(バイト)
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for row in range(h):
        f = raw[p]; p += 1
        line = bytearray(raw[p:p + stride]); p += stride
        if f == 1:
            for i in range(fdist, stride):
                line[i] = (line[i] + line[i - fdist]) & 255
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                a = line[i - fdist] if i >= fdist else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - fdist] if i >= fdist else 0
                b = prev[i]
                c = prev[i - fdist] if i >= fdist else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out[row * stride:(row + 1) * stride] = line
        prev = line

    def pixel(px, py):
        if ctype == 3:
            bit = px * depth
            byte = out[py * stride + bit // 8]
            idx = (byte >> (8 - depth - bit % 8)) & ((1 << depth) - 1)
            if trns and idx < len(trns) and trns[idx] == 0:
                return None                        # 透明 = 降水なし
            return (plte[idx * 3], plte[idx * 3 + 1], plte[idx * 3 + 2])
        i = (py * w + px) * nch
        if ctype == 6 and out[i + 3] == 0:
            return None
        return (out[i], out[i + 1], out[i + 2])
    return pixel, w


def station_pixels(stations, z=Z):
    """(station_id, lat, lon) → タイル座標とピクセル位置の対応表。"""
    import math
    n = 2 ** z
    table = {}
    for sid, lat, lon in stations:
        xf = (lon + 180) / 360 * n
        yf = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
        tx, ty = int(xf), int(yf)
        px, py = int((xf - tx) * 256), int((yf - ty) * 256)
        table.setdefault((tx, ty), []).append((sid, px, py))
    return table


def sample_stations(stations, verbose=False):
    """最新実況から各局の雨強度をサンプリングする(DB非依存の中核)。

    stations: (id, lat, lon) のリスト
    戻り値: (obs_iso, [(sid, obs_iso, mmh), ...], 未知色set)
    """
    bt, vt = latest_time()
    obs_iso = f"{vt[:4]}-{vt[4:6]}-{vt[6:8]}T{vt[8:10]}:{vt[10:12]}:00+00:00"
    table = station_pixels(stations)
    unknown = set()
    rows = []
    for (tx, ty), pts in sorted(table.items()):
        if tx not in X_RANGE or ty not in Y_RANGE:
            continue
        try:
            png = _get(f"{BASE}/{bt}/none/{vt}/surf/hrpns/{Z}/{tx}/{ty}.png")
            pixel, _ = decode_png(png)
        except Exception as e:
            if verbose:
                print(f"  tile {tx}/{ty}: {e}")
            continue
        for sid, px, py in pts:
            rgb = pixel(px, py)
            if rgb is None:
                mmh = 0.0
            elif rgb in PALETTE:
                mmh = PALETTE[rgb]
            else:
                unknown.add(rgb)
                mmh = 0.0
            rows.append((sid, obs_iso, mmh))
        time.sleep(0.3)
    if unknown and verbose:
        print(f"  未知の色(0扱い): {sorted(unknown)[:8]}")
    return obs_iso, rows, unknown


def update_radar_rain(db_path, verbose=False):
    """最新実況の雨強度を全局ぶん radar_rain テーブルへ追記。戻り値 (時刻, 雨>0局数)。"""
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS radar_rain (
        station_id TEXT NOT NULL, obs_time TEXT NOT NULL, mmh REAL,
        PRIMARY KEY (station_id, obs_time))""")
    stations = conn.execute("""SELECT id, latitude, longitude FROM stations
                               WHERE latitude IS NOT NULL""").fetchall()
    obs_iso, rows, unknown = sample_stations(stations, verbose)
    conn.executemany("INSERT OR IGNORE INTO radar_rain VALUES (?,?,?)", rows)
    conn.commit()
    n_rain = sum(1 for _, _, mmh in rows if mmh > 0)
    return obs_iso, n_rain, len(rows), sorted(unknown)


if __name__ == "__main__":
    from pathlib import Path
    db = Path(__file__).resolve().parent / "data" / "ramis.sqlite3"
    t, n_rain, n_all, unknown = update_radar_rain(db, verbose=True)
    print(f"レーダー実況 {t}: 雨>0 は {n_rain}/{n_all} 局")
    if unknown:
        print(f"注意: 未知色 {unknown}")
