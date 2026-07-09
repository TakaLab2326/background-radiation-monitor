#!/usr/bin/env python3
"""3つの可視化(全国マップ/学習・精度/3D標高)をタブ切り替えの1ページに統合する。

各ページは <template> に丸ごと格納し、タブ選択時に iframe(srcdoc) として展開。
名前空間が完全に分離されるので既存3ページは無改造で同居できる。
使い方: 先に make_map.py / viz_learning.py / viz_3d.py を実行してから
        python3 viz_hub.py
出力: out/ramis_hub.html
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

PAGES = [
    ("ramis_map.html", "全国マップ"),
    ("learn_viz.html", "学習・精度"),
    ("ramis_3d.html", "3D 標高"),
]

TEMPLATE = """<meta charset="utf-8">
<title>RAMIS 非公式ダッシュボード（試作）</title>
<style>
  :root {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --hairline: rgba(11,11,11,0.10); --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root { --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
            --hairline: rgba(255,255,255,0.10); --accent: #3987e5; }
  }
  :root[data-theme="dark"] {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --hairline: rgba(255,255,255,0.10); --accent: #3987e5;
  }
  :root[data-theme="light"] {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --hairline: rgba(11,11,11,0.10); --accent: #2a78d6;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", "Hiragino Sans", sans-serif;
    display: flex; flex-direction: column; }
  nav {
    display: flex; gap: 4px; align-items: center; padding: 8px 14px 0;
    border-bottom: 1px solid var(--hairline); background: var(--surface);
    flex: 0 0 auto;
  }
  nav .brand { font-size: 12px; font-weight: 700; color: var(--ink-2);
    margin-right: 10px; letter-spacing: .08em; }
  nav button {
    appearance: none; border: none; background: none; cursor: pointer;
    font: inherit; font-size: 13px; color: var(--ink-2);
    padding: 8px 14px 9px; border-bottom: 2px solid transparent;
  }
  nav button[aria-selected="true"] {
    color: var(--ink); font-weight: 700; border-bottom-color: var(--accent);
  }
  nav button:hover { color: var(--ink); }
  nav button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  main { flex: 1 1 auto; min-height: 0; position: relative; }
  iframe { position: absolute; inset: 0; width: 100%; height: 100%;
    border: none; display: none; background: var(--page); }
  iframe.active { display: block; }
</style>

<nav role="tablist" aria-label="表示切り替え">
  <span class="brand">RAMIS 試作（非公式）</span>
  __TABS__
</nav>
<main>__FRAMES__</main>

__TEMPLATES__

<script>
"use strict";
const frames = [...document.querySelectorAll("iframe")];
const tabs = [...document.querySelectorAll("nav button")];

function applyTheme() {
  const t = document.documentElement.dataset.theme || "";
  for (const f of frames) {
    try {
      if (f.contentDocument) {
        if (t) f.contentDocument.documentElement.dataset.theme = t;
        else delete f.contentDocument.documentElement.dataset.theme;
      }
    } catch (e) {}
  }
}

function show(i) {
  tabs.forEach((b, j) => b.setAttribute("aria-selected", j === i));
  frames.forEach((f, j) => {
    f.classList.toggle("active", j === i);
    if (j === i) {
      if (!f.srcdoc) {                       // 初回のみ展開(遅延ロード)
        f.srcdoc = document.getElementById(`pg${j}`).innerHTML;
        f.addEventListener("load", () => {
          applyTheme();
          f.contentWindow.dispatchEvent(new f.contentWindow.Event("resize"));
        }, { once: true });
      } else if (f.contentWindow) {          // 表示切替後に再描画させる
        f.contentWindow.dispatchEvent(new f.contentWindow.Event("resize"));
      }
    }
  });
}

tabs.forEach((b, i) => b.addEventListener("click", () => show(i)));
new MutationObserver(applyTheme).observe(document.documentElement,
  { attributes: true, attributeFilter: ["data-theme"] });
show(0);
</script>
"""


def build_hub(pages):
    """[(ラベル, ページHTML), ...] からタブ切り替えの統合HTMLを作る。"""
    tabs, frames, tpls = [], [], []
    for i, (label, html) in enumerate(pages):
        assert "</template>" not in html, f"{label} にtemplate終了タグが含まれる"
        tabs.append(f'<button role="tab" aria-selected="false">{label}</button>')
        frames.append(f'<iframe title="{label}"></iframe>')
        tpls.append(f'<template id="pg{i}">\n{html}\n</template>')
    return (TEMPLATE
            .replace("__TABS__", "\n  ".join(tabs))
            .replace("__FRAMES__", "".join(frames))
            .replace("__TEMPLATES__", "\n".join(tpls)))


def main():
    pages = [(label, (OUT / fname).read_text()) for fname, label in PAGES]
    out = OUT / "ramis_hub.html"
    out.write_text(build_hub(pages))
    print(f"OK: {len(pages)}ページ統合 -> {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
