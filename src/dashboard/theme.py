"""
src/dashboard/theme.py
대시보드 공통 화면 요소 (참고 레이아웃: 크림색 패널 + 골드 포인트 + 반원형 게이지 + 요인 카드 + 도넛)
데이터는 가지지 않고, 받은 값을 그리기만 한다.
"""
from __future__ import annotations

import html
import math

import streamlit as st

GOLD, GOLD_DARK, INK, CREAM, LINE = "#e0a840", "#c4871a", "#1f1f1f", "#fcf9f3", "#efe6d2"
GRAY_D, GRAY_M, GRAY_L = "#2b2b2b", "#8a8f98", "#c9ccd1"

CSS = f"""
<style>
.stApp {{ background:#ffffff; color:{INK}; }}
[data-testid="stSidebar"] {{ background:{CREAM}; border-right:1px solid {LINE}; }}
.block-container {{ padding-top:2.2rem; max-width:1280px; }}
/* 패널: st.container(border=True, key="panel_*") */
[class*="st-key-panel"] {{ background:{CREAM} !important; border:1px solid {LINE} !important; border-radius:6px !important; }}
.rp-head {{ display:flex; align-items:center; gap:14px; margin:0 0 4px 0; }}
.rp-stripe {{ width:46px; height:30px; background:repeating-linear-gradient(135deg,{GOLD} 0 3px,#fff 3px 7px); flex:none; }}
.rp-title {{ font-size:2.1rem; font-weight:600; letter-spacing:-0.5px; color:{INK}; line-height:1.15; }}
.rp-sub {{ font-size:.88rem; font-style:italic; color:#444; margin:2px 0 14px 60px; }}
.rp-bar {{ background:{GOLD}; color:#fff; font-weight:700; padding:9px 16px; border-radius:2px; display:inline-block; min-width:230px; }}
.rp-h {{ font-size:1.15rem; font-weight:700; color:{INK}; display:flex; align-items:center; gap:10px; margin:2px 0 2px 0; }}
.rp-info {{ background:{GRAY_D}; color:#fff; width:22px; height:22px; border-radius:50%; font-size:.8rem; display:inline-flex; align-items:center; justify-content:center; font-weight:700; }}
.rp-line {{ height:1px; background:linear-gradient(90deg,{GOLD},rgba(224,168,64,0)); margin:4px 0 12px 0; }}
.rp-card {{ background:#fff; border-radius:6px; padding:12px 16px; margin:0 0 14px 0; box-shadow:0 2px 6px rgba(0,0,0,.12); }}
.rp-card b {{ display:block; font-size:1.25rem; color:{INK}; }}
.rp-card span {{ font-size:.86rem; color:#444; }}
.rp-card small {{ display:block; font-size:.78rem; color:#7a7a7a; margin-top:2px; }}
.rp-pill {{ display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:.95rem; }}
.rp-ok {{ background:#e7f4ea; color:#1d6b34; }} .rp-no {{ background:#fdeaea; color:#a12626; }}
.rp-unit {{ font-size:.8rem; color:#555; }}
.rp-legend td {{ padding:2px 6px; font-size:.95rem; }}
.rp-opt {{ background:#fff; border-radius:6px; padding:14px 16px 12px 16px; margin:0 0 14px 0; box-shadow:0 2px 6px rgba(0,0,0,.12); border-top:3px solid {GOLD}; }}
.rp-opt-h {{ font-weight:700; font-size:1.02rem; color:{INK}; margin:0 0 8px 0; }}
.rp-opt-h span {{ font-weight:400; font-size:.82rem; color:#666; margin-left:6px; }}
.rp-chg {{ padding:6px 0; border-top:1px dashed #e6dfcf; }}
.rp-chg:first-of-type {{ border-top:0; }}
.rp-chg-name {{ font-size:.84rem; color:#555; }}
.rp-chg-val {{ font-size:1.12rem; font-weight:700; color:{INK}; }}
.rp-chg-val i {{ font-style:normal; color:{GOLD_DARK}; padding:0 6px; }}
.rp-chg small {{ display:block; font-size:.76rem; color:#7a7a7a; }}
.rp-opt-res {{ margin-top:8px; padding-top:8px; border-top:1px solid #efe6d2; font-size:.88rem; color:#333; }}
.rp-opt-res b {{ color:#1d6b34; }}
.rp-foot {{ text-align:center; font-size:.88rem; color:#333; border-top:1px solid #555; margin-top:18px; padding-top:8px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="rp-head"><div class="rp-stripe"></div><div class="rp-title">{html.escape(title)}</div></div>'
                f'<div class="rp-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def panel(key: str):
    """크림색 패널 컨테이너 (구버전 Streamlit은 key 인자가 없어 기본 테두리 컨테이너로 대체)"""
    try:
        return st.container(border=True, key=f"panel_{key}")
    except TypeError:
        return st.container(border=True)


def section(title: str, info: bool = True) -> None:
    st.markdown(f'<div class="rp-h">{html.escape(title)}{"<span class=rp-info>i</span>" if info else ""}</div>'
                '<div class="rp-line"></div>', unsafe_allow_html=True)


def date_bar(text: str) -> None:
    st.markdown(f'<div class="rp-bar">{html.escape(text)} &nbsp;▾</div>', unsafe_allow_html=True)


def factor_card(value: str, label: str, sub: str = "") -> None:
    """값(굵게) + 이름 + 보조 줄(예: 원화 표기)"""
    st.markdown(f'<div class="rp-card"><b>{html.escape(value)}</b><span>{html.escape(label)}</span>'
                + (f'<small>{html.escape(sub)}</small>' if sub else "") + '</div>', unsafe_allow_html=True)


def option_card(title: str, tag: str, changes: list[tuple[str, str, str, str]], result: str) -> None:
    """반사실적 설명 방안 카드. changes = [(항목 이름, 지금 값, 제안 값, 보조 줄(원화 등))]"""
    rows = "".join(
        f'<div class="rp-chg"><div class="rp-chg-name">{html.escape(n)}</div>'
        f'<div class="rp-chg-val">{html.escape(a)}<i>→</i>{html.escape(b)}</div>'
        + (f'<small>{html.escape(sub)}</small>' if sub else "") + '</div>'
        for n, a, b, sub in changes)
    st.markdown(f'<div class="rp-opt"><div class="rp-opt-h">{html.escape(title)}<span>{html.escape(tag)}</span></div>'
                f'{rows}<div class="rp-opt-res">{result}</div></div>', unsafe_allow_html=True)


def pill(text: str, ok: bool) -> None:
    st.markdown(f'<span class="rp-pill {"rp-ok" if ok else "rp-no"}">{html.escape(text)}</span>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="rp-foot">{html.escape(text)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 게이지 (270도 링 + 가운데 점수 + 포인터)
# ----------------------------------------------------------------------
def _pt(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def gauge_svg(score: float | None, vmax: int = 1000, grade: str = "", grade_text: str = "",
              ticks=(0, 250, 500, 750, 1000), unit: str = "점") -> str:
    """score 가 None 이면 '-' 로 비어 있는 게이지. 각도는 SVG 기준(시계방향), 135도에서 시작해 270도 회전"""
    cx = cy = 200
    r, w = 150, 30
    sx, sy = _pt(cx, cy, r, 135)
    ex, ey = _pt(cx, cy, r, 45)
    arc = f"M {sx:.1f} {sy:.1f} A {r} {r} 0 1 1 {ex:.1f} {ey:.1f}"
    f = 0 if score is None else max(0.0, min(1.0, score / vmax))
    ang = 135 + 270 * f
    tick_svg = ""
    for t in ticks:
        tx, ty = _pt(cx, cy, r + 30, 135 + 270 * t / vmax)
        tick_svg += f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="15" fill="#444" text-anchor="middle" dominant-baseline="middle">{t}</text>'
    pointer = ("" if score is None else
               f'<polygon points="{cx + 92},{cy - 10} {cx + 92},{cy + 10} {cx + 112},{cy}" fill="{GRAY_D}" '
               f'transform="rotate({ang:.1f} {cx} {cy})"/>')
    shown = "-" if score is None else f"{int(round(score))}"
    svg = f"""
<svg viewBox="0 0 400 400" width="100%" style="max-width:380px;display:block;margin:0 auto" role="img"
     aria-label="신용 점수 {shown}{unit}">
  <defs>
    <linearGradient id="gg" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="#3a3a3a"/><stop offset="0.45" stop-color="#d4d6da"/><stop offset="1" stop-color="{GOLD}"/>
    </linearGradient>
    <radialGradient id="gi" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#ececec"/></radialGradient>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity=".18"/></filter>
  </defs>
  <path d="{arc}" fill="none" stroke="url(#gg)" stroke-width="{w}" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="104" fill="url(#gi)" filter="url(#sh)"/>
  <circle cx="{cx}" cy="{cy}" r="72" fill="#fafafa" stroke="#e4e4e4"/>
  {pointer}
  {tick_svg}
  <text x="{cx}" y="{cy - 4}" font-size="54" font-weight="700" fill="{GOLD_DARK}" text-anchor="middle">{shown}</text>
  <text x="{cx + 52}" y="{cy - 12}" font-size="16" fill="#555" text-anchor="start">{unit if score is not None else ""}</text>
  <text x="{cx}" y="{cy + 30}" font-size="17" font-weight="700" fill="{GOLD_DARK}" text-anchor="middle">{html.escape(grade_text)}</text>
</svg>"""
    # 빈 줄이 있으면 마크다운이 HTML 블록을 끊어 SVG 가 글자로 깨져 보이므로 빈 줄을 제거
    return "\n".join(line for line in svg.splitlines() if line.strip())
