"""
src/dashboard/fairness_tab.py
Streamlit 대시보드 - 공정성 탭

MSA 구조: 이 파일은 모델 파일(.pkl)이나 src/fairness 모듈, 리포트 파일을 읽지 않는다.
모든 값은 FastAPI 서버와 HTTP 통신으로만 가져온다.
  GET {API_URL}/fairness/mitigation : 성별 DI, ΔTPR, ΔFPR (완화 전/후)          ← 요구사항 엔드포인트
  GET {API_URL}/fairness/detail     : 연령 DI, 속성별 판정, 집단표, 기준값        ← 상세 (추가 엔드포인트)
화면의 모든 숫자는 위 응답에서만 오며, 이 파일에는 데이터 값이 없다.

사용
  - 단독 실행 : streamlit run src/dashboard/fairness_tab.py
  - 기존 앱에  : from src.dashboard.fairness_tab import render_fairness_tab; with tab: render_fairness_tab()
  - Docker    : 환경변수 API_URL=http://api:8000  (compose 서비스 이름, localhost 아님)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.dashboard import theme  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# 판정 기준: 프로젝트 요구사항 값. 서버가 /fairness/detail 로 기준값을 보내 주면 그 값을 우선 사용
DEFAULT_THRESHOLDS = {"di_lower": 0.8, "di_upper": 1.25, "eo_threshold": 0.10, "min_defaults_per_group": 100}
AGE_REF = "35~54"

# 검증된 범주 색 (파랑=완화 전, 주황=완화 후). 상태색(빨강/초록)은 판정 전용이라 쓰지 않음
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
GRID = "rgba(128,128,128,0.25)"


def _stretch_kwargs() -> dict:
    """Streamlit 1.50+ 는 width="stretch", 그보다 오래된 버전은 use_container_width=True"""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


STRETCH = _stretch_kwargs()


# ----------------------------------------------------------------------
# 데이터 로드 (HTTP only)
# ----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch(url: str, path: str):
    """API GET. 실패하면 None (예외를 화면에 던지지 않음)"""
    try:
        r = requests.get(f"{url}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _status(v, th):
    if v is None:
        return "UNKNOWN"
    return "WARNING" if (v < th["di_lower"] or v > th["di_upper"]) else "OK"


def _badge(label: str, v, th, kind: str = "di") -> str:
    """색만으로 구분하지 않도록 아이콘 + 글자를 함께 표기. '~'는 마크다운 취소선이라 이스케이프"""
    if v is None:
        return f"❔ {label}: 계산 불가"
    bad = _status(v, th) == "WARNING" if kind == "di" else v > th["eo_threshold"]
    unit = "배" if kind == "di" else ""
    return f"{'⚠️ 경고' if bad else '✅ 정상'} · {label.replace('~', chr(92) + '~')} {v:.3f}{unit}"


# ----------------------------------------------------------------------
# 차트
# ----------------------------------------------------------------------
def _layout(fig, title: str, ytitle: str, height: int = 360):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        barmode="group", bargap=0.35, bargroupgap=0.08, height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=ytitle, gridcolor=GRID, zeroline=False),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _bars(fig, x, before, after, inside: bool = False, fmt: str = "%{y:.3f}"):
    for name, ys, col in (("완화 전", before, C_BEFORE), ("완화 후", after, C_AFTER)):
        fig.add_bar(name=name, x=x, y=ys, marker_color=col, marker_line=dict(color="rgba(0,0,0,0)", width=0),
                    text=ys, texttemplate=fmt, cliponaxis=False,
                    **(dict(textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=13))
                       if inside else dict(textposition="outside")),
                    hovertemplate="%{x}<br>" + name + ": " + fmt + "<extra></extra>")
    return fig


def di_chart(before: dict, after: dict, th: dict):
    labels = ["성별 (여성/남성)"] + [f"연령 {k}/{AGE_REF}" for k in before["age_di"]]
    b = [before["di_ratio"]] + list(before["age_di"].values())
    a = [after["di_ratio"]] + list(after["age_di"].values())
    fig = _bars(go.Figure(), labels, b, a, inside=True, fmt="%{y:.3f}배")
    for y, txt in ((th["di_lower"], f"하한 {th['di_lower']}배"), (th["di_upper"], f"상한 {th['di_upper']}배")):
        fig.add_hline(y=y, line=dict(dash="dash", width=1.5, color="gray"), annotation_text=txt,
                      annotation_position="top left", annotation_font_size=11)
    top = max([v for v in b + a if v is not None] + [th["di_upper"]]) * 1.15
    _layout(fig, "Disparate Impact Ratio (1.0배에 가까울수록 균등)", "DI (배)")
    fig.update_yaxes(range=[0, top])
    return fig


def eo_chart(before: dict, after: dict, with_age: bool, th: dict):
    labels, b, a = ["성별 ΔTPR", "성별 ΔFPR"], [before["equalized_odds_tpr_diff"], before["equalized_odds_fpr_diff"]], \
        [after["equalized_odds_tpr_diff"], after["equalized_odds_fpr_diff"]]
    if with_age:
        labels += ["연령 ΔTPR", "연령 ΔFPR"]
        b += [before["age_equalized_odds_tpr_diff"], before["age_equalized_odds_fpr_diff"]]
        a += [after["age_equalized_odds_tpr_diff"], after["age_equalized_odds_fpr_diff"]]
    b, a = [None if v is None else v * 100 for v in b], [None if v is None else v * 100 for v in a]   # 비율 차이 -> %p
    thr = th["eo_threshold"] * 100
    fig = _bars(go.Figure(), labels, b, a, fmt="%{y:.1f}%p")
    fig.add_hline(y=thr, line=dict(dash="dash", width=1.5, color="gray"),
                  annotation_text=f"기준 {thr:.0f}%p", annotation_position="top left", annotation_font_size=11)
    _layout(fig, "Equalized Odds 차이 (0%p에 가까울수록 균등)", "집단 간 차이 (%p)")
    fig.update_yaxes(range=[0, max([v for v in b + a if v is not None] + [thr]) * 1.2])
    return fig


def approval_chart(before: dict, after: dict, target: float | None):
    """집단별 승인율 막대 차트 (요구사항: 공정성 탭 필수 구성요소)"""
    def rates(d):
        out = {}
        for r in d["gender_table"]:
            out[f"성별 {r['group']}"] = r["approval_rate"]
        for r in d["age_table"]:
            out[f"연령 {r['group']}"] = r["approval_rate"]
        return out
    rb, ra = rates(before), rates(after)
    labels = list(rb)
    fig = _bars(go.Figure(), labels, [rb[k] for k in labels], [ra.get(k) for k in labels], fmt="%{y:.1%}")
    if target is not None:
        fig.add_hline(y=target, line=dict(dash="dash", width=1.5, color="gray"),
                      annotation_text=f"전체 목표 승인율 {target:.0%}", annotation_position="top left",
                      annotation_font_size=11)
    _layout(fig, "집단별 승인율 (집단 간 막대 높이가 같을수록 균등)", "승인율 (%)")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return fig


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_fairness_tab() -> None:
    with theme.panel("fair"):
        _render_body()


def _render_body() -> None:
    theme.section("공정성 검증")
    api = fetch(API_URL, "/fairness/mitigation")
    det = fetch(API_URL, "/fairness/detail")

    if api is None and det is None:
        st.error(f"공정성 결과를 불러올 수 없습니다. API 서버({API_URL})에 연결되지 않았습니다.")
        st.markdown("- API 서버가 실행 중인지, `API_URL` 환경변수가 맞는지 확인하세요. (Docker: `http://api:8000`)\n"
                    "- 서버 쪽에서 모델(`models/`)과 분할 데이터가 준비돼 있어야 `/fairness` 가 응답합니다.")
        return

    th = {**DEFAULT_THRESHOLDS, **((det or {}).get("thresholds") or {})}
    # 헤드라인은 요구사항 엔드포인트 우선, 없으면 detail 에서
    before = (api or {}).get("before") or det["non_miti"]
    after = (api or {}).get("after") or det["miti"]
    st.caption(f"출처: API {API_URL}  |  상세(연령·집단표): "
               f"{'/fairness/detail' if det else '없음 (/fairness/detail 미응답)'}")

    # /fairness/mitigation 과 /fairness/detail 이 서로 다른 모델 결과면 상세를 쓰지 않음
    if det and api and abs(det["non_miti"]["di_ratio"] - api["before"]["di_ratio"]) > 1e-6:
        st.warning("두 API 응답의 값이 서로 다릅니다 (서버 재시작 중이거나 모델 교체 직후일 수 있음). 상세는 표시하지 않습니다.")
        det = None
    if det:
        before, after = {**det["non_miti"], **before}, {**det["miti"], **after}

    # --- 판정 (속성마다 따로) -----------------------------------------
    st.markdown("##### 속성별 판정")
    c1, c2 = st.columns(2)
    for col, title, d in ((c1, "완화 전", before), (c2, "완화 후", after)):
        with col:
            st.markdown(f"**{title}**")
            st.markdown(f"- {_badge('성별 DI', d['di_ratio'], th)}")
            if det:
                for k, v in d["age_di"].items():
                    st.markdown(f"- {_badge(f'연령 {k}/{AGE_REF} DI', v, th)}")
    if det and before["overall_status"] == "WARNING":
        st.warning("완화 전 모델은 위반 속성이 있습니다. 성별이 정상이어도 연령이 위반이면 경고로 표시합니다.")

    # --- 핵심 수치 -----------------------------------------------------
    k1, k2, k3 = st.columns(3)
    col, label, key = k1, "성별 DI (배)", "di_ratio"
    col.metric(label, f"{after[key]:.3f}배", delta=f"{after[key] - before[key]:+.3f}배 (완화 전 {before[key]:.3f}배)",
               delta_color="off")
    for col, label, key in ((k2, "성별 ΔTPR (%p)", "equalized_odds_tpr_diff"), (k3, "성별 ΔFPR (%p)", "equalized_odds_fpr_diff")):
        col.metric(label, f"{after[key] * 100:.1f}%p",
                   delta=f"{(after[key] - before[key]) * 100:+.1f}%p (완화 전 {before[key] * 100:.1f}%p)", delta_color="off")

    # --- 차트 ---------------------------------------------------------
    if det:
        st.plotly_chart(approval_chart(before, after, det["info"].get("target_approval_rate")), **STRETCH)
        st.plotly_chart(di_chart(before, after, th), **STRETCH)
    st.plotly_chart(eo_chart(before, after, with_age=bool(det), th=th), **STRETCH)

    if not det:
        st.info("집단별 승인율, 연령대 DI, 집단표, 완화 비용은 /fairness/detail 응답이 있어야 표시됩니다.")
        return

    # --- 완화 비용 ----------------------------------------------------
    st.markdown("##### 완화의 비용")
    info = det["info"]
    m1, m2, m3 = st.columns(3)
    m1.metric("전체 승인율 (%)", f"{after['approval_rate']:.2%}",
              delta=f"{(after['approval_rate'] - before['approval_rate']) * 100:+.2f}%p", delta_color="off")
    m2.metric("승인자 부도율 (%)", f"{after['approved_default_rate']:.2%}",
              delta=f"{(after['approved_default_rate'] - before['approved_default_rate']) * 100:+.2f}%p", delta_color="off",
              help="승인된 사람 중 실제로 부도 난 비율. 낮을수록 좋음")
    m3.metric("Test AUC (0~1)", f"{info['test_auc']:.4f}", help="기준값만 바꾸고 점수는 그대로라 완화 전/후가 같음")

    # --- 집단표 -------------------------------------------------------
    st.markdown("##### 집단별 표")
    which = st.radio("대상", ["완화 전", "완화 후"], horizontal=True, key="fair_tbl")
    d = before if which == "완화 전" else after
    pct_cols = ("승인율 (%)", "TPR · 정상 중 승인 (%)", "FPR · 부도 중 승인 (%)")
    ren = {"group": "집단", "n": "인원 (명)", "n_default": "부도 건수 (건)", "approval_rate": pct_cols[0],
           "tpr": pct_cols[1], "fpr": pct_cols[2]}
    for title, key in (("성별", "gender_table"), ("연령대", "age_table")):
        t = pd.DataFrame(d[key]).rename(columns=ren)
        for c in pct_cols:                                   # 비율(0~1) -> 퍼센트
            t[c] = t[c] * 100
        st.markdown(f"**{title}**")
        st.dataframe(t, hide_index=True, **STRETCH,
                     column_config={c: st.column_config.NumberColumn(format="%.2f%%") for c in pct_cols})
        small = t.loc[t["부도 건수 (건)"] < th["min_defaults_per_group"], "집단"].tolist()
        if small:
            st.caption(f"⚠️ 부도 {th['min_defaults_per_group']}건 미만 집단: {small} — 차이 값이 불안정할 수 있음")

    # --- 설명 ---------------------------------------------------------
    with st.expander("완화 방법과 집단별 승인 기준값"):
        st.markdown(
            f"**{info['mitigation']}** — {' × '.join(info['mitigation_by'])} 집단마다 Validation 부도확률의 "
            f"하위 {info['target_approval_rate']:.0%} 지점을 승인 기준값으로 사용합니다. "
            f"(완화 전: 모든 집단에 부도확률 {info['approval_cutoff'] * 100:.2f}% 하나)")
        st.dataframe(pd.DataFrame({"집단": list(info["group_cutoffs"]),
                                   "승인 기준값 · 부도확률 (%)": [v * 100 for v in info["group_cutoffs"].values()]}),
                     hide_index=True, **STRETCH,
                     column_config={"승인 기준값 · 부도확률 (%)": st.column_config.NumberColumn(format="%.2f%%")})
    with st.expander("정의와 한계"):
        st.markdown(
            f"- **승인 규칙**: {info['approval_rule']}\n"
            "- **TPR** = P(승인 | 실제 정상), **FPR** = P(승인 | 실제 부도). 성별 gender=1 이 여성.\n"
            f"- **모델**: {info['model']} v{info['model_version']} (학습 {info['trained_at']}), Test {info['n_test']:,}명\n"
            "- 성별·편향은 시뮬레이터가 가정한 값으로 만든 합성 데이터이며 실제 대출 데이터의 결과가 아닙니다.\n"
            "- 집단별 기준값은 보호 속성을 판정에 쓰므로, 실제 서비스 적용 여부는 법적·정책적 검토가 필요합니다.")


if __name__ == "__main__":
    st.set_page_config(page_title="공정성 검증", layout="wide")
    render_fairness_tab()
