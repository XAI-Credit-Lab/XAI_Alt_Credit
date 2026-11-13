"""
src/dashboard/fairness_tab.py
Streamlit 대시보드 - 공정성 탭

데이터 출처 (우선순위)
  1) FastAPI  GET {API_URL}/fairness/mitigation   -> 성별 DI, ΔTPR, ΔFPR (완화 전/후)   ※ 헤드라인
  2) reports/fairness_report.json                 -> 연령 DI, 속성별 판정, 집단표, 기준값  ※ 상세
     (python -m src.fairness.fairness 로 생성. API가 꺼져 있으면 헤드라인도 여기서 읽음)
  - 화면에 보이는 모든 숫자는 위 두 곳에서만 오며, 이 파일에는 값이 없다.
  - 판정 기준(DI 0.8~1.25, EO 0.10)은 src/fairness/fairness.py 의 상수를 그대로 가져온다.

사용
  - 단독 실행 : streamlit run src/dashboard/fairness_tab.py
  - 기존 앱에  : from src.dashboard.fairness_tab import render_fairness_tab; with tab: render_fairness_tab()
  - Docker    : 환경변수 API_URL=http://api:8000  (compose 서비스 이름, localhost 아님)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 판정 기준은 한 곳(fairness.py)에서만 관리
    from src.fairness.fairness import DI_LOWER, DI_UPPER, EO_THRESHOLD, MIN_DEFAULTS_PER_GROUP
except Exception:  # fairness.py 없이 대시보드만 배포한 경우의 대비값 (fairness.py와 같아야 함)
    DI_LOWER, DI_UPPER, EO_THRESHOLD, MIN_DEFAULTS_PER_GROUP = 0.8, 1.25, 0.10, 100

def _stretch_kwargs() -> dict:
    """Streamlit 1.50+ 는 width="stretch", 그보다 오래된 버전은 use_container_width=True"""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


STRETCH = _stretch_kwargs()
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REPORT_PATH = Path(os.environ.get("FAIRNESS_REPORT", ROOT / "reports" / "fairness_report.json"))

# 검증된 범주 색 (파랑=완화 전, 주황=완화 후). 상태색(빨강/초록)은 판정 전용이라 쓰지 않음
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
GRID = "rgba(128,128,128,0.25)"


# ----------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def load_api(url: str):
    try:
        r = requests.get(f"{url}/fairness/mitigation", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def load_report(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _status(v):
    if v is None:
        return "UNKNOWN"
    return "WARNING" if (v < DI_LOWER or v > DI_UPPER) else "OK"


def _badge(label: str, v, kind: str = "di") -> str:
    """색만으로 구분하지 않도록 아이콘 + 글자를 함께 표기"""
    if v is None:
        return f"❔ {label}: 계산 불가"
    bad = _status(v) == "WARNING" if kind == "di" else v > EO_THRESHOLD
    # '~' 두 개가 마크다운 취소선이 되므로 이스케이프
    return f"{'⚠️ 경고' if bad else '✅ 정상'} · {label.replace('~', chr(92) + '~')} {v:.3f}"


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
                    hovertemplate="%{x}<br>" + name + ": %{y:.4f}<extra></extra>")
    return fig


def di_chart(before: dict, after: dict):
    labels = ["성별 (여성/남성)"] + [f"연령 {k}/35~54" for k in before["age_di"]]
    b = [before["di_ratio"]] + list(before["age_di"].values())
    a = [after["di_ratio"]] + list(after["age_di"].values())
    fig = _bars(go.Figure(), labels, b, a, inside=True)
    for y, txt in ((DI_LOWER, f"하한 {DI_LOWER}"), (DI_UPPER, f"상한 {DI_UPPER}")):
        fig.add_hline(y=y, line=dict(dash="dash", width=1.5, color="gray"), annotation_text=txt,
                      annotation_position="top left", annotation_font_size=11)
    top = max([v for v in b + a if v is not None] + [DI_UPPER]) * 1.15
    _layout(fig, "Disparate Impact Ratio (1.0에 가까울수록 균등)", "DI")
    fig.update_yaxes(range=[0, top])
    return fig


def eo_chart(before: dict, after: dict, with_age: bool):
    labels, b, a = ["성별 ΔTPR", "성별 ΔFPR"], [before["equalized_odds_tpr_diff"], before["equalized_odds_fpr_diff"]], \
        [after["equalized_odds_tpr_diff"], after["equalized_odds_fpr_diff"]]
    if with_age:
        labels += ["연령 ΔTPR", "연령 ΔFPR"]
        b += [before["age_equalized_odds_tpr_diff"], before["age_equalized_odds_fpr_diff"]]
        a += [after["age_equalized_odds_tpr_diff"], after["age_equalized_odds_fpr_diff"]]
    fig = _bars(go.Figure(), labels, b, a)
    fig.add_hline(y=EO_THRESHOLD, line=dict(dash="dash", width=1.5, color="gray"),
                  annotation_text=f"기준 {EO_THRESHOLD}", annotation_position="top left", annotation_font_size=11)
    _layout(fig, "Equalized Odds 차이 (0에 가까울수록 균등)", "집단 간 차이")
    fig.update_yaxes(range=[0, max([v for v in b + a if v is not None] + [EO_THRESHOLD]) * 1.2])
    return fig


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_fairness_tab() -> None:
    st.subheader("공정성 검증")
    api, rep = load_api(API_URL), load_report(str(REPORT_PATH))

    if api is None and rep is None:
        st.error("공정성 결과를 불러올 수 없습니다.")
        st.markdown(
            f"- API({API_URL})가 꺼져 있고 `{REPORT_PATH.name}` 도 없습니다.\n"
            "- 프로젝트 최상위 폴더에서 `python -m src.fairness.fairness` 를 실행하면 리포트가 만들어집니다.\n"
            "- 이 명령은 `models/` 와 `data/processed/` (X·y·sensitive 의 val/test) 가 있어야 합니다."
        )
        return

    # 헤드라인: API 우선, 없으면 리포트
    before = (api or {}).get("before") or rep["non_miti"]
    after = (api or {}).get("after") or rep["miti"]
    st.caption(f"헤드라인 출처: {'API ' + API_URL if api else '리포트 파일 (API 연결 안 됨)'}"
               f"  |  상세(연령·집단표): {'리포트 파일' if rep else '없음 — python -m src.fairness.fairness 실행'}")

    # 상세는 리포트가 있을 때만. 헤드라인과 리포트가 다른 모델 결과면 알려 줌
    if rep and api and abs(rep["non_miti"]["di_ratio"] - api["before"]["di_ratio"]) > 1e-6:
        st.warning("리포트 파일의 값이 API 결과와 다릅니다. 모델을 다시 학습했다면 "
                   "`python -m src.fairness.fairness` 로 리포트를 새로 만드세요. (상세 부분은 구버전일 수 있음)")
        rep = None
    if rep:
        before, after = {**rep["non_miti"], **before}, {**rep["miti"], **after}

    # --- 판정 (속성마다 따로) -----------------------------------------
    st.markdown("##### 속성별 판정")
    c1, c2 = st.columns(2)
    for col, title, d in ((c1, "완화 전", before), (c2, "완화 후", after)):
        with col:
            st.markdown(f"**{title}**")
            st.markdown(f"- {_badge('성별 DI', d['di_ratio'])}")
            if rep:
                for k, v in d["age_di"].items():
                    st.markdown(f"- {_badge(f'연령 {k}/35~54 DI', v)}")
    if rep and (before["overall_status"] == "WARNING"):
        st.warning("완화 전 모델은 위반 속성이 있습니다. 성별이 정상이어도 연령이 위반이면 경고로 표시합니다.")

    # --- 핵심 수치 -----------------------------------------------------
    k1, k2, k3 = st.columns(3)
    for col, label, key in ((k1, "성별 DI", "di_ratio"), (k2, "성별 ΔTPR", "equalized_odds_tpr_diff"),
                            (k3, "성별 ΔFPR", "equalized_odds_fpr_diff")):
        col.metric(label, f"{after[key]:.3f}", delta=f"{after[key] - before[key]:+.3f} (완화 전 {before[key]:.3f})",
                   delta_color="off")

    # --- 차트 ---------------------------------------------------------
    if rep:
        st.plotly_chart(di_chart(before, after), **STRETCH)
    st.plotly_chart(eo_chart(before, after, with_age=bool(rep)), **STRETCH)

    if not rep:
        st.info("연령대 DI, 집단표, 완화 비용은 리포트 파일이 있어야 표시됩니다.")
        return

    # --- 완화 비용 ----------------------------------------------------
    st.markdown("##### 완화의 비용")
    info = rep["info"]
    m1, m2, m3 = st.columns(3)
    m1.metric("전체 승인율", f"{after['approval_rate']:.2%}", delta=f"{after['approval_rate'] - before['approval_rate']:+.2%}p",
              delta_color="off")
    m2.metric("승인자 부도율", f"{after['approved_default_rate']:.2%}",
              delta=f"{after['approved_default_rate'] - before['approved_default_rate']:+.2%}p", delta_color="off",
              help="승인된 사람 중 실제로 부도 난 비율. 낮을수록 좋음")
    m3.metric("Test AUC", f"{info['test_auc']:.4f}", help="기준값만 바꾸고 점수는 그대로라 완화 전/후가 같음")

    # --- 집단표 -------------------------------------------------------
    st.markdown("##### 집단별 표")
    which = st.radio("대상", ["완화 전", "완화 후"], horizontal=True, key="fair_tbl")
    d = before if which == "완화 전" else after
    ren = {"group": "집단", "n": "인원", "n_default": "부도 건수", "approval_rate": "승인율",
           "tpr": "TPR (정상 중 승인)", "fpr": "FPR (부도 중 승인)"}
    for title, key in (("성별", "gender_table"), ("연령대", "age_table")):
        t = pd.DataFrame(d[key]).rename(columns=ren)
        st.markdown(f"**{title}**")
        st.dataframe(t, hide_index=True, **STRETCH,
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                    ("승인율", "TPR (정상 중 승인)", "FPR (부도 중 승인)")})
        small = t.loc[t["부도 건수"] < MIN_DEFAULTS_PER_GROUP, "집단"].tolist()
        if small:
            st.caption(f"⚠️ 부도 {MIN_DEFAULTS_PER_GROUP}건 미만 집단: {small} — 차이 값이 불안정할 수 있음")

    # --- 설명 ---------------------------------------------------------
    with st.expander("완화 방법과 집단별 승인 기준값"):
        st.markdown(
            f"**{info['mitigation']}** — {' × '.join(info['mitigation_by'])} 집단마다 Validation 부도확률의 "
            f"하위 {info['target_approval_rate']:.0%} 지점을 승인 기준값으로 사용합니다. "
            f"(완화 전: 모든 집단에 {info['approval_cutoff']:.4f} 하나)")
        st.dataframe(pd.DataFrame({"집단": list(info["group_cutoffs"]), "승인 기준값": list(info["group_cutoffs"].values())}),
                     hide_index=True, **STRETCH)
    with st.expander("정의와 한계"):
        st.markdown(
            f"- **승인 규칙**: {info['approval_rule']}\n"
            "- **TPR** = P(승인 | 실제 정상), **FPR** = P(승인 | 실제 부도). 성별 gender=1 이 여성.\n"
            f"- **모델**: {info['model']} v{info['model_version']} (학습 {info['trained_at']}), Test {info['n_test']:,}행\n"
            "- 성별·편향은 시뮬레이터가 가정한 값으로 만든 합성 데이터이며 실제 대출 데이터의 결과가 아닙니다.\n"
            "- 집단별 기준값은 보호 속성을 판정에 쓰므로, 실제 서비스 적용 여부는 법적·정책적 검토가 필요합니다.")


if __name__ == "__main__":
    st.set_page_config(page_title="공정성 검증", layout="wide")
    render_fairness_tab()
