"""
src/dashboard/app.py
XAI 기반 대안신용평가 시스템 - Streamlit 대시보드 (5개 탭)

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

MSA 원칙: 모델·전처리기(.pkl)나 reports/ 파일을 직접 읽지 않고, 모든 데이터는 FastAPI에 HTTP로만 요청한다.
  탭1 고객 심사  : POST /predict
  탭2 XAI 분석   : POST /explain/plots (이미지 포함, 없으면 /explain)
  탭3 공정성     : GET  /fairness/mitigation, /fairness/detail   (fairness_tab.py)
  탭4 모델 성능  : GET  /metrics, /health
  탭5 통계 검증  : GET  /anova
(입력 폼의 이름·단위·허용 범위는 src/data/feature_spec.py 한 곳에서 가져온다. 이 파일은 순수 사전이라 모델·파일에 접근하지 않는다.)

표기 규칙
  - 모든 수치에 단위를 붙인다. 금액은 달러가 기본이고 바로 아래에 원화(고정 환율, 만원 미만 버림)를 함께 보인다. (units.py)

실행: 프로젝트 루트에서  API_URL=http://localhost:8000 streamlit run src/dashboard/app.py
      (Docker 에서는 API_URL=http://api:8000)
"""
from __future__ import annotations

import base64
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# `streamlit run src/dashboard/app.py` 로 실행해도 src 패키지를 찾도록 루트 추가
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import theme, units                                           # noqa: E402
from src.dashboard.fairness_tab import STRETCH, render_fairness_tab               # noqa: E402
from src.data.feature_spec import (FEATURE_SPEC, SHAP_SIGN_NOTE, format_value,     # noqa: E402
                                   input_label, label)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DISCLAIMER = "※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."
BLUE, ORANGE = "#2a78d6", "#eb6834"
KST = timezone(timedelta(hours=9))
GRADE_TEXT = {"A": "매우 우수", "B": "우수", "C": "보통", "D": "주의", "E": "위험"}

st.set_page_config(page_title="XAI 대안신용평가", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------
# HTTP 헬퍼
# ----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str):
    """GET. 실패하면 (None, 오류메시지)"""
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, _err_text(e)


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
        if r.status_code == 422:   # 입력값 검증 실패 → 어떤 칸이 문제인지 보여줌
            msgs = [f"{label(str(d['loc'][-1]))}: {d['msg']}" for d in r.json().get("detail", [])]
            return None, "입력값 오류 - " + " / ".join(msgs)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, _err_text(e)


def _err_text(e: Exception) -> str:
    if isinstance(e, requests.ConnectionError):
        return f"API 서버({API_URL})에 연결할 수 없습니다."
    if isinstance(e, requests.HTTPError) and e.response is not None:
        try:
            return f"API 오류 {e.response.status_code}: {e.response.json().get('detail', '')}"
        except Exception:
            return f"API 오류 {e.response.status_code}"
    return f"요청 실패: {e}"


def _png(b64):
    """API가 base64 문자열로 준 PNG를 화면용 바이트로 변환 (실패한 그림은 None)"""
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


# ----------------------------------------------------------------------
# 사이드바 입력 (탭1·탭2 공용). 이름·단위·범위·도움말은 feature_spec.py 기준
# ----------------------------------------------------------------------
def _num(name: str, default, key: str):
    s = FEATURE_SPEC[name]
    kw = dict(min_value=s["min"], max_value=s["max"], value=default, step=s["step"], help=s["help"], key=key)
    if isinstance(s["step"], float):
        kw["format"] = "%.2f" if s["step"] < 1 else "%.0f"
    return st.sidebar.number_input(input_label(name), **kw)


def _slider(name: str, default, key: str):
    s = FEATURE_SPEC[name]
    return st.sidebar.slider(input_label(name), float(s["min"]), float(s["max"]), default, float(s["step"]),
                             help=s["help"], key=key)


def sidebar_inputs() -> dict:
    """필수 입력은 항상 값을 보내고, 선택 입력(결측 가능 컬럼)은 체크하면 값을 보낸다."""
    st.sidebar.header("고객 정보 입력")
    st.sidebar.caption("필수 항목은 모두 입력, 선택 항목은 체크한 경우에만 전송합니다. 금액은 달러로 입력합니다.")
    p: dict = {}

    st.sidebar.subheader("기존 금융 변수 (필수)")
    p["RevolvingUtilizationOfUnsecuredLines"] = _num("RevolvingUtilizationOfUnsecuredLines", 0.30, "rev")
    p["DebtRatio"] = _num("DebtRatio", 0.30, "debt")
    p["NumberOfOpenCreditLinesAndLoans"] = _num("NumberOfOpenCreditLinesAndLoans", 6, "open")
    p["NumberRealEstateLoansOrLines"] = _num("NumberRealEstateLoansOrLines", 1, "re")
    p["card_history_months"] = _num("card_history_months", 60, "hist")

    st.sidebar.subheader("기존 금융 변수 (선택)")
    if st.sidebar.checkbox("월 소득 입력", True, key="use_income"):
        p["MonthlyIncome"] = _num("MonthlyIncome", 5000.0, "income")
        st.sidebar.caption(f"원화 : {units.krw(p['MonthlyIncome'])}  ({units.FX_NOTE})")
    else:
        st.sidebar.caption("월 소득을 입력하지 않으면 부채 비율은 계산에 반영되지 않습니다.")
    if st.sidebar.checkbox("부양 가족 수 입력", True, key="use_dep"):
        p["NumberOfDependents"] = _num("NumberOfDependents", 1, "dep")
    for name, field, key in (("NumberOfTime30-59DaysPastDueNotWorse", "past_due_30_59", "pd30"),
                             ("NumberOfTime60-89DaysPastDueNotWorse", "past_due_60_89", "pd60"),
                             ("NumberOfTimes90DaysLate", "past_due_90", "pd90")):
        if st.sidebar.checkbox(f"{label(name)} 입력", True, key=f"use_{key}"):
            p[field] = _num(name, 0, key)

    st.sidebar.subheader("대안 변수 (필수)")
    p["telecom_payment_rate"] = _slider("telecom_payment_rate", 0.95, "tel")
    p["utility_payment_rate"] = _slider("utility_payment_rate", 0.95, "util")
    p["spending_consistency"] = _slider("spending_consistency", 60.0, "spend")
    p["regular_payment_count"] = _num("regular_payment_count", 4, "reg")
    p["app_login_frequency"] = _num("app_login_frequency", 20, "app")
    return p


# ----------------------------------------------------------------------
# 탭 1. 고객 심사 (참고 레이아웃: 게이지 + 신용 요인 카드 + 계좌 구성 도넛)
# ----------------------------------------------------------------------
def _donut(n_total: int, n_re: int):
    others = max(n_total - n_re, 0)
    fig = go.Figure(go.Pie(
        labels=["부동산 대출", "그 외 대출·카드"], values=[n_re, others] if n_total else [1, 0],
        hole=0.62, sort=False, marker=dict(colors=["#2b2b2b", "#e0a840"], line=dict(color="#fcf9f3", width=3)),
        textinfo="none", hovertemplate="%{label}: %{value}개<extra></extra>", direction="clockwise"))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      annotations=[dict(text=f"<b>{n_total}개</b>", x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=26, color="#1f1f1f"))])
    return fig


_PRINT_HTML = ('<button onclick="try{window.parent.print()}catch(e){window.print()}" '
               'style="width:100%;height:38px;background:#1f1f1f;color:#fff;border:0;border-radius:3px;'
               'font-size:15px;cursor:pointer">🖨 인쇄</button>')


def _print_button():
    """인쇄 버튼. 신버전 Streamlit 은 st.iframe, 구버전은 components.html (이 함수는 신버전에서 제거 예정)"""
    if hasattr(st, "iframe"):
        st.iframe(_PRINT_HTML, height=44)
    else:
        import streamlit.components.v1 as components
        components.html(_PRINT_HTML, height=44)


def tab_predict(p: dict):
    res = st.session_state.get("predict")
    left, right = st.columns([2.1, 1], gap="medium")

    with left, theme.panel("score"):
        c1, c2, c3 = st.columns([3.2, 1.1, 1])
        with c1:
            theme.date_bar(f"심사 기준일 {datetime.now(KST):%Y-%m-%d}")
        with c2:
            run = st.button("심사 실행", type="primary", key="run_predict", **STRETCH)
        with c3:
            _print_button()
        if run:
            r, err = api_post("/predict", p)
            if err:
                st.error(err)
            else:
                st.session_state["predict"] = res = r

        g, f = st.columns([1.15, 1], gap="medium")
        with g:
            theme.section("신용 점수 (단위: 점, 0~1,000점)", info=False)
            if res:
                grade = res["risk_grade"]
                st.markdown(theme.gauge_svg(res["credit_score"], 1000, grade,
                                            f"{grade}등급 · {GRADE_TEXT.get(grade, '')}"), unsafe_allow_html=True)
                approved = res["approval_status"] == "APPROVED"
                theme.pill(("✅ 승인" if approved else "❌ 거절") + f" · 부도 확률 {units.pct(res['default_probability'])}",
                           approved)
            else:
                st.markdown(theme.gauge_svg(None, 1000), unsafe_allow_html=True)
                st.caption("[심사 실행]을 누르면 결과가 표시됩니다.")
        with f:
            theme.section("신용 요인 (입력값)")
            theme.factor_card(format_value("RevolvingUtilizationOfUnsecuredLines", p["RevolvingUtilizationOfUnsecuredLines"]),
                              "무담보 한도 사용률 (%)")
            avg = (p["telecom_payment_rate"] + p["utility_payment_rate"]) / 2
            theme.factor_card(units.pct(avg, 0), "정상 납부율 · 통신비·공과금 평균 (%)")
            theme.factor_card(units.years_months(p["card_history_months"]), "카드 이용 기간 (년·개월)")
            if "MonthlyIncome" in p:
                u_, k_ = units.money_lines(p["MonthlyIncome"])
                theme.factor_card(u_, "월 소득 (단위: 달러/월)", k_)
            else:
                theme.factor_card("미입력", "월 소득 (단위: 달러/월)")

    with right, theme.panel("mix"):
        theme.section("계좌 구성 (단위: 개)")
        n_total, n_re = p["NumberOfOpenCreditLinesAndLoans"], p["NumberRealEstateLoansOrLines"]
        if n_re > n_total:
            st.caption("⚠️ 부동산 대출 수가 보유 대출·카드 수보다 많습니다. 입력값을 확인하세요.")
        if "MonthlyIncome" in p:
            u_, k_ = units.money_lines(p["MonthlyIncome"])
            st.markdown(f"**Total: {u_}/월**  \n<span class='rp-unit'>(단위 : 달러) · {k_}</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Total: 월 소득 미입력**")
        st.plotly_chart(_donut(n_total, n_re), **STRETCH)
        st.markdown(
            "<table class='rp-legend'>"
            f"<tr><td><span style='color:#e0a840'>●</span></td><td>그 외 대출·카드</td><td><b>{max(n_total - n_re, 0)}개</b></td></tr>"
            f"<tr><td><span style='color:#2b2b2b'>●</span></td><td>부동산 대출</td><td><b>{n_re}건</b></td></tr>"
            f"<tr><td><span style='color:#8a8f98'>●</span></td><td>부양 가족</td><td><b>{p.get('NumberOfDependents', '미입력')}"
            f"{'명' if 'NumberOfDependents' in p else ''}</b></td></tr></table>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 탭 2. XAI 분석
# ----------------------------------------------------------------------
def tab_explain(p: dict):
    with theme.panel("xai"):
        c1, c2 = st.columns([4, 1])
        with c1:
            theme.section("XAI 분석 (SHAP)")
            st.caption(SHAP_SIGN_NOTE + ". 기여도는 모델 출력(log-odds) 단위이며 단위 없는 값입니다.")
        with c2:
            run = st.button("설명 생성", type="primary", key="run_explain", **STRETCH)
        if run:
            # 이미지까지 한 번에 받는 /explain/plots 를 우선 사용, 서버에 없으면 /explain 으로 대체
            res, err = api_post("/explain/plots", p)
            if err and "404" in err:
                res, err = api_post("/explain", p)
            if err:
                st.error(err)
                return
            st.session_state["explain"] = res
        res = st.session_state.get("explain")
        if not res:
            st.info("왼쪽에 고객 정보를 입력하고 [설명 생성]을 누르세요.")
            return
        approved = res["approval_status"] == "APPROVED"
        theme.pill("✅ 승인" if approved else "❌ 거절", approved)

        reasons = res.get("rejection_reasons") or []
        if reasons:
            total = res.get("total_rejection_factors")
            st.markdown(f"##### 부도 위험을 높인 주요 요인 Top {len(reasons)}"
                        + (f"  ·  전체 {total}개 중" if total else ""))
            for r in reasons:
                name = r.get("feature_name_kr") or r.get("feature_label") or label(r["feature"])
                st.markdown(f"**{r.get('rank', '')}. {name}** — 기여도 {r['shap_contribution']:+.3f}  \n{r['explanation']}")
                if r["feature"] == "MonthlyIncome":     # 금액은 달러 아래에 원화를 함께 표기
                    parts = []
                    if r.get("current_value") is not None:
                        parts.append(f"입력 {units.krw(r['current_value'])}")
                    if r.get("reference_value") is not None:
                        parts.append(f"중앙값 {units.krw(r['reference_value'])}")
                    if parts:
                        st.caption("원화 : " + " / ".join(parts))
        else:
            st.info("거절 사유가 없습니다(승인 건이거나 부도 위험을 높인 요인이 없음).")

        wf = _png(res.get("waterfall_png"))
        if wf:
            st.markdown("##### 이 고객의 SHAP Waterfall")
            st.image(wf, **STRETCH)
        elif "waterfall_png" in res:
            st.caption("Waterfall 그림을 만들지 못했습니다.")

        summ, deps = _png(res.get("summary_png")), res.get("dependence_png") or {}
        if summ or deps:
            with st.expander("모델 전체 설명 (Summary / Dependence)", expanded=False):
                if summ:
                    st.markdown("**Summary plot** - 모델이 중요하게 보는 변수 순서")
                    st.image(summ, **STRETCH)
                for feat, b64 in deps.items():
                    img = _png(b64)
                    if img:
                        st.markdown(f"**Dependence plot** - {label(feat)}")
                        st.image(img, **STRETCH)

        shap_vals = res.get("shap_values") or {}
        if shap_vals and "waterfall_png" not in res:    # /explain 응답일 때만 막대 그래프로 대체
            s = pd.Series(shap_vals).sort_values(key=abs).tail(15)
            fig = go.Figure(go.Bar(
                x=s.values, y=[label(i) for i in s.index], orientation="h",
                marker_color=[ORANGE if v > 0 else BLUE for v in s.values],
                text=[f"{v:+.3f}" for v in s.values], textposition="outside"))
            fig.update_layout(title="SHAP 기여도 (주황: 부도 위험 ↑ / 파랑: 부도 위험 ↓)", height=480,
                              margin=dict(l=10, r=40, t=50, b=10), xaxis_title="SHAP 값 (log-odds, 단위 없음)")
            st.plotly_chart(fig, **STRETCH)


# ----------------------------------------------------------------------
# 탭 4. 모델 성능
# ----------------------------------------------------------------------
def tab_metrics():
    with theme.panel("metrics"):
        theme.section("모델 성능 (Test 데이터)")
        m, err = api_get("/metrics")
        if err:
            st.error(err)
            return
        specs = [("auc", "AUC (0~1)", False), ("ks", "KS (0~1)", False), ("precision", "정밀도 Precision (%)", True),
                 ("recall", "재현율 Recall (%)", True), ("f1", "F1 점수 (%)", True)]
        for col, (k, name, as_pct) in zip(st.columns(5), specs):
            col.metric(name, units.pct(m[k], 2) if as_pct else f"{m[k]:.4f}")
        goals = [("auc", "AUC (0~1)", 0.78), ("ks", "KS (0~1)", 0.28)]
        rows = [{"지표": n, "값": round(m[k], 4), "목표": f"0.78 이상" if k == "auc" else "0.28 이상",
                 "판정": "✅ 통과" if m[k] >= g else "❌ 미달"} for k, n, g in goals]
        st.markdown("##### 성능 목표")
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
        st.caption("씬파일러 AUC 향상(+0.03 이상)과 PSI(0.1 미만)는 /metrics 응답에 포함되지 않아 이 탭에서 표시하지 않습니다.")
        h, _ = api_get("/health")
        if h:
            st.caption(f"모델: {h['model_version']} · 서버 시각 {h['timestamp']}")


# ----------------------------------------------------------------------
# 탭 5. 통계 검증
# ----------------------------------------------------------------------
def tab_anova():
    a, err = api_get("/anova")
    with theme.panel("anova"):
        theme.section("통계 검증 (One-way ANOVA)")
        if err:
            st.error(err)
            return
        for key, title in [("test_1_feature_group", "검증 1. 피처 그룹별 AUC (전통 / 대안 / 통합)"),
                           ("test_2_algorithm", "검증 2. 알고리즘별 AUC (LR / XGBoost / LightGBM)")]:
            t = a[key]
            st.markdown(f"##### {title}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F 통계량 (단위 없음)", f"{t['f_statistic']:.2f}")
            c2.metric("p-value (0~1)", f"{t['p_value']:.2e}")
            c3.metric("효과 크기 η² (0~1)", f"{t['effect_size_eta_squared']:.3f}")
            c4.metric("유의 여부 (α=0.05)", "유의함" if t["is_significant"] else "유의하지 않음")
            means = t["group_means"]
            fig = go.Figure(go.Bar(x=list(means), y=list(means.values()), marker_color=BLUE,
                                   text=[f"{v:.4f}" for v in means.values()], textposition="outside"))
            lo = min(means.values())
            fig.update_layout(height=300, yaxis_title="평균 AUC (0~1)", margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_range=[max(0, lo - 0.05), min(1, max(means.values()) + 0.02)])
            st.plotly_chart(fig, **STRETCH)
            if not t["is_significant"]:
                st.info("그룹 간 평균 AUC 차이가 통계적으로 유의하지 않습니다.")
            elif t.get("post_hoc_tukey"):
                st.markdown("사후 검정 (Tukey HSD) · 평균 차이 단위: AUC")
                st.dataframe(pd.DataFrame(t["post_hoc_tukey"]), hide_index=True, **STRETCH)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    theme.inject_css()
    theme.header("Credit Report Dashboard", "XAI 기반 대안신용평가 · 신용 점수, 설명, 공정성을 한 화면에서 확인합니다. "
                 f"금액 단위: 달러 ({units.FX_NOTE})")
    st.warning(DISCLAIMER)
    p = sidebar_inputs()
    t1, t2, t3, t4, t5 = st.tabs(["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"])
    with t1:
        tab_predict(p)
    with t2:
        tab_explain(p)
    with t3:
        render_fairness_tab()
    with t4:
        tab_metrics()
    with t5:
        tab_anova()
    theme.footer(DISCLAIMER)


main()
