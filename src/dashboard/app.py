"""
src/dashboard/app.py
XAI 기반 대안신용평가 시스템 - Streamlit 대시보드 (5개 탭)

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

MSA 원칙: 모델·전처리기(.pkl)나 reports/ 파일을 직접 읽지 않고, 모든 데이터는 FastAPI에 HTTP로만 요청한다.
  탭1 고객 심사  : POST /predict
  탭2 XAI 분석   : POST /explain
  탭3 공정성     : GET  /fairness/mitigation, /fairness/detail   (fairness_tab.py)
  탭4 모델 성능  : GET  /metrics, /health
  탭5 통계 검증  : GET  /anova

실행: 프로젝트 루트에서  API_URL=http://localhost:8000 streamlit run src/dashboard/app.py
      (Docker 에서는 API_URL=http://api:8000)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# `streamlit run src/dashboard/app.py` 로 실행해도 src 패키지를 찾도록 루트 추가
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.fairness_tab import STRETCH, render_fairness_tab  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DISCLAIMER = "※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8f98"

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
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
        if r.status_code == 422:   # 입력값 검증 실패 → 어떤 칸이 문제인지 보여줌
            msgs = [f"{'.'.join(str(x) for x in d['loc'][1:])}: {d['msg']}" for d in r.json().get("detail", [])]
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


# ----------------------------------------------------------------------
# 사이드바 입력 (탭1·탭2 공용)
# ----------------------------------------------------------------------
def sidebar_inputs() -> dict:
    """필수 입력은 항상 값을 보내고, 선택 입력(결측 가능 컬럼)은 체크하면 값을 보낸다."""
    st.sidebar.header("고객 정보 입력")
    st.sidebar.caption("필수 항목은 모두 입력, 선택 항목은 체크한 경우에만 전송합니다.")
    p: dict = {}

    st.sidebar.subheader("기존 금융 변수 (필수)")
    p["RevolvingUtilizationOfUnsecuredLines"] = st.sidebar.number_input("회전 한도 사용률", 0.0, 10.0, 0.3, 0.05)
    p["DebtRatio"] = st.sidebar.number_input("부채 비율", 0.0, 10.0, 0.3, 0.05)
    p["NumberOfOpenCreditLinesAndLoans"] = st.sidebar.number_input("개설 대출·카드 수", 0, 60, 6)
    p["NumberRealEstateLoansOrLines"] = st.sidebar.number_input("부동산 대출 수", 0, 30, 1)
    p["card_history_months"] = st.sidebar.number_input("카드 이용 기간(개월)", 0, 360, 60)

    st.sidebar.subheader("기존 금융 변수 (선택)")
    if st.sidebar.checkbox("월 소득 입력", True):
        p["MonthlyIncome"] = st.sidebar.number_input("월 소득", 0.0, 1_000_000.0, 5000.0, 100.0)
    if st.sidebar.checkbox("부양 가족 수 입력", True):
        p["NumberOfDependents"] = st.sidebar.number_input("부양 가족 수", 0, 20, 1)
    if st.sidebar.checkbox("30~59일 연체 횟수 입력", True):
        p["past_due_30_59"] = st.sidebar.number_input("30~59일 연체 (최근 2년)", 0, 24, 0)
    if st.sidebar.checkbox("60~89일 연체 횟수 입력", True):
        p["past_due_60_89"] = st.sidebar.number_input("60~89일 연체 (최근 2년)", 0, 12, 0)
    if st.sidebar.checkbox("90일 이상 연체 횟수 입력", True):
        p["past_due_90"] = st.sidebar.number_input("90일 이상 연체", 0, 17, 0)

    st.sidebar.subheader("대안 변수 (필수)")
    p["telecom_payment_rate"] = st.sidebar.slider("통신비 납부율", 0.0, 1.0, 0.95, 0.01)
    p["utility_payment_rate"] = st.sidebar.slider("공과금 납부율", 0.0, 1.0, 0.95, 0.01)
    p["spending_consistency"] = st.sidebar.slider("소비 일관성 (0~100)", 0.0, 100.0, 60.0, 1.0)
    p["regular_payment_count"] = st.sidebar.number_input("정기 결제 건수", 0, 100, 4)
    p["app_login_frequency"] = st.sidebar.number_input("앱 로그인 빈도", 0, 1000, 20)
    return p


# ----------------------------------------------------------------------
# 탭
# ----------------------------------------------------------------------
def tab_predict(payload: dict):
    st.subheader("고객 심사")
    st.caption("왼쪽 사이드바에 고객 정보를 입력하고 [심사 실행]을 누르세요.")
    if st.button("심사 실행", type="primary", key="run_predict"):
        res, err = api_post("/predict", payload)
        if err:
            st.error(err)
            return
        st.session_state["predict"] = res
    res = st.session_state.get("predict")
    if not res:
        return
    approved = res["approval_status"] == "APPROVED"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("승인 여부", "✅ 승인" if approved else "❌ 거절")
    c2.metric("신용 점수 (0~1000)", res["credit_score"])
    c3.metric("부도 확률", f"{res['default_probability']:.2%}")
    c4.metric("리스크 등급", res["risk_grade"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=res["credit_score"], number={"suffix": " 점"},
        gauge={"axis": {"range": [0, 1000]}, "bar": {"color": BLUE if approved else ORANGE}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
    st.plotly_chart(fig, **STRETCH)


def tab_explain(payload: dict):
    st.subheader("XAI 분석 (SHAP)")
    st.caption("각 변수가 부도 확률을 올린 정도(+)와 내린 정도(−)를 보여줍니다.")
    if st.button("설명 생성", type="primary", key="run_explain"):
        res, err = api_post("/explain", payload)
        if err:
            st.error(err)
            return
        st.session_state["explain"] = res
    res = st.session_state.get("explain")
    if not res:
        return
    st.markdown(f"**결과: {'✅ 승인' if res['approval_status'] == 'APPROVED' else '❌ 거절'}**")

    reasons = res.get("rejection_reasons") or []
    if reasons:
        st.markdown("##### 부도 확률을 높인 주요 요인 Top 5")
        for i, r in enumerate(reasons, 1):
            v = r.get("current_value")
            st.markdown(f"{i}. **{r['feature']}** (입력값 {'미입력' if v is None else v}) "
                        f"— 기여도 +{r['shap_contribution']:.3f}  \n   {r['explanation']}")
    else:
        st.info("거절 사유가 없습니다(승인 건이거나 부도 확률을 높인 요인이 없음).")

    shap_vals = res.get("shap_values") or {}
    if shap_vals:
        s = pd.Series(shap_vals).sort_values(key=abs).tail(15)
        fig = go.Figure(go.Bar(
            x=s.values, y=s.index, orientation="h",
            marker_color=[ORANGE if v > 0 else BLUE for v in s.values],
            text=[f"{v:+.3f}" for v in s.values], textposition="outside"))
        fig.update_layout(title="SHAP 기여도 (주황: 부도 확률 ↑ / 파랑: 부도 확률 ↓)",
                          height=480, margin=dict(l=10, r=40, t=50, b=10), xaxis_title="SHAP 값")
        st.plotly_chart(fig, **STRETCH)


def tab_metrics():
    st.subheader("모델 성능")
    m, err = api_get("/metrics")
    if err:
        st.error(err)
        return
    cols = st.columns(5)
    for col, (k, label) in zip(cols, [("auc", "AUC"), ("ks", "KS"), ("precision", "Precision"),
                                       ("recall", "Recall"), ("f1", "F1")]):
        col.metric(label, f"{m[k]:.4f}")
    goals = {"auc": ("AUC", 0.78), "ks": ("KS", 0.28)}
    rows = [{"지표": n, "값": m[k], "목표": f">= {g}", "판정": "✅ PASS" if m[k] >= g else "❌ FAIL"}
            for k, (n, g) in goals.items()]
    st.markdown("##### 성능 목표 (Test)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    st.caption("씬파일러 AUC 향상(≥ +0.03)과 PSI(< 0.1)는 /metrics 응답에 포함되지 않아 이 탭에서 표시하지 않습니다.")
    h, _ = api_get("/health")
    if h:
        st.caption(f"모델: {h['model_version']} · 서버 시각 {h['timestamp']}")


def tab_anova():
    st.subheader("통계 검증 (One-way ANOVA)")
    a, err = api_get("/anova")
    if err:
        st.error(err)
        return
    for key, title in [("test_1_feature_group", "검증 1. 피처 그룹별 AUC (전통 / 대안 / 통합)"),
                       ("test_2_algorithm", "검증 2. 알고리즘별 AUC (LR / XGBoost / LightGBM)")]:
        t = a[key]
        st.markdown(f"##### {title}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F 통계량", f"{t['f_statistic']:.2f}")
        c2.metric("p-value", f"{t['p_value']:.2e}")
        c3.metric("효과 크기 η²", f"{t['effect_size_eta_squared']:.3f}")
        c4.metric("유의 여부", "유의함" if t["is_significant"] else "유의하지 않음")
        means = t["group_means"]
        fig = go.Figure(go.Bar(x=list(means), y=list(means.values()), marker_color=BLUE,
                               text=[f"{v:.4f}" for v in means.values()], textposition="outside"))
        lo = min(means.values())
        fig.update_layout(height=300, yaxis_title="평균 AUC", margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_range=[max(0, lo - 0.05), min(1, max(means.values()) + 0.02)])
        st.plotly_chart(fig, **STRETCH)
        if not t["is_significant"]:
            st.info("그룹 간 평균 AUC 차이가 통계적으로 유의하지 않습니다.")
        elif t.get("post_hoc_tukey"):
            st.markdown("사후 검정 (Tukey HSD)")
            st.dataframe(pd.DataFrame(t["post_hoc_tukey"]), hide_index=True, **STRETCH)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    st.title("XAI 기반 대안신용평가 시스템")
    st.warning(DISCLAIMER)
    payload = sidebar_inputs()
    t1, t2, t3, t4, t5 = st.tabs(["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"])
    with t1:
        tab_predict(payload)
    with t2:
        tab_explain(payload)
    with t3:
        render_fairness_tab()
    with t4:
        tab_metrics()
    with t5:
        tab_anova()
    st.caption(DISCLAIMER)


main()
