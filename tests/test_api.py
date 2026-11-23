"""
tests/test_api.py
FastAPI 엔드포인트 통합 테스트 (서버를 띄우지 않고 TestClient 로 호출)
저장된 모델·전처리기·분할 데이터·ANOVA 결과가 있을 때만 실행 (없으면 건너뜀)

확인하는 것
  1) /health, /metrics, /anova 가 메타 파일과 같은 값을 돌려줌
  2) /predict: 신용 점수·승인 여부·등급이 부도확률과 서로 맞음
  3) /explain: /predict 와 같은 판정, 거절 사유 형식, 같은 입력이면 같은 결과
  4) 월 소득이 없으면 부채 비율이 결과에 영향을 주지 않고, 화면에도 그렇게 안내
  5) 잘못된 입력은 422
  6) /explain/plots: PNG 이미지와 dependence dict
  7) main.py 내부 함수: result_from_p(승인·점수·등급 규칙), raw_to_features(여러 행)

필요 패키지: fastapi, httpx (TestClient 가 사용)
실행: 프로젝트 최상위 폴더에서  pytest tests/test_api.py -v
"""
import base64
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

NEEDED = [PROJECT_ROOT / "data" / "processed" / "X_train.csv",
          PROJECT_ROOT / "reports" / "anova_results.json"]
HAS_ARTIFACTS = bool(list((PROJECT_ROOT / "models").glob("model_meta_v*.json"))) and all(p.exists() for p in NEEDED)
pytestmark = pytest.mark.skipif(not HAS_ARTIFACTS, reason="models/ 또는 data/processed/, reports/ 파일이 없음")

GOOD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.01, "DebtRatio": 0.24, "MonthlyIncome": 10000.0,
    "NumberOfOpenCreditLinesAndLoans": 9, "NumberRealEstateLoansOrLines": 1, "NumberOfDependents": 0,
    "card_history_months": 211, "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfTimes90DaysLate": 0,
    "telecom_payment_rate": 0.99, "utility_payment_rate": 0.98, "spending_consistency": 80.0,
    "regular_payment_count": 3, "app_login_frequency": 55,
}
BAD = {
    **GOOD, "RevolvingUtilizationOfUnsecuredLines": 0.99, "DebtRatio": 1.2, "MonthlyIncome": 2500.0,
    "NumberOfTime30-59DaysPastDueNotWorse": 3, "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberOfTimes90DaysLate": 1, "telecom_payment_rate": 0.3, "utility_payment_rate": 0.35,
    "spending_consistency": 10.0, "app_login_frequency": 2,
}
EXCLUDED_PREFIXES = ("is_thin_filer", "missingindicator_", "past_due_code_flag")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:           # with: lifespan(모델·전처리기 로드)이 실행됨
        yield c


@pytest.fixture(scope="module")
def meta(client):
    from src.api import main
    return main.art["meta"]


def _ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


# ----------------------------------------------------------------------
# 1) 상태·메타 값
# ----------------------------------------------------------------------
def test_health(client, meta):
    body = _ok(client.get("/health"))
    assert body["status"] == "healthy"
    assert body["model_version"] == meta["best_model"]["algorithm"]


def test_metrics_match_meta(client, meta):
    body = _ok(client.get("/metrics"))
    fm = meta["final_metrics"]
    for key, meta_key in [("auc", "test_auc"), ("ks", "test_ks"), ("f1", "test_f1"),
                          ("precision", "test_precision"), ("recall", "test_recall")]:
        assert body[key] == pytest.approx(fm[meta_key]), key


def test_anova_has_both_tests(client):
    body = _ok(client.get("/anova"))
    assert {"test_1_feature_group", "test_2_algorithm"} <= set(body)


# ----------------------------------------------------------------------
# 2) /predict
# ----------------------------------------------------------------------
@pytest.mark.parametrize("payload", [GOOD, BAD], ids=["good", "bad"])
def test_predict_fields_are_consistent(client, meta, payload):
    import numpy as np
    body = _ok(client.post("/predict", json=payload))
    p, bm = body["default_probability"], meta["best_model"]
    assert 0 <= p <= 1
    assert body["credit_score"] == pytest.approx(round((1 - p) * 1000), abs=1)
    # 응답의 부도확률은 소수 4자리로 반올림돼 있어, 경계값 바로 근처면 판정 비교를 건너뜀
    if all(abs(p - c) > 1e-4 for c in bm["grade_cutoffs"] + [bm["approval_cutoff"]]):
        assert body["approval_status"] == ("APPROVED" if p <= bm["approval_cutoff"] else "REJECTED")
        expected_grade = bm["grade_labels"][int(np.searchsorted(bm["grade_cutoffs"], p, side="left"))]
        assert body["risk_grade"] == expected_grade
    # A~C 승인 / D~E 거절
    assert (body["risk_grade"] in "ABC") == (body["approval_status"] == "APPROVED")


def test_bad_customer_riskier_than_good(client):
    good = _ok(client.post("/predict", json=GOOD))["default_probability"]
    bad = _ok(client.post("/predict", json=BAD))["default_probability"]
    assert bad > good


# ----------------------------------------------------------------------
# 3) /explain
# ----------------------------------------------------------------------
def test_explain_agrees_with_predict(client):
    pred = _ok(client.post("/predict", json=BAD))
    exp = _ok(client.post("/explain", json=BAD))
    assert exp["approval_status"] == pred["approval_status"]


def test_explain_reason_format(client, meta):
    body = _ok(client.post("/explain", json=BAD))
    reasons = body["rejection_reasons"]
    assert 0 < len(reasons) <= 5
    contrib = [r["shap_contribution"] for r in reasons]
    assert all(c > 0 for c in contrib) and contrib == sorted(contrib, reverse=True)
    assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
    assert body["total_rejection_factors"] >= len(reasons)
    for r in reasons:
        assert not r["feature"].startswith(EXCLUDED_PREFIXES)
        assert r["feature_name_kr"] and r["feature_label"] == r["feature_name_kr"]
        assert "평균" not in r["explanation"]                 # 비교 기준은 중앙값
    assert set(body["shap_values"]) == set(meta["feature_names"])


def test_explain_is_deterministic(client):
    a = _ok(client.post("/explain", json=BAD))
    b = _ok(client.post("/explain", json=BAD))
    assert a["rejection_reasons"] == b["rejection_reasons"]


def test_approved_customer_reasons_respect_min_contribution(client):
    from src.api import main
    body = _ok(client.post("/explain", json=GOOD))
    if body["approval_status"] != "APPROVED":
        pytest.skip("GOOD 고객이 이 모델에서는 거절됨")
    m = main.art["explainer"].min_contribution
    assert all(r["shap_contribution"] > m for r in body["rejection_reasons"])


# ----------------------------------------------------------------------
# 4) 월 소득이 없을 때 부채 비율
# ----------------------------------------------------------------------
def test_debt_ratio_ignored_without_income(client):
    no_income = {k: v for k, v in BAD.items() if k != "MonthlyIncome"}
    low = _ok(client.post("/predict", json={**no_income, "DebtRatio": 0.1}))
    high = _ok(client.post("/predict", json={**no_income, "DebtRatio": 9.0}))
    assert low["default_probability"] == pytest.approx(high["default_probability"])


def test_unused_debt_ratio_is_explained(client):
    no_income = {k: v for k, v in BAD.items() if k != "MonthlyIncome"}
    body = _ok(client.post("/explain", json={**no_income, "DebtRatio": 9.0}))
    for r in body["rejection_reasons"]:
        if r["feature"] == "DebtRatio":
            assert r["current_value"] is None                 # 입력한 9.0 을 보여 주면 안 됨
            assert "월 소득" in r["explanation"]


# ----------------------------------------------------------------------
# 5) 잘못된 입력
# ----------------------------------------------------------------------
@pytest.mark.parametrize("change", [
    {"DebtRatio": 15.0}, {"RevolvingUtilizationOfUnsecuredLines": 11.0},
    {"telecom_payment_rate": 1.5}, {"MonthlyIncom": 3000},
], ids=["debt>10", "revolving>10", "rate>1", "typo_field"])
def test_invalid_input_returns_422(client, change):
    for path in ("/predict", "/explain"):
        assert client.post(path, json={**GOOD, **change}).status_code == 422


# ----------------------------------------------------------------------
# 6) /explain/plots
# ----------------------------------------------------------------------
def test_explain_plots_returns_images(client):
    body = _ok(client.post("/explain/plots", json=BAD))

    def is_png(s):
        return isinstance(s, str) and base64.b64decode(s)[:8] == b"\x89PNG\r\n\x1a\n"

    assert is_png(body["waterfall_png"])
    assert is_png(body["summary_png"])
    assert isinstance(body["dependence_png"], dict) and body["dependence_png"]
    assert all(is_png(v) for v in body["dependence_png"].values())
    json.dumps(body)                                           # 응답이 JSON 으로 직렬화되는지


def test_explain_plots_dependence_keys_are_feature_names(client, meta):
    body = _ok(client.post("/explain/plots", json=BAD))
    keys = list(body["dependence_png"])
    assert len(keys) == 2
    assert all(k in meta["feature_names"] for k in keys)      # 영어 변수명 (대시보드가 키로 씀)
    assert not any(k.startswith(EXCLUDED_PREFIXES) for k in keys)


def test_explain_and_plots_give_same_reasons(client):
    a = _ok(client.post("/explain", json=BAD))
    b = _ok(client.post("/explain/plots", json=BAD))
    assert a["rejection_reasons"] == b["rejection_reasons"]
    assert a["total_rejection_factors"] == b["total_rejection_factors"]


# ----------------------------------------------------------------------
# 7) main.py 내부 함수
# ----------------------------------------------------------------------
def test_result_from_p_rules(client, meta):
    from src.api.main import result_from_p
    import numpy as np
    bm = meta["best_model"]
    cut = bm["approval_cutoff"]
    assert result_from_p(cut)["approval_status"] == "APPROVED"            # 기준과 같으면 승인 (<=)
    assert result_from_p(cut + 1e-6)["approval_status"] == "REJECTED"
    for p in (0.01, 0.05, cut, 0.5, 0.95):
        r = result_from_p(p)
        assert r["credit_score"] == int(round((1 - p) * 1000))
        assert 0 <= r["credit_score"] <= 1000
        assert r["risk_grade"] == bm["grade_labels"][int(np.searchsorted(bm["grade_cutoffs"], p, side="left"))]


def test_predict_result_equals_model_probability(client):
    from src.api import main
    req = main.schema.predict_request(**BAD)
    X = main.to_features(req)
    p = float(main.art["model"].predict_proba(X)[:, 1][0])
    assert main.predict_result(X) == main.result_from_p(p)


def test_raw_to_features_handles_many_rows(client):
    """여러 고객을 한 번에 넣어도 한 명씩 넣은 결과와 같음 (/counterfactual 이 사용)"""
    import pandas as pd
    import numpy as np
    from src.api import main
    reqs = [main.schema.predict_request(**c) for c in (GOOD, BAD)]
    many = main.raw_to_features(pd.DataFrame([r.model_dump(by_alias=True) for r in reqs]))
    one_by_one = [main.to_features(r) for r in reqs]
    model = main.art["model"]
    assert np.allclose(model.predict_proba(many)[:, 1],
                       [model.predict_proba(x)[:, 1][0] for x in one_by_one])