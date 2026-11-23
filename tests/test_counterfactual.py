"""
tests/test_counterfactual.py
반사실적 설명 엔드포인트(/counterfactual) 통합 테스트 (서버를 띄우지 않고 TestClient 로 호출)
저장된 모델·전처리기·분할 데이터가 있고 dice-ml 이 설치돼 있을 때만 실행 (없으면 건너뜀)

확인하는 것
  1) 응답 형식: 지금 결과가 /predict 와 같고, 방안마다 rank 1..n, 요청 개수 이하
  2) 방안이 진짜인지: 제안대로 바꿔서 /predict 에 다시 넣으면 실제로 승인 + 같은 부도 확률
  3) 현실성 규칙: 바꿀 수 있는 변수만, 정해진 방향으로만, 입력 범위 안, 소득 30%·소비 일관성 30점 제한
  4) 표시: 자릿수(비율 1%p, 건수·점수 정수), 단위 문자열이 feature_spec.format_value 와 같음
  5) 이미 승인된 고객, 월 소득 미입력 고객, 같은 입력 반복, 잘못된 입력(422), 준비 실패(503)

필요 패키지: fastapi, httpx, dice-ml
실행: 프로젝트 최상위 폴더에서  python -m pytest tests/test_counterfactual.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("dice_ml")

PROC = PROJECT_ROOT / "data" / "processed"
NEEDED = [PROC / "X_train.csv", PROC / "raw_train.csv", PROC / "y_train.csv",
          PROJECT_ROOT / "reports" / "anova_results.json"]
HAS_ARTIFACTS = bool(list((PROJECT_ROOT / "models").glob("model_meta_v*.json"))) and all(p.exists() for p in NEEDED)
pytestmark = pytest.mark.skipif(not HAS_ARTIFACTS, reason="models/ 또는 data/processed/(raw_train, y_train), reports/ 파일이 없음")

from src.data.feature_spec import FEATURE_SPEC, format_value, label  # noqa: E402

# 거절 고객: 사용률 99%, 30~59일 연체 5회, 대안 데이터가 나쁨 (부도 확률 약 86%로 확인한 고객)
REJECTED = {
    "RevolvingUtilizationOfUnsecuredLines": 0.9902, "DebtRatio": 0.2657, "MonthlyIncome": 8000.0,
    "NumberOfOpenCreditLinesAndLoans": 12, "NumberRealEstateLoansOrLines": 0, "NumberOfDependents": 0,
    "card_history_months": 339, "NumberOfTime30-59DaysPastDueNotWorse": 5,
    "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfTimes90DaysLate": 0,
    "telecom_payment_rate": 0.3169, "utility_payment_rate": 0.4995, "spending_consistency": 14.2089,
    "regular_payment_count": 24, "app_login_frequency": 3,
}
# 승인 고객: 사용률 1%, 연체 없음, 납부율 높음
APPROVED = {
    "RevolvingUtilizationOfUnsecuredLines": 0.01, "DebtRatio": 0.24, "MonthlyIncome": 10000.0,
    "NumberOfOpenCreditLinesAndLoans": 9, "NumberRealEstateLoansOrLines": 1, "NumberOfDependents": 0,
    "card_history_months": 211, "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfTimes90DaysLate": 0,
    "telecom_payment_rate": 0.99, "utility_payment_rate": 0.98, "spending_consistency": 80.0,
    "regular_payment_count": 3, "app_login_frequency": 55,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:           # with: lifespan(모델·전처리기·DiCE 데이터 준비)이 실행됨
        yield c


@pytest.fixture(scope="module")
def main_mod(client):
    from src.api import main
    if main.art.get("cf_data") is None:
        pytest.skip("서버 시작 시 반사실적 설명 준비 실패 (로그의 '반사실적 설명 준비 실패' 확인)")
    return main


@pytest.fixture(scope="module")
def cf(client, main_mod):
    """거절 고객의 반사실적 설명 (여러 테스트가 같이 씀)"""
    return _ok(client.post("/counterfactual", json={**REJECTED, "total_cfs": 3}))


@pytest.fixture(scope="module")
def options(cf):
    if not cf["counterfactuals"]:
        pytest.skip("이 고객에 대해 방안을 찾지 못함 (모델이 바뀌었으면 REJECTED 고객 값을 조정)")
    return cf["counterfactuals"]


def _ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


def _digits(name):
    from src.api.main import CF_PRECISION
    return CF_PRECISION.get(FEATURE_SPEC[name].get("kind"), 2)


def _apply(request: dict, option: dict) -> dict:
    """제안대로 바꾼 요청 (건수 변수는 int 로)"""
    new = dict(request)
    for ch in option["changes"]:
        v = ch["suggested_value"]
        new[ch["feature"]] = int(round(v)) if FEATURE_SPEC[ch["feature"]]["kind"] == "count" else v
    return new


# ----------------------------------------------------------------------
# 1) 응답 형식
# ----------------------------------------------------------------------
def test_original_matches_predict(client, cf):
    pred = _ok(client.post("/predict", json=REJECTED))
    assert cf["original"] == pred
    assert cf["original"]["approval_status"] == "REJECTED"


def test_finds_at_least_one_option(cf):
    assert len(cf["counterfactuals"]) >= 1, cf["message"]
    assert "가지 중 하나를" in cf["message"]


def test_rank_and_count(options):
    assert [o["rank"] for o in options] == list(range(1, len(options) + 1))
    assert len(options) <= 3


def test_fewer_changes_come_first(options):
    sizes = [len(o["changes"]) for o in options]
    assert sizes == sorted(sizes)


def test_every_option_is_approved(main_mod, options):
    cutoff = main_mod.art["meta"]["best_model"]["approval_cutoff"]
    for o in options:
        assert o["approval_status"] == "APPROVED"
        assert o["default_probability"] <= cutoff
        # 점수는 반올림 전 확률로 계산하므로 1점까지 차이 허용
        assert abs(o["credit_score"] - (1 - o["default_probability"]) * 1000) <= 1


# ----------------------------------------------------------------------
# 2) 방안이 진짜인지: 제안대로 바꿔서 /predict 에 넣어 보기
# ----------------------------------------------------------------------
def test_suggestion_really_gets_approved(client, options):
    for o in options:
        pred = _ok(client.post("/predict", json=_apply(REJECTED, o)))
        assert pred["approval_status"] == "APPROVED", (o, pred)
        assert pred["default_probability"] == pytest.approx(o["default_probability"], abs=1e-4)
        assert pred["risk_grade"] == o["risk_grade"]


# ----------------------------------------------------------------------
# 3) 현실성 규칙
# ----------------------------------------------------------------------
def test_varied_and_fixed_split_inputs(main_mod, cf):
    varied, fixed = set(cf["varied_features"]), set(cf["fixed_features"])
    assert not varied & fixed
    assert varied | fixed == set(main_mod.art["cf_cols"])
    assert varied <= set(main_mod.CF_ACTIONABLE)
    # 조작 가능한 변수, 바꿀 수 없는 기록은 항상 고정
    for name in ("app_login_frequency", "regular_payment_count", "card_history_months",
                 "NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"):
        assert name in fixed


def test_only_varied_features_change(cf, options):
    for o in options:
        for ch in o["changes"]:
            assert ch["feature"] in cf["varied_features"]


def test_direction_is_respected(main_mod, options):
    for o in options:
        for ch in o["changes"]:
            direction = main_mod.CF_ACTIONABLE[ch["feature"]]
            if direction == "down":
                assert ch["suggested_value"] <= ch["current_value"] + 1e-9, ch
            elif direction == "up":
                # 반올림 때문에 현재 값보다 아주 조금 작아질 수 있음 (예: 0.3169 -> 0.31 은 안 되지만 0.32 는 됨)
                assert ch["suggested_value"] >= round(ch["current_value"], _digits(ch["feature"])) - 1e-9, ch


def test_values_within_input_range(options):
    for o in options:
        for ch in o["changes"]:
            spec = FEATURE_SPEC[ch["feature"]]
            if spec.get("min") is not None:
                assert ch["suggested_value"] >= spec["min"], ch
            if spec.get("max") is not None:
                assert ch["suggested_value"] <= spec["max"], ch   # 예: 납부율 110% 같은 값이 없어야 함


def test_max_increase_limits(main_mod, options):
    for o in options:
        for ch in o["changes"]:
            f, cur, new = ch["feature"], ch["current_value"], ch["suggested_value"]
            if f in main_mod.CF_MAX_UP_RATIO:
                assert new <= cur * main_mod.CF_MAX_UP_RATIO[f] + 1, ch      # +1: 정수 반올림 여유
            if f in getattr(main_mod, "CF_MAX_UP_STEP", {}):
                assert new <= cur + main_mod.CF_MAX_UP_STEP[f] + 1, ch


# ----------------------------------------------------------------------
# 4) 표시
# ----------------------------------------------------------------------
def test_precision(options):
    for o in options:
        for ch in o["changes"]:
            v, d = ch["suggested_value"], _digits(ch["feature"])
            assert v == pytest.approx(round(v, d)), ch               # 비율 0.01 단위, 건수·점수 정수


def test_changes_are_real_and_displayed(options):
    for o in options:
        for ch in o["changes"]:
            f = ch["feature"]
            # 반올림 차이만 있는 항목(0.9902 -> 0.99)은 '바뀐 항목'에 없어야 함
            assert abs(ch["suggested_value"] - round(ch["current_value"], _digits(f))) > 1e-6, ch
            assert ch["feature_name_kr"] == label(f)
            assert ch["current_display"] == format_value(f, ch["current_value"])
            assert ch["suggested_display"] == format_value(f, ch["suggested_value"])


# ----------------------------------------------------------------------
# 5) 여러 상황
# ----------------------------------------------------------------------
def test_same_input_same_result(client, cf):
    again = _ok(client.post("/counterfactual", json={**REJECTED, "total_cfs": 3}))
    assert again == cf                                               # 난수 고정 -> 시연 때 결과가 바뀌지 않음


def test_approved_customer_returns_empty(client, main_mod):
    body = _ok(client.post("/counterfactual", json=APPROVED))
    assert body["original"]["approval_status"] == "APPROVED"
    assert body["counterfactuals"] == []
    assert "이미 승인" in body["message"]


def test_missing_income_freezes_income_and_debt_ratio(client, main_mod):
    req = {**REJECTED, "MonthlyIncome": None}
    body = _ok(client.post("/counterfactual", json=req))
    if body["original"]["approval_status"] == "APPROVED":
        pytest.skip("소득을 빼니 승인됨")
    # 입력하지 않은 소득은 제안하지 않고, 소득이 없으면 부채 비율은 모델에 반영되지 않으므로 제안하지 않음
    assert "MonthlyIncome" not in body["varied_features"]
    assert "DebtRatio" not in body["varied_features"]
    for o in body["counterfactuals"]:
        assert {ch["feature"] for ch in o["changes"]}.isdisjoint({"MonthlyIncome", "DebtRatio"})
        # 소득 미입력 그대로 다시 채점해도 승인 (미입력 처리가 /predict 와 같은지)
        assert _ok(client.post("/predict", json=_apply(req, o)))["approval_status"] == "APPROVED"


def test_total_cfs_limits(client, main_mod):
    body = _ok(client.post("/counterfactual", json={**REJECTED, "total_cfs": 1}))
    assert len(body["counterfactuals"]) <= 1


@pytest.mark.parametrize("extra", [{"total_cfs": 0}, {"total_cfs": 6}, {"total_cfx": 3}],
                         ids=["total_cfs=0", "total_cfs=6", "typo_field"])
def test_invalid_request_returns_422(client, extra):
    assert client.post("/counterfactual", json={**REJECTED, **extra}).status_code == 422


def test_returns_503_when_not_ready(client, main_mod, monkeypatch):
    monkeypatch.setitem(main_mod.art, "cf_data", None)               # 테스트가 끝나면 원래대로 돌아감
    assert client.post("/counterfactual", json=REJECTED).status_code == 503