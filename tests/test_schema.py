"""
tests/test_schema.py
API 입력·응답 스키마 테스트 (서버를 띄우지 않고 pydantic 검증만 확인)

확인하는 것
  1) 정상 입력은 통과하고, 범위를 벗어난 값·오타 필드는 거부 (422의 원인)
  2) 입력 범위가 feature_spec.py, 전처리 기준과 같음
  3) 연체 컬럼은 alias(원래 컬럼명)와 변수명 둘 다 받음
  4) 거절 사유 응답의 선택 필드(rank, reference_value 등)는 없어도 오류가 나지 않음,
     /metrics 의 추가 지표(씬 파일러 AUC 향상·PSI)도 없으면 null
  5) 반사실적 설명 요청·응답 스키마

실행: 프로젝트 최상위 폴더에서  pytest tests/test_schema.py -v
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import schema  # noqa: E402
from src.data.feature_spec import FEATURE_SPEC  # noqa: E402
from src.data.preprocessor import DEBT_RATIO_MAX, REVOLVING_MAX  # noqa: E402

VALID = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "DebtRatio": 0.3,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfDependents": 1,
    "card_history_months": 60,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0,
    "telecom_payment_rate": 0.95,
    "utility_payment_rate": 0.95,
    "spending_consistency": 60.0,
    "regular_payment_count": 4,
    "app_login_frequency": 20,
}
OPTIONAL = ["MonthlyIncome", "NumberOfDependents", "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]


def make(**changes):
    return schema.predict_request(**{**VALID, **changes})


# ----------------------------------------------------------------------
# 1) 정상·비정상 입력
# ----------------------------------------------------------------------
def test_valid_input_passes():
    req = make()
    assert req.DebtRatio == 0.3


def test_optional_fields_can_be_omitted():
    data = {k: v for k, v in VALID.items() if k not in OPTIONAL}
    req = schema.predict_request(**data)
    dumped = req.model_dump(by_alias=True)
    assert all(dumped[k] is None for k in OPTIONAL)


@pytest.mark.parametrize("field, value", [
    ("DebtRatio", DEBT_RATIO_MAX + 0.01),              # 전처리에서 결측 처리되는 값은 입력 단계에서 거부
    ("RevolvingUtilizationOfUnsecuredLines", REVOLVING_MAX + 0.01),
    ("DebtRatio", -0.1),
    ("telecom_payment_rate", 1.01),
    ("spending_consistency", 101),
    ("card_history_months", 361),
    ("NumberOfTime30-59DaysPastDueNotWorse", 25),
    ("MonthlyIncome", -1),
])
def test_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


@pytest.mark.parametrize("field, value", [
    ("DebtRatio", DEBT_RATIO_MAX), ("RevolvingUtilizationOfUnsecuredLines", REVOLVING_MAX),
    ("telecom_payment_rate", 1.0), ("card_history_months", 360),
])
def test_boundary_values_accepted(field, value):
    make(**{field: value})


def test_unknown_field_rejected():
    """extra='forbid': 오타 필드는 조용히 무시되지 않고 에러 (예: MonthlyIncom)"""
    with pytest.raises(ValidationError):
        schema.predict_request(**{**VALID, "MonthlyIncom": 2500})


def test_required_field_missing_rejected():
    data = dict(VALID)
    data.pop("telecom_payment_rate")
    with pytest.raises(ValidationError):
        schema.predict_request(**data)


# ----------------------------------------------------------------------
# 2) 입력 범위 = feature_spec = 전처리 기준
# ----------------------------------------------------------------------
def _bounds(field_name):
    """pydantic 필드에 걸린 ge / le 값"""
    info = schema.predict_request.model_fields[field_name]
    ge = next((m.ge for m in info.metadata if getattr(m, "ge", None) is not None), None)
    le = next((m.le for m in info.metadata if getattr(m, "le", None) is not None), None)
    return ge, le


def test_schema_bounds_match_feature_spec():
    for name, f in schema.predict_request.model_fields.items():
        col = f.alias or name
        if col not in FEATURE_SPEC:
            continue
        ge, le = _bounds(name)
        assert ge == FEATURE_SPEC[col]["min"], col
        assert le == FEATURE_SPEC[col]["max"], col


def test_schema_covers_every_feature_spec_input():
    cols = {f.alias or n for n, f in schema.predict_request.model_fields.items()}
    assert set(FEATURE_SPEC) <= cols, set(FEATURE_SPEC) - cols


# ----------------------------------------------------------------------
# 3) alias
# ----------------------------------------------------------------------
def test_past_due_accepts_alias_and_field_name():
    by_alias = make(**{"NumberOfTimes90DaysLate": 2})
    data = {k: v for k, v in VALID.items() if k != "NumberOfTimes90DaysLate"}
    by_name = schema.predict_request(**data, past_due_90=2)
    assert by_alias.past_due_90 == by_name.past_due_90 == 2


def test_dump_uses_original_column_names():
    """전처리기는 원래 컬럼명(alias)으로 받으므로 by_alias=True 결과에 원래 이름이 있어야 함"""
    dumped = make().model_dump(by_alias=True)
    assert "NumberOfTime30-59DaysPastDueNotWorse" in dumped and "past_due_30_59" not in dumped


# ----------------------------------------------------------------------
# 4) 응답 스키마
# ----------------------------------------------------------------------
def test_explain_response_accepts_feature_spec_format():
    reason = {"rank": 1, "feature": "DebtRatio", "feature_name_kr": "부채 비율", "feature_label": "부채 비율",
              "current_value": 0.98, "reference_value": 0.29, "shap_contribution": 0.45,
              "explanation": "부채 비율이 98%로, 중앙값(29%)보다 높습니다 (기여도: +0.450)"}
    resp = schema.explain_response(approval_status="REJECTED", rejection_reasons=[reason],
                                   total_rejection_factors=3, shap_values={"DebtRatio": 0.45})
    assert resp.rejection_reasons[0].rank == 1


def test_reason_optional_fields_default_to_none():
    r = schema.reason(feature="DebtRatio", shap_contribution=0.1, explanation="설명")
    assert r.rank is None and r.reference_value is None and r.feature_name_kr is None


def test_explain_response_without_total_is_ok():
    schema.explain_response(approval_status="APPROVED", rejection_reasons=[], shap_values={})


METRICS = {"auc": 0.88, "ks": 0.61, "precision": 0.53, "recall": 0.50, "f1": 0.52}


def test_metrics_response_new_fields_optional():
    """이전 버전 메타에는 씬 파일러 AUC 향상·PSI가 없음 -> 없어도 오류 없이 null"""
    resp = schema.metrics_response(**METRICS)
    assert resp.thin_filer_auc_uplift is None and resp.psi is None


def test_metrics_response_with_new_fields():
    resp = schema.metrics_response(**METRICS, thin_filer_auc_uplift=0.1134, psi=0.0005)
    assert resp.thin_filer_auc_uplift == pytest.approx(0.1134)
    assert resp.psi == pytest.approx(0.0005)


# ----------------------------------------------------------------------
# 5) 반사실적 설명 스키마
# ----------------------------------------------------------------------
def test_counterfactual_request_default_total():
    req = schema.counterfactual_request(**VALID)
    assert req.total_cfs == 3


@pytest.mark.parametrize("total", [0, 6])
def test_counterfactual_request_total_out_of_range(total):
    with pytest.raises(ValidationError):
        schema.counterfactual_request(**VALID, total_cfs=total)


def test_counterfactual_request_inherits_input_rules():
    # /predict 와 같은 입력 검사 (범위, 오타 거부, alias)
    with pytest.raises(ValidationError):
        schema.counterfactual_request(**{**VALID, "DebtRatio": DEBT_RATIO_MAX + 0.01})
    with pytest.raises(ValidationError):
        schema.counterfactual_request(**VALID, MonthlyIncom=3000)
    dumped = schema.counterfactual_request(**VALID).model_dump(by_alias=True, exclude={"total_cfs"})
    assert "NumberOfTimes90DaysLate" in dumped and "total_cfs" not in dumped


CF_CHANGE = {"feature": "telecom_payment_rate", "feature_name_kr": "통신비 정상납부율",
             "current_value": 0.32, "suggested_value": 0.98,
             "current_display": "32%", "suggested_display": "98%"}
CF_OPTION = {"rank": 1, "changes": [CF_CHANGE], "credit_score": 692, "approval_status": "APPROVED",
             "default_probability": 0.3078, "risk_grade": "C"}
ORIGINAL = {"credit_score": 139, "approval_status": "REJECTED", "default_probability": 0.8605, "risk_grade": "E"}


def test_counterfactual_response_parses_example():
    resp = schema.counterfactual_response(
        original=ORIGINAL, counterfactuals=[CF_OPTION],
        varied_features=["telecom_payment_rate"], fixed_features=["card_history_months"], message="안내")
    assert resp.counterfactuals[0].changes[0].suggested_display == "98%"
    assert resp.original.approval_status == "REJECTED"


def test_counterfactual_empty_list_is_ok():
    schema.counterfactual_response(original=ORIGINAL, counterfactuals=[], varied_features=[],
                                   fixed_features=[], message="이미 승인된 고객입니다.")


@pytest.mark.parametrize("missing", ["rank", "changes", "approval_status"])
def test_cf_option_required_fields(missing):
    # rank 가 빠지면 API 가 500 을 내므로 스키마에서 필수인지 확인
    with pytest.raises(ValidationError):
        schema.cf_option(**{k: v for k, v in CF_OPTION.items() if k != missing})