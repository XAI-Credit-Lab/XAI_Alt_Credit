"""tests/test_feature_spec.py - 단위 사전·거절 사유 문장 테스트"""
import math

import pytest

from src.data.feature_spec import (
    FEATURE_SPEC, build_rejection_reasons, field_bounds, format_value,
    input_label, is_missing, josa, reason_sentence,
)

REQUIRED_KEYS = {"label", "unit", "kind", "min", "max", "step", "fmt", "help"}


def test_all_features_have_required_keys():
    for name, spec in FEATURE_SPEC.items():
        assert REQUIRED_KEYS <= spec.keys(), name


def test_min_not_greater_than_max():
    for name, s in FEATURE_SPEC.items():
        if s["min"] is not None and s["max"] is not None:
            assert s["min"] <= s["max"], name


def test_field_bounds_matches_spec():
    assert field_bounds("DebtRatio") == {"ge": 0.0, "le": 10.0}
    assert field_bounds("MonthlyIncome") == {"ge": 0.0}          # max=None 은 제외


def test_input_label_shows_unit():
    assert input_label("DebtRatio") == "부채 비율 (비율, 예: 0.3)"


@pytest.mark.parametrize("v", [None, float("nan"), math.nan])
def test_missing_values(v):
    assert is_missing(v)
    assert format_value("DebtRatio", v) == "미입력"


def test_numpy_nan_is_missing():
    np = pytest.importorskip("numpy")
    assert format_value("DebtRatio", np.float32("nan")) == "미입력"


@pytest.mark.parametrize("word,pair,expected", [
    ("98%", "으로/로", "98%로"), ("3", "으로/로", "3으로"), ("6개월", "으로/로", "6개월로"),
    ("부채 비율", "이/가", "부채 비율이"), ("소비 일관성", "이/가", "소비 일관성이"),
])
def test_josa(word, pair, expected):
    assert josa(word, pair) == expected


def test_ratio_formatted_as_percent():
    assert reason_sentence("DebtRatio", 0.98, 0.366, 0.454) == \
        "부채 비율이 98%로, 중앙값(37%)보다 높습니다 (기여도: +0.454)"


def test_count_with_zero_median():
    assert "대부분의 고객(0회)보다 많습니다" in reason_sentence("NumberOfTimes90DaysLate", 2, 0, 0.3)


def test_period_uses_short_word():
    assert "짧습니다" in reason_sentence("card_history_months", 6, 193, 0.1)


def test_no_median_sentence_is_complete():
    assert reason_sentence("MonthlyIncome", 2308.0, None, 0.2) == "월 소득이 2,308달러입니다 (기여도: +0.200)"


def test_rejection_reasons_only_positive_and_sorted():
    out = build_rejection_reasons(
        {"DebtRatio": 0.98, "telecom_payment_rate": 0.78, "MonthlyIncome": 6000},
        {"DebtRatio": 0.2, "telecom_payment_rate": 0.4, "MonthlyIncome": -0.1, "unknown_flag": 0.9},
        {"DebtRatio": 0.366, "telecom_payment_rate": 0.92, "MonthlyIncome": 5400},
    )
    feats = [r["feature"] for r in out["rejection_reasons"]]
    assert feats == ["telecom_payment_rate", "DebtRatio"]
    assert out["total_rejection_factors"] == 2
    assert set(out["rejection_reasons"][0]) == {
        "rank", "feature", "feature_name_kr", "current_value",
        "reference_value", "shap_contribution", "explanation"}


def test_ratio_max_matches_preprocessor():
    """API 입력 상한과 전처리 결측 기준이 같아야 함 (둘 중 하나만 바뀌는 것 방지)"""
    from src.data.preprocessor import DEBT_RATIO_MAX, REVOLVING_MAX
    assert FEATURE_SPEC["DebtRatio"]["max"] == DEBT_RATIO_MAX
    assert FEATURE_SPEC["RevolvingUtilizationOfUnsecuredLines"]["max"] == REVOLVING_MAX