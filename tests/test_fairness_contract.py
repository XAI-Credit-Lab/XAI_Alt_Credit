"""
공정성 모듈 계약 테스트: simulator.py 가 기대하는 5개 이름이 유지되는지 확인
공정성 모듈을 교체한 뒤 python -m pytest tests/test_fairness_contract.py 로 실행
"""
import pandas as pd

from src.fairness.fairness import AGE_BINS, AGE_LABELS, age_di, fmt_age_di, gender_di


def test_age_labels_are_three_groups():
    assert len(AGE_LABELS) == 3


def test_age_bins_cover_gmsc_age_range():
    ages = pd.Series([21, 34, 35, 54, 55, 109])   # GMSC 실제 나이 범위의 경계값
    groups = pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS)
    assert groups.notna().all()


def test_di_functions_return_expected_types():
    young, mid, old = AGE_LABELS
    df = pd.DataFrame({"gender": [0, 0, 1, 1], "age_group": [young, mid, old, mid]})
    approved = pd.Series([1, 1, 1, 0])

    assert isinstance(gender_di(df, approved), float)
    result = age_di(df, approved)
    assert young in result and old in result
    assert isinstance(fmt_age_di(result), str)


def test_simulator_bias_keys_match_age_labels():
    from src.data.simulator import BIAS_PENALTY, FEATURE_BIAS

    allowed = {"female", *AGE_LABELS}
    assert set(BIAS_PENALTY) <= allowed
    for feature, groups in FEATURE_BIAS.items():
        assert set(groups) <= allowed, f"{feature}에 알 수 없는 집단 키: {set(groups) - allowed}"