"""
tests/test_fairness.py
공정성 모듈 테스트

[단위 테스트]  손으로 계산한 정답이 있는 작은 데이터 (파일 불필요)
[통합 테스트]  저장된 모델·분할 데이터가 있을 때만 실행, 독립적으로 다시 계산해 비교
실행: 프로젝트 최상위 폴더에서  pytest tests/test_fairness.py -v
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fairness import fairness as F  # noqa: E402


# ----------------------------------------------------------------------
# 단위 테스트 (정답을 손으로 계산)
# ----------------------------------------------------------------------
def _sens(gender, age_group):
    return pd.DataFrame({"gender": gender, "age_group": age_group})


def test_gender_di_by_hand():
    # 여성 승인율 0.5 (2명 중 1명), 남성 1.0 -> DI 0.5
    s = _sens([1, 1, 0, 0], ["35~54"] * 4)
    assert F.gender_di(s, [1, 0, 1, 1]) == pytest.approx(0.5)


@pytest.mark.parametrize("value, expected", [
    (0.79, "WARNING"), (0.80, "OK"), (1.00, "OK"), (1.25, "OK"), (1.26, "WARNING"),
    (None, "UNKNOWN"), (float("nan"), "UNKNOWN"),
])
def test_di_status_boundaries(value, expected):
    assert F.di_status(value) == expected


def _gender_ok_age_violation():
    """성별 DI = 1.0 (정상), ~34/35~54 = 0.5 (위반), 55+/35~54 = 1.0 (정상)"""
    age = (["~34"] * 4) + (["35~54"] * 4) + (["55+"] * 4)
    gender = [1, 1, 0, 0] * 3
    approved = [1, 0, 1, 0] + [1] * 4 + [1] * 4
    return _sens(gender, age), np.array(approved)


def test_age_violation_warns_even_when_gender_is_ok(caplog):
    sens, approved = _gender_ok_age_violation()
    di = F.compute_di(sens, approved)
    assert di["gender"] == pytest.approx(1.0)
    assert di["age"]["~34"] == pytest.approx(0.5)

    with caplog.at_level(logging.WARNING, logger=F.logger.name):
        rows = F.check_di_warnings(di, "테스트")
    status = {(r["attribute"], r["group"]): r["status"] for r in rows}
    assert status[("gender", "여성/남성")] == "OK"
    assert status[("age_group", "~34/35~54")] == "WARNING"
    assert status[("age_group", "55+/35~54")] == "OK"
    # 로그도 위반한 속성에 대해서만 1건
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "age_group" in msgs[0]


def test_gender_violation_warns_even_when_age_is_ok(caplog):
    # 모든 연령대 승인율 동일(DI 1.0), 성별은 여성 0 / 남성 1 이라 DI 0 -> 성별만 위반
    sens = _sens([1, 0] * 6, ["~34", "~34", "35~54", "35~54", "55+", "55+"] * 2)
    approved = np.array([0, 1] * 6)
    with caplog.at_level(logging.WARNING, logger=F.logger.name):
        rows = F.check_di_warnings(F.compute_di(sens, approved))
    bad = [(r["attribute"]) for r in rows if r["status"] == "WARNING"]
    assert bad == ["gender"]


def test_equalized_odds_by_hand():
    # 여성: 정상 2명 중 1명 승인(TPR .5), 부도 2명 중 1명 승인(FPR .5)
    # 남성: 정상 2명 모두 승인(TPR 1), 부도 2명 모두 거절(FPR 0)
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    a = [1, 0, 1, 0, 1, 1, 0, 0]
    g = ["여성"] * 4 + ["남성"] * 4
    eo = F.equalized_odds(y, a, g)
    assert eo["tpr_diff"] == pytest.approx(0.5)
    assert eo["fpr_diff"] == pytest.approx(0.5)
    t = eo["table"].set_index("group")
    assert t.loc["여성", "tpr"] == pytest.approx(0.5) and t.loc["남성", "tpr"] == pytest.approx(1.0)
    assert t.loc["여성", "fpr"] == pytest.approx(0.5) and t.loc["남성", "fpr"] == pytest.approx(0.0)


def test_small_default_groups_are_reported():
    y = [0, 1] * 5
    eo = F.equalized_odds(y, [1, 0] * 5, ["A", "B"] * 5)
    assert set(eo["small_groups"]) == {"A", "B"}   # 부도 건수가 100건 미만


def test_approve_rule_is_less_or_equal():
    assert F.approve([0.1, 0.2, 0.3], 0.2).tolist() == [1, 1, 0]


def test_group_cutoffs_equalize_group_approval_rates():
    rng = np.random.default_rng(0)       # 테스트용 난수 (재현 가능)
    n = 4000
    sens = _sens(rng.integers(0, 2, n), ["35~54"] * n)
    # 여성(1)의 부도확률이 전반적으로 더 높게 나오는 모델 (편향된 모델을 흉내)
    p = np.where(sens["gender"] == 1, rng.beta(3, 8, n), rng.beta(2, 8, n))
    target = 0.7
    cuts = F.fit_group_cutoffs(p, sens, target, by=["gender"])
    before = F.approve(p, np.quantile(p, target))
    after = F.apply_group_cutoffs(p, sens, cuts, np.quantile(p, target), by=["gender"])
    assert F.gender_di(sens, before) < 0.9
    assert F.gender_di(sens, after) == pytest.approx(1.0, abs=0.01)
    assert after.mean() == pytest.approx(target, abs=0.01)


def test_unseen_group_falls_back_to_global_cutoff():
    sens_val = _sens([0, 0, 1, 1], ["35~54"] * 4)
    cuts = F.fit_group_cutoffs([0.1, 0.2, 0.3, 0.4], sens_val, 0.5, by=["gender"])
    sens_new = _sens([2, 2], ["55+", "55+"])     # Validation에 없던 집단(gender=2)
    out = F.apply_group_cutoffs([0.05, 0.9], sens_new, cuts, fallback=0.5, by=["gender"])
    assert out.tolist() == [1, 0]                 # 처음 보는 집단은 fallback(0.5)으로 판정


# ----------------------------------------------------------------------
# 통합 테스트 (저장된 모델·데이터가 있을 때만)
# ----------------------------------------------------------------------
NEEDED = [F.DATA_DIR / f"{k}_{s}.csv" for k in ("X", "y", "sensitive") for s in ("val", "test")]
HAS_ARTIFACTS = bool(list(F.MODEL_DIR.glob("model_meta_v*.json"))) and all(p.exists() for p in NEEDED)
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="models/ 또는 data/processed/ 분할 파일이 없음")


@pytest.fixture(scope="module")
def result():
    return F.run_fairness()


@needs_artifacts
def test_independent_recomputation_matches_module(result):
    """모듈을 거치지 않고 파일에서 직접 다시 계산한 값과 일치하는지"""
    import joblib
    meta = json.loads(sorted(F.MODEL_DIR.glob("model_meta_v*.json"))[-1].read_text(encoding="utf-8"))
    best = meta["best_model"]
    model = joblib.load(F.MODEL_DIR / best["file"])
    X = pd.read_csv(F.DATA_DIR / "X_test.csv", index_col=0)
    y = pd.read_csv(F.DATA_DIR / "y_test.csv", index_col=0).iloc[:, 0].to_numpy()
    s = pd.read_csv(F.DATA_DIR / "sensitive_test.csv", index_col=0)

    appr = model.predict_proba(X)[:, 1] <= best["approval_cutoff"]
    fem, mal = s["gender"].to_numpy() == 1, s["gender"].to_numpy() == 0
    di = appr[fem].mean() / appr[mal].mean()
    tpr = lambda m: appr[m & (y == 0)].mean()    # noqa: E731
    fpr = lambda m: appr[m & (y == 1)].mean()    # noqa: E731

    nm = result["non_miti"]
    assert nm["di_ratio"] == pytest.approx(di)
    assert nm["equalized_odds_tpr_diff"] == pytest.approx(abs(tpr(fem) - tpr(mal)))
    assert nm["equalized_odds_fpr_diff"] == pytest.approx(abs(fpr(fem) - fpr(mal)))
    assert nm["approval_rate"] == pytest.approx(appr.mean())
    # model.py가 meta에 저장한 Test 승인율과도 같아야 함
    assert nm["approval_rate"] == pytest.approx(best["test_approval_rate"])


@needs_artifacts
def test_api_keys_present_and_finite(result):
    for part in ("non_miti", "miti"):
        for k in ("di_ratio", "equalized_odds_tpr_diff", "equalized_odds_fpr_diff"):
            assert result[part][k] is not None and np.isfinite(result[part][k]), (part, k)


@needs_artifacts
def test_model_fairness_returns_non_miti_then_miti(result):
    non_miti, miti = F.model_fairness()
    assert non_miti["di_ratio"] == pytest.approx(result["non_miti"]["di_ratio"])
    assert miti["di_ratio"] == pytest.approx(result["miti"]["di_ratio"])


@needs_artifacts
def test_mitigation_does_not_move_dis_away_from_one_when_there_was_a_violation(result):
    nm, mi = result["non_miti"], result["miti"]
    if nm["overall_status"] == "OK":
        pytest.skip("완화 전에 위반이 없는 모델 (통제 데이터 모델 등)")
    assert abs(1 - mi["di_ratio"]) < abs(1 - nm["di_ratio"])
    for grp, v in nm["age_di"].items():
        if v < F.DI_LOWER or v > F.DI_UPPER:
            assert abs(1 - mi["age_di"][grp]) < abs(1 - v)


@needs_artifacts
def test_cutoffs_come_from_validation_only(result):
    """집단별 기준값 = Validation 부도확률의 집단별 분위수 (Test는 쓰지 않음)"""
    import joblib
    info = result["info"]
    model = joblib.load(F.MODEL_DIR / info["model_file"])
    X = pd.read_csv(F.DATA_DIR / "X_val.csv", index_col=0)
    s = pd.read_csv(F.DATA_DIR / "sensitive_val.csv", index_col=0)
    p = model.predict_proba(X)[:, 1]
    keys = F._group_keys(s, info["mitigation_by"]).to_numpy()
    for k, c in info["group_cutoffs"].items():
        assert c == pytest.approx(np.quantile(p[keys == k], info["target_approval_rate"]))


@needs_artifacts
def test_detail_is_strict_json_serializable():
    """/fairness/detail 은 NaN 이 있으면 FastAPI 가 500 을 낸다 -> NaN 없이 엄격한 JSON 이어야 함"""
    d = F.model_fairness_detail()
    json.dumps(d, allow_nan=False)
    assert set(d) == {"thresholds", "info", "non_miti", "miti"}
    assert d["thresholds"]["di_lower"] == F.DI_LOWER and d["thresholds"]["eo_threshold"] == F.EO_THRESHOLD


def test_json_safe_converts_nan_and_numpy():
    out = F._json_safe({"a": float("nan"), "b": np.float64(0.5), "c": [np.int64(3), float("inf")]})
    assert out == {"a": None, "b": 0.5, "c": [3, None]}


@needs_artifacts
def test_comparison_table_for_report(result):
    t = F.comparison_table(result)
    assert list(t.columns) == ["지표", "완화 전", "완화 후", "기준"]
    assert "성별 DI (여/남)" in t["지표"].tolist() and "전체 승인율" in t["지표"].tolist()
    row = t.set_index("지표").loc["성별 DI (여/남)"]
    assert row["완화 전"] == pytest.approx(result["non_miti"]["di_ratio"])
    assert row["완화 후"] == pytest.approx(result["miti"]["di_ratio"])
