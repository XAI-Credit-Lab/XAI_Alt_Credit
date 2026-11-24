"""
tests/test_xai.py
SHAP 설명 모듈(ModelExplainer) 테스트 (작은 가짜 모델로 몇 초 안에 끝남, 저장된 모델 불필요)

확인하는 것
  1) TreeExplainer 사용 + 가법성: 기준값 + SHAP 합 = 모델 출력(log-odds)
  2) 거절 사유: 기여도 + 인 변수만, 큰 순서, 제외 변수(결측 지시변수 등) 없음, 같은 입력이면 같은 결과
  3) 승인 고객 최소 기여도(min_contribution) 적용
  4) 문장: 미입력은 '미입력' 문장, 비교 기준은 Train 중앙값, 원본 입력이 없으면 원래 단위로 복원
  5) 그래프 함수가 PNG(base64)를 돌려줌
  6) 그래프 표시: 마이너스 기호(□ 방지), 한글 변수 이름, dependence 의 원래 단위 축, dict 키는 영어

실행: 프로젝트 최상위 폴더에서  pytest tests/test_xai.py -v
"""
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("shap")
lightgbm = pytest.importorskip("lightgbm")

import matplotlib  # noqa: E402
matplotlib.use("Agg")                                  # 화면 없이 그림 그리기

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402

from src.data.feature_spec import FEATURE_SPEC, label  # noqa: E402
from src.explainer import xai as xai_mod  # noqa: E402
from src.explainer.xai import ModelExplainer  # noqa: E402

REV, DR, CARD = "RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "card_history_months"
MISS, FLAG = "missingindicator_DebtRatio", "past_due_code_flag"
COLS = [REV, DR, CARD, MISS, FLAG]
EXCLUDE = ("is_thin_filer", "missingindicator_", "past_due_code_flag")   # main.py 와 같은 제외 목록

# 원래 단위 복원용 (StandardScaler 평균·표준편차를 흉내), 비교 기준 중앙값
RAW_MEANS = {REV: 0.32, DR: 0.37, CARD: 180.0, MISS: 0.2, FLAG: 0.0}
RAW_SCALES = {REV: 0.30, DR: 0.50, CARD: 100.0, MISS: 0.4, FLAG: 1.0}
MEDIANS = {REV: 0.15, DR: 0.29, CARD: 193.0}

# 모든 변수가 부도 위험을 높이는 방향인 고객 (표준화된 값) + 사용자가 입력한 원래 값
RISKY = {REV: 2.0, DR: 2.0, CARD: -2.0, MISS: 1.0, FLAG: 1.0}
RISKY_RAW = {REV: 0.92, DR: 1.37, CARD: 6}


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({
        REV: rng.normal(size=n), DR: rng.normal(size=n), CARD: rng.normal(size=n),
        MISS: (rng.random(n) < 0.2).astype(float), FLAG: (rng.random(n) < 0.02).astype(float),
    })[COLS]
    logit = 1.2 * X[REV] + 1.5 * X[DR] - 0.8 * X[CARD] + 1.0 * X[MISS] + 2.0 * X[FLAG] - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = lightgbm.LGBMClassifier(n_estimators=60, learning_rate=0.1, num_leaves=15,
                                    random_state=0, verbose=-1).fit(X, y)
    expl = ModelExplainer(model, X, predict_fn=model.predict_proba, class_index=1,
                          raw_means=RAW_MEANS, raw_scales=RAW_SCALES, raw_medians=MEDIANS,
                          exclude_from_reasons=EXCLUDE, global_sample_size=300)
    return model, X, expl


def _is_png(b64: str) -> bool:
    return base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------
# 1) Explainer 종류·가법성
# ----------------------------------------------------------------------
def test_uses_tree_explainer(setup):
    _, _, expl = setup
    assert expl.kind == "tree"            # Kernel 로 넘어가면 실행마다 거절 사유가 달라짐


def test_shap_values_add_up_to_model_output(setup):
    """기준값 + SHAP 합 = 모델의 log-odds (Waterfall 이 맞게 그려지는 근거)"""
    model, X, expl = setup
    rows = X.iloc[:50]
    sv, base = expl._shap_for(rows)
    p = np.clip(model.predict_proba(rows)[:, 1], 1e-12, 1 - 1e-12)
    np.testing.assert_allclose(sv.sum(axis=1) + base, np.log(p / (1 - p)), atol=1e-4)


# ----------------------------------------------------------------------
# 2) 거절 사유 선정
# ----------------------------------------------------------------------
def test_reasons_positive_sorted_and_ranked(setup):
    _, _, expl = setup
    res = expl.explain(RISKY, top_k=5, raw_values=RISKY_RAW)
    reasons = res["rejection_reasons"]
    contrib = [r["shap_contribution"] for r in reasons]
    assert reasons, "위험한 고객인데 거절 사유가 없음"
    assert all(c > 0 for c in contrib)
    assert contrib == sorted(contrib, reverse=True)
    assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
    assert len(reasons) <= 5 and res["total_rejection_factors"] >= len(reasons)


def test_excluded_variables_never_become_reasons(setup):
    """결측 지시변수·기록 이상 코드는 SHAP 이 + 여도 거절 사유에 나오면 안 됨"""
    _, _, expl = setup
    res = expl.explain(RISKY, top_k=5, raw_values=RISKY_RAW)
    names = [r["feature"] for r in res["rejection_reasons"]]
    assert not any(n.startswith(EXCLUDE) for n in names)
    assert all(n in FEATURE_SPEC for n in names)
    # 전체 SHAP 값(대시보드 막대그래프용)에는 모든 변수가 있어야 함
    assert set(res["shap_values"]) == set(COLS)


def test_same_input_same_reasons(setup):
    _, _, expl = setup
    a = expl.explain(RISKY, raw_values=RISKY_RAW)
    b = expl.explain(RISKY, raw_values=RISKY_RAW)
    assert a["rejection_reasons"] == b["rejection_reasons"]


def test_label_fields_are_filled(setup):
    _, _, expl = setup
    for r in expl.explain(RISKY, raw_values=RISKY_RAW)["rejection_reasons"]:
        assert r["feature_name_kr"] == FEATURE_SPEC[r["feature"]]["label"]
        assert r["feature_label"] == r["feature_name_kr"]          # 이전 필드 이름 호환


# ----------------------------------------------------------------------
# 3) 승인 고객 최소 기여도
# ----------------------------------------------------------------------
def test_min_contribution_filters_small_factors(setup):
    _, _, expl = setup
    res = expl.explain(RISKY, raw_values=RISKY_RAW, use_min_contribution=True)
    assert all(r["shap_contribution"] > expl.min_contribution for r in res["rejection_reasons"])


def test_huge_min_contribution_gives_no_reasons(setup):
    model, X, _ = setup
    strict = ModelExplainer(model, X, predict_fn=model.predict_proba, raw_medians=MEDIANS,
                            exclude_from_reasons=EXCLUDE, min_contribution=100.0, global_sample_size=100)
    res = strict.explain(RISKY, raw_values=RISKY_RAW, use_min_contribution=True)
    assert res["rejection_reasons"] == [] and res["total_rejection_factors"] == 0


# ----------------------------------------------------------------------
# 4) 문장 내용
# ----------------------------------------------------------------------
def test_reference_value_is_train_median(setup):
    _, _, expl = setup
    for r in expl.explain(RISKY, raw_values=RISKY_RAW)["rejection_reasons"]:
        assert r["reference_value"] == pytest.approx(MEDIANS[r["feature"]])
        assert "평균" not in r["explanation"]                       # 예전 '평균(308)' 문장이 다시 나오지 않도록


def test_missing_input_gives_missing_sentence(setup):
    _, _, expl = setup
    res = expl.explain(RISKY, raw_values={**RISKY_RAW, DR: None})
    dr = [r for r in res["rejection_reasons"] if r["feature"] == DR]
    assert dr, "부채 비율이 거절 사유에 없음 (가짜 모델에서는 위험 방향이어야 함)"
    assert dr[0]["current_value"] is None
    assert "미입력" in dr[0]["explanation"]


# ----------------------------------------------------------------------
# 5) 그래프
# ----------------------------------------------------------------------
def test_waterfall_returns_png(setup):
    _, X, expl = setup
    res = expl.explain(RISKY, raw_values=RISKY_RAW)
    png = expl.plot_waterfall(pd.DataFrame([RISKY])[COLS], display_values={**RISKY_RAW, DR: None},
                              shap_values=res["shap_values"], base_value=res["base_value"])
    assert _is_png(png)


def test_summary_and_dependence_return_png(setup):
    _, _, expl = setup
    assert _is_png(expl.plot_summary(max_display=5))
    dep = expl.plot_dependence()
    assert isinstance(dep, dict) and len(dep) == 2               # 대시보드가 dict 로 받음
    assert not any(k.startswith(EXCLUDE) for k in dep)           # 0/1 변수는 dependence 대상에서 제외
    assert all(_is_png(v) for v in dep.values())


def test_value_restored_to_original_units_when_input_absent(setup):
    """원본 입력(raw_values)에 없는 변수는 표준화 값 × 표준편차 + 평균 으로 복원해서 보여줌"""
    _, _, expl = setup
    for r in expl.explain(RISKY, raw_values={})["rejection_reasons"]:
        f = r["feature"]
        assert r["current_value"] == pytest.approx(RISKY[f] * RAW_SCALES[f] + RAW_MEANS[f])


# ----------------------------------------------------------------------
# 6) 그래프 표시 (마이너스 기호, 한글 이름, 원래 단위)
# ----------------------------------------------------------------------
@pytest.fixture
def captured(monkeypatch):
    """그래프가 PNG 로 저장되기 직전의 글자·축 정보를 기록 (그림은 그대로 저장)"""
    records = []
    original = xai_mod.ModelExplainer._to_base64

    def recording(fig, dpi=120):
        xai_mod._fix_minus(fig)                       # 실제 _to_base64 와 같은 순서로 교체한 뒤 기록
        axes = fig.axes
        records.append({
            "texts": [t.get_text() for t in fig.findobj(matplotlib.text.Text)],
            "xlabels": [ax.get_xlabel() for ax in axes],
            "ylabels": [ax.get_ylabel() for ax in axes],
            "x": (axes[0].collections[0].get_offsets()[:, 0].copy()
                  if axes and axes[0].collections else None),
        })
        return original(fig, dpi)

    monkeypatch.setattr(xai_mod.ModelExplainer, "_to_base64", staticmethod(recording))
    return records


def test_fix_minus_replaces_unicode_minus():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "\u22120.46")                  # SHAP 이 막대 숫자에 쓰는 유니코드 마이너스
    xai_mod._fix_minus(fig)
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    plt.close(fig)
    assert "-0.46" in texts
    assert not any("\u2212" in t for t in texts)


def test_waterfall_has_no_unicode_minus_and_korean_names(setup, captured):
    _, _, expl = setup
    res = expl.explain(RISKY, raw_values=RISKY_RAW)
    expl.plot_waterfall(pd.DataFrame([RISKY])[COLS], display_values=RISKY_RAW,
                        shap_values=res["shap_values"], base_value=res["base_value"])
    texts = captured[-1]["texts"]
    assert not any("\u2212" in t for t in texts)      # 나눔고딕에서 □ 로 보이던 글자
    assert any(label(REV) in t for t in texts)        # 한글 변수 이름
    assert any("%" in t for t in texts)               # 원본 값에 단위 (예: 92%)


def test_summary_uses_korean_names(setup, captured):
    _, _, expl = setup
    expl.plot_summary(max_display=5)
    texts = captured[-1]["texts"]
    assert any(label(DR) in t for t in texts)
    assert not any(t == DR for t in texts)            # 영어 원래 이름이 축에 남지 않음


def test_dependence_korean_label_and_original_units(setup, captured):
    _, _, expl = setup
    dep = expl.plot_dependence([REV])
    assert list(dep) == [REV]                         # dict 키는 영어 (API 응답 형식 유지)
    rec = captured[-1]
    assert rec["xlabels"][0] == label(REV)
    # x축 값 = 표준화 값 × 표준편차 + 평균 (원래 단위)
    expected = np.sort(expl._global_X[REV].to_numpy() * RAW_SCALES[REV] + RAW_MEANS[REV])
    assert np.allclose(np.sort(rec["x"]), expected)


def test_dependence_accepts_english_interaction_name(setup):
    _, _, expl = setup
    dep = expl.plot_dependence([REV], interaction_index=CARD)   # 한글 feature_names 와 함께 써도 에러 없음
    assert _is_png(dep[REV])


def test_to_original_units_does_not_change_input(setup):
    _, _, expl = setup
    before = expl._global_X.copy()
    out = expl._to_original_units(expl._global_X)
    pd.testing.assert_frame_equal(expl._global_X, before)       # SHAP 계산용 데이터는 그대로
    assert np.allclose(out[CARD], before[CARD] * RAW_SCALES[CARD] + RAW_MEANS[CARD])


def test_to_original_units_skipped_without_raw_means(setup):
    model, X, _ = setup
    plain = ModelExplainer(model, X, predict_fn=model.predict_proba, global_sample_size=50)
    out = plain._to_original_units(X.head())
    pd.testing.assert_frame_equal(out, X.head())               # 원본 평균이 없으면 되돌리지 않음


def test_kr_names_follow_column_order(setup):
    _, _, expl = setup
    assert expl._kr_names() == [label(c) for c in COLS]