"""
src/statistics/anova.py

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

One-way ANOVA 기반 모델 성능 비교 모듈.

  검증 1 (test_1_feature_group): 피처 그룹별 모델(전통 Only / 대안 Only / 통합) AUC 비교
                                → 대안 데이터 효용성 검증
  검증 2 (test_2_algorithm)    : 알고리즘별(LR / XGB / LGBM) AUC 비교
                                → 최적 알고리즘 선정 근거

각 검증 결과에는 F-통계량, p-value, η²(효과 크기)가 포함되며,
p < alpha 인 경우 사후 검정(Tukey HSD) 결과가 포함된다.

관측치(표본)는 Repeated Stratified K-Fold 의 fold별 AUC 이다.
※ 학습 데이터(Train split)만 사용해야 한다. 테스트 라벨을 여기에 넣지 말 것.

모델 연결 (src/models/model.py 와 연동):
  - 비교 대상 모델은 model.py 가 저장한 최종 모델 파일을 불러와 사용한다.
      models/logistic_regression_v1.0.pkl, xgboost_v1.0.pkl, lightgbm_v1.0.pkl
  - 검증 1 에 쓸 모델은 models/model_meta_v1.0.json 의 best_model 이다.
  - 불러온 모델은 이미 Train 전체로 학습되어 있으므로 그대로 예측하면 데이터 누수가 생긴다.
    → clone() 으로 "하이퍼파라미터·불균형 설정만 복사한 빈 모델"을 만들어 fold 마다 새로 학습한다.
  - 데이터도 model.py 와 같은 CreditPreprocessor 로 전처리한 X_train 을 사용한다.

실행: 프로젝트 최상위 폴더에서  python -m src.statistics.anova
      (먼저 python -m src.models.model 로 모델 파일이 만들어져 있어야 함)

전체 흐름:
  데이터 → 모델별 fold AUC 수집 → ANOVA(F, p, η²) → 유의하면 Tukey HSD
  → JSON 저장 → FastAPI /anova 와 대시보드에서 읽기

"""
# ↑ 파일 맨 위 큰따옴표 세 개는 "모듈 docstring".
#   실행에는 영향이 없고, 이 파일이 무엇을 하는지 적어둔 문서다. help(anova) 로 볼 수 있다.

# =========================================================================== #
# import
#   규칙: 코드에서 실제로 쓰는 것만 import 하면 되고, 쓰는 건 반드시 import 해야 한다.
#   (쓰는데 빠지면 NameError, 안 쓰는데 넣으면 에러는 없지만 불필요한 코드)
#   순서(표준 라이브러리 → 외부 라이브러리 → 내 모듈, 사이에 빈 줄)는 PEP 8 관례일 뿐
#   동작에는 영향이 없다. 다만 코드 품질 측면에서 지키는 편이 좋다.
# =========================================================================== #

# ---- 표준 라이브러리 (파이썬 기본 내장, 설치 불필요) ----
import argparse   # 터미널에서 --mlflow 옵션을 받기 위함
import json       # 결과 dict ↔ JSON 파일 변환. API 응답도 JSON이라 가장 자연스러운 저장 방식
import logging    # print 대신 로그 출력 (INFO, WARNING 등 레벨 구분)
import math       # NaN, 무한대(inf) 체크용
import os         # 환경 변수 읽기 (os.getenv)
import sys       # import 경로(sys.path) 조정용. 파일 경로로 직접 실행해도 src 를 찾게 함
from dataclasses import dataclass, field   # 설정값을 묶는 클래스를 간단히 만드는 도구
from pathlib import Path                   # 경로를 문자열 대신 객체로 다룸 (OS 무관하게 동작)
# 타입 힌트 전용. 실행에는 영향 없음.
# Python 3.10+ 에서는 Optional[float] → float | None, List[str] → list[str] 처럼 대체 가능
from typing import Any, Dict, List, Mapping, Optional, Sequence

# ---- 외부 라이브러리 (requirements.txt 에 명시 필요) ----
import numpy as np                 # 배열 계산
import pandas as pd                # DataFrame
from scipy import stats            # ANOVA(f_oneway), Tukey HSD, Shapiro, Levene 검정
from sklearn.base import clone     # 모델을 "학습 안 된 새 복사본"으로 복제
from sklearn.metrics import roc_auc_score                    # AUC 계산
from sklearn.model_selection import RepeatedStratifiedKFold  # 클래스 비율을 유지하는 K-Fold를 여러 번 반복

# 이 파일 전용 로거. __name__ 은 "src.statistics.anova" 가 되어
# 로그에 어느 모듈에서 찍힌 것인지 표시된다.
logger = logging.getLogger(__name__)

# __file__ = 현재 파일 경로(.../src/statistics/anova.py)
# .resolve() 로 절대 경로로 바꾸고,
# .parents[0] = statistics/, [1] = src/, [2] = 프로젝트 루트
# → 절대 경로를 하드코딩하지 않고도 어디서 실행하든 루트를 찾을 수 있다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# VS Code 의 ▶(Run/Debug) 버튼이나 python src/statistics/anova.py 처럼 "파일 경로"로 실행하면
# 파이썬은 이 파일이 있는 src/statistics/ 폴더만 import 경로에 넣는다.
# 그러면 프로젝트 루트가 경로에 없어서 "from src.data..." 가 ModuleNotFoundError 를 낸다.
# → 프로젝트 루트를 import 경로 맨 앞에 직접 추가한다.
#   (python -m src.statistics.anova 로 실행할 때는 이미 들어 있으므로 중복 추가하지 않음)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 피처 목록은 여기서 하드코딩하지 않는다.
#   - 대안 변수 5종: src/models/model.py 의 ALT_COLS 를 그대로 사용
#   - 전통 변수   : 전처리된 X_train 컬럼 중 ALT_COLS 를 뺀 나머지 전부
#                  (결측 지시변수 missingindicator_xxx 는 원래 변수 xxx 의 그룹을 따름)
# → preprocessor 에서 컬럼이 바뀌어도 이 파일은 고칠 필요가 없다.

# model.py 의 알고리즘 키 → ANOVA 결과에 표시할 이름 (노션 예시와 같은 표기)
ALGORITHM_DISPLAY_NAMES = {
    "logistic_regression": "LogisticRegression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# 검증 1 의 세 그룹 이름 (튜플이라 수정 불가)
FEATURE_GROUP_NAMES = ("전통모델", "대안모델", "통합모델")


# =========================================================================== #
# 설정 (하드코딩 금지 → 환경 변수로 덮어쓰기)
#   노션 요건: "설정값(포트, 경로, 임계값)은 환경 변수 또는 config 파일로 분리"
# =========================================================================== #
def _resolve_path(value: str) -> Path:
    # 함수 이름 앞의 _ 는 "모듈 내부용"이라는 관례
    path = Path(value)  # 문자열 → Path 객체
    # 절대 경로면 그대로, 상대 경로면 프로젝트 루트 기준으로 붙인다
    return path if path.is_absolute() else PROJECT_ROOT / path


# @dataclass: __init__ 등을 자동으로 만들어 준다.
# frozen=True: 생성 후 값 변경 불가 (실수로 설정이 바뀌는 것 방지)
@dataclass(frozen=True)
class AnovaConfig:
    # 환경 변수 ANOVA_ALPHA 가 있으면 그 값, 없으면 "0.05" 를 float 로 변환.
    # default_factory=lambda: ... 를 쓴 이유:
    #   객체를 만들 때마다 환경 변수를 다시 읽기 위해서.
    #   그냥 = float(os.getenv(...)) 로 쓰면 import 시점에 한 번만 읽힌다.
    alpha: float = field(default_factory=lambda: float(os.getenv("ANOVA_ALPHA", "0.05")))
    # K-Fold 의 K (기본 5)
    n_splits: int = field(default_factory=lambda: int(os.getenv("ANOVA_N_SPLITS", "5")))
    # K-Fold 반복 횟수 (기본 3 → 5 × 3 = 그룹당 AUC 15개)
    n_repeats: int = field(default_factory=lambda: int(os.getenv("ANOVA_N_REPEATS", "3")))
    # 재현성을 위한 시드
    random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    # 결과 JSON 저장 위치
    output_path: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("ANOVA_RESULT_PATH", "reports/anova_results.json")
        )
    )


# =========================================================================== #
# 유틸 함수
# =========================================================================== #
def _to_float(value: Any, ndigits: Optional[int] = 6) -> Optional[float]:
    """numpy 값을 JSON 직렬화 가능한 float 로 변환 (NaN/inf → None).
    ndigits=None 이면 반올림하지 않는다."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value if ndigits is None else round(value, ndigits)


def _to_pvalue(value: Any) -> Optional[float]:
    """p값은 소수 자릿수 반올림 대신 유효숫자 4자리로 유지.
    예) 0.000000000032144 → 3.214e-11  (round(…, 6) 이면 0.0 이 돼버림)"""
    v = _to_float(value, ndigits=None)
    return None if v is None else float(f"{v:.4g}")

def interpret_eta_squared(eta_squared: Optional[float]) -> str:
    """노션 가이드의 η² 해석 기준."""
    if eta_squared is None:
        return "N/A"
    if eta_squared < 0.01:
        return "Negligible"   # 0.01 미만: 사실상 효과 없음
    if eta_squared < 0.06:
        return "Small"        # 노션 표: 0.01 ~ 0.06
    if eta_squared < 0.14:
        return "Medium"       # 0.06 ~ 0.14
    return "Large"            # 0.14 이상


def eta_squared(samples: Sequence[np.ndarray]) -> float:
    """η² = SS_between / SS_total.

    "전체 AUC 변동 중 그룹 차이로 설명되는 비율".
    예: 0.34 → AUC 차이의 34% 가 그룹(피처 구성/알고리즘) 때문.
    """
    all_values = np.concatenate(samples)   # 모든 그룹 값을 하나로 합침
    grand_mean = all_values.mean()         # 전체 평균
    # 집단 간 제곱합: 각 그룹 평균이 전체 평균에서 떨어진 정도² × 그룹 크기
    ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
    # 전체 제곱합: 모든 값이 전체 평균에서 떨어진 정도²
    ss_total = ((all_values - grand_mean) ** 2).sum()
    # 전체 변동이 0 이면 0 반환 (0 으로 나누기 방지)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _check_assumptions(samples: Sequence[np.ndarray], names: Sequence[str]) -> Dict[str, Any]:
    """ANOVA 가정 확인: 정규성(Shapiro-Wilk), 등분산성(Levene).

    두 검정의 p값이 0.05 이상이면 가정을 기각하지 못함
    → ANOVA 결과를 믿을 근거가 된다 (리포트에 쓰기 좋음).
    """
    normality = {}
    for name, s in zip(names, samples):   # 그룹 이름과 값을 짝지어 순회
        # Shapiro 는 관측치 3개 이상 필요.
        # np.ptp = max - min. 0 보다 크면 값이 전부 같지는 않다는 뜻
        if len(s) >= 3 and np.ptp(s) > 0:
            normality[name] = _to_float(stats.shapiro(s).pvalue)   # 정규성 검정 p값
        else:
            normality[name] = None   # 검정 불가

    levene_p = None
    if all(np.ptp(s) > 0 for s in samples):   # 모든 그룹에 분산이 있을 때만
        # 등분산성 검정. *samples 는 리스트를 풀어서 각각의 인자로 전달
        levene_p = _to_float(stats.levene(*samples).pvalue)

    return {
        "shapiro_p_values": normality,
        "levene_p_value": levene_p,
        "note": "p >= 0.05 이면 정규성/등분산성 가정을 기각하지 않음",
    }


# =========================================================================== #
# 핵심: One-way ANOVA + Tukey HSD
# =========================================================================== #
def one_way_anova(
    group_scores: Mapping[str, Sequence[float]],
    test_name: str = "One-way ANOVA",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """그룹별 점수(예: fold별 AUC)에 대해 One-way ANOVA를 수행한다.

    Args:
        group_scores: {그룹명: 점수 리스트}. 그룹 2개 이상, 그룹당 2개 이상 관측치.
        test_name: 결과에 표시될 검증 이름.
        alpha: 유의수준.

    Returns:
        JSON 직렬화 가능한 dict (API /anova 응답 형식).
    """
    # 그룹이 1개면 비교 불가
    if len(group_scores) < 2:
        raise ValueError("ANOVA는 2개 이상의 그룹이 필요합니다.")

    names = list(group_scores.keys())   # 예: ["전통모델", "대안모델", "통합모델"]
    # 각 그룹 점수를 numpy 배열로
    samples = [np.asarray(group_scores[n], dtype=float) for n in names]
    for name, s in zip(names, samples):
        if len(s) < 2:   # 관측치 1개면 분산 계산 불가
            raise ValueError(f"그룹 '{name}'의 관측치가 2개 미만입니다.")
        if np.isnan(s).any():   # NaN 이 섞이면 결과가 오염되므로 차단
            raise ValueError(f"그룹 '{name}'에 NaN 값이 있습니다.")

    # 모든 그룹의 모든 값이 완전히 같으면 F = 0/0 → 정의되지 않음.
    # {s[0] for ...} 는 집합(set)이라 중복이 제거됨 → 길이 1 이면 모든 그룹 값이 동일.
    # 이 경우 NaN 으로 두고, 뒤의 _to_float 에서 None(null) 으로 바뀐다.
    if all(np.ptp(s) == 0 for s in samples) and len({s[0] for s in samples}) == 1:
        f_stat, p_value = float("nan"), float("nan")
    else:
        # 실제 One-way ANOVA 수행.
        # F = 집단 간 분산 / 집단 내 분산.
        # 그룹 평균 차이가 그룹 내부의 흔들림보다 클수록 F↑, p↓
        f_stat, p_value = stats.f_oneway(*samples)

    eta2 = eta_squared(samples)       # 효과 크기
    p_clean = _to_pvalue(p_value)
    is_significant = p_clean is not None and p_clean < alpha   # p < 0.05 이면 유의

    # 노션 예시의 /anova 응답 형식에 맞춘 딕셔너리
    result: Dict[str, Any] = {
        "test_name": test_name,
        "groups": names,
        # 그룹별 평균 AUC
        "group_means": {n: _to_float(s.mean(), 4) for n, s in zip(names, samples)},
        # 표본 표준편차. ddof=1 → n-1 로 나눔(불편추정)
        "group_stds": {n: _to_float(s.std(ddof=1), 4) for n, s in zip(names, samples)},
        # 그룹별 관측치 수
        "n_per_group": {n: int(len(s)) for n, s in zip(names, samples)},
        "f_statistic": _to_float(f_stat, 4),
        "p_value": p_clean,
        "effect_size_eta_squared": _to_float(eta2, 4),
        "effect_size_interpretation": interpret_eta_squared(eta2),
        "alpha": alpha,
        # numpy bool 을 파이썬 bool 로 (JSON 호환)
        "is_significant": bool(is_significant),
        # 일단 None, 유의할 때만 아래에서 채움
        "post_hoc_tukey": None,
        "assumptions": _check_assumptions(samples, names),
    }

    # 노션 요건: "p < 0.05 인 경우 사후 검정(Tukey HSD) 결과가 포함된다"
    # ANOVA 는 "어딘가에 차이가 있다"까지만 알려주므로,
    # "어느 쌍이" 다른지는 Tukey 로 확인한다.
    if is_significant:
        result["post_hoc_tukey"] = tukey_hsd(names, samples, alpha=alpha)

    # 자연어 해석 문장 추가 (리포트/대시보드용)
    result["interpretation"] = _interpret(result)

    # 결과를 로그로 출력.
    # %s 방식은 로그가 실제로 찍힐 때만 문자열을 만들어 효율적이다.
    logger.info(
        "[ANOVA] %s | F=%s, p=%s, eta2=%s (%s), significant=%s",
        test_name, result["f_statistic"], result["p_value"],
        result["effect_size_eta_squared"], result["effect_size_interpretation"],
        is_significant,
    )
    return result


def tukey_hsd(
    names: Sequence[str], samples: Sequence[np.ndarray], alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Tukey HSD 사후 검정. mean_diff = 앞 그룹 평균 - 뒤 그룹 평균.

    여러 쌍을 동시에 비교하면 우연히 "유의"가 나올 확률(1종 오류)이 커지는데,
    Tukey HSD 는 이를 보정한 p값(p_adj)을 준다.
    """
    res = stats.tukey_hsd(*samples)   # 모든 그룹 쌍 비교
    ci = res.confidence_interval(confidence_level=1 - alpha)   # 95% 신뢰구간

    comparisons = []
    for i in range(len(names)):
        # i < j 인 쌍만 → (0,1), (0,2), (1,2) 3쌍. 중복·자기 자신 비교 제외
        for j in range(i + 1, len(names)):
            p_adj = _to_pvalue(res.pvalue[i, j])
            comparisons.append({
                "comparison": f"{names[i]} vs {names[j]}",   # 예: "전통모델 vs 통합모델"
                # 평균(i) − 평균(j). 음수면 j 가 더 높음
                # (노션 예시 "전통 vs 통합", mean_diff: -0.07 과 같은 부호)
                "mean_diff": _to_float(res.statistic[i, j], 4),
                "p_adj": p_adj,
                "ci_lower": _to_float(ci.low[i, j], 4),    # 차이의 신뢰구간 하한
                "ci_upper": _to_float(ci.high[i, j], 4),   # 상한. 구간에 0 이 없으면 유의
                "significant": bool(p_adj is not None and p_adj < alpha),
            })
    return comparisons


def _interpret(result: Dict[str, Any]) -> str:
    """리포트/대시보드에 바로 붙여 쓸 수 있는 자연어 해석."""
    means = result["group_means"]
    # 평균이 가장 높은 그룹 이름. None 이면 -1 로 취급해 선택되지 않게 함
    best = max(means, key=lambda k: means[k] if means[k] is not None else -1)
    eta = result["effect_size_eta_squared"]

    # 유의하지 않으면 여기서 끝
    if not result["is_significant"]:
        return (
            f"그룹 간 평균 차이가 통계적으로 유의하지 않습니다 "
            f"(F={result['f_statistic']}, p={result['p_value']}, α={result['alpha']}). "
            f"η²={eta} ({result['effect_size_interpretation']})."
        )

    text = (
        f"그룹 간 평균 차이가 통계적으로 유의합니다 "
        f"(F={result['f_statistic']}, p={result['p_value']}). "
        f"η²={eta}로 효과 크기는 {result['effect_size_interpretation']}이며, "
        f"평균이 가장 높은 그룹은 {best}({means[best]})입니다."
    )
    # Tukey 결과에서 유의한 쌍 / 유의하지 않은 쌍 목록
    sig = [c["comparison"] for c in result["post_hoc_tukey"] if c["significant"]]
    nonsig = [c["comparison"] for c in result["post_hoc_tukey"] if not c["significant"]]
    if sig:
        text += f" Tukey HSD 결과 유의한 차이: {', '.join(sig)}."
    if nonsig:
        text += f" 유의하지 않은 차이: {', '.join(nonsig)}."
    return text


# =========================================================================== #
# CV 기반 AUC 표본 생성
# =========================================================================== #
def _predict_score(model, X) -> np.ndarray:
    # 확률 예측이 가능한 모델이면 클래스 1(연체) 확률 열만 사용
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # SVM 등 확률이 없는 모델은 결정 점수 사용 (AUC 는 순위만 보므로 OK)
    return model.decision_function(X)


def cross_val_auc_scores(estimator, X: pd.DataFrame, y: pd.Series, config: AnovaConfig) -> List[float]:
    """Repeated Stratified K-Fold 로 fold별 AUC 목록을 계산한다.

    random_state 가 같으므로 모든 모델이 똑같은 fold 분할로 평가된다(공정 비교).
    → AUC 차이가 데이터 분할 운이 아니라 모델 차이에서 나온다고 볼 수 있다.
    """
    # Stratified: 각 fold 에서 연체 비율을 원본과 같게 유지 (불균형 데이터에서 필수)
    # 5-fold 를 3번 반복 → 15개 fold
    cv = RepeatedStratifiedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.random_state
    )
    # 인덱스를 0,1,2... 로 초기화 (iloc 위치 인덱싱과 어긋나지 않게)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    scores = []
    for train_idx, val_idx in cv.split(X, y):   # fold 마다 학습/검증 행 번호를 받음
        model = clone(estimator)   # 매 fold 마다 새 모델 (이전 fold 학습 결과가 섞이지 않게)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])   # 학습 fold 로 학습
        # 검증 fold AUC 저장
        scores.append(roc_auc_score(y.iloc[val_idx], _predict_score(model, X.iloc[val_idx])))
    return scores   # 길이 15 의 AUC 리스트 = ANOVA 의 한 그룹 표본


def split_feature_groups(columns: Sequence[str]) -> Dict[str, List[str]]:
    """전처리된 컬럼을 전통 / 대안 피처로 나눈다.
    결측 지시변수(missingindicator_xxx)는 원래 변수 xxx 의 그룹을 따른다."""
    from src.models.model import ALT_COLS

    def base(c: str) -> str:
        return c.removeprefix("missingindicator_")

    alternative = [c for c in columns if base(c) in ALT_COLS]
    traditional = [c for c in columns if base(c) not in ALT_COLS]
    missing = sorted(set(ALT_COLS) - {base(c) for c in alternative})
    if missing:
        raise KeyError(f"전처리 데이터에 대안 변수가 없습니다: {missing} (시뮬레이터 결과 CSV 인지 확인)")
    return {"traditional": traditional, "alternative": alternative}


def load_final_estimators() -> Dict[str, Any]:
    """model.py 가 저장한 알고리즘별 최종 모델 3개를 불러온다.

    각 파일은 model.py 에서 '알고리즘별로 Validation AUC 가 더 좋은 불균형 기법'으로 저장된 모델이다.
    (class_weight 버전이면 모델 객체, SMOTE 버전이면 imblearn Pipeline[smote → clf])
    둘 다 컬럼 이름에 의존하는 전처리를 포함하지 않으므로, 검증 1 에서 피처 일부만 넣어도 동작한다.
    """
    import joblib
    from src.models.model import ALGORITHMS, MODEL_DIR, MODEL_VERSION

    estimators = {}
    for algo in ALGORITHMS:   # ["logistic_regression", "xgboost", "lightgbm"]
        path = MODEL_DIR / f"{algo}_v{MODEL_VERSION}.pkl"   # 예: models/xgboost_v1.0.pkl
        if not path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {path} (먼저 python -m src.models.model 실행)")
        # 학습된 모델을 그대로 담아도 된다.
        # cross_val_auc_scores() 안의 clone() 이 학습 결과는 버리고 설정만 복사하기 때문.
        estimators[ALGORITHM_DISPLAY_NAMES.get(algo, algo)] = joblib.load(path)
        logger.info("[ANOVA] 모델 로드: %s", path.name)
    return estimators


def load_best_model_name() -> str:
    """model_meta_v1.0.json 에 기록된 최종 모델의 표시 이름 (검증 1 에 사용)."""
    from src.models.model import MODEL_DIR, MODEL_VERSION

    meta_path = MODEL_DIR / f"model_meta_v{MODEL_VERSION}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"메타 파일이 없습니다: {meta_path} (먼저 python -m src.models.model 실행)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    algo = meta["best_model"]["algorithm"]          # 예: "xgboost"
    return ALGORITHM_DISPLAY_NAMES.get(algo, algo)  # 예: "XGBoost"


def load_train_data():
    """model.py 와 같은 방식으로 전처리한 Train 데이터만 돌려준다.

    CreditPreprocessor 는 같은 RANDOM_STATE 로 분할하므로 model.py 와 Train 구성이 동일하다.
    Validation/Test 는 여기서 쓰지 않는다 (테스트 라벨 사용 금지 조건).
    """
    from src.data.preprocessor import DEFAULT_INPUT, CreditPreprocessor

    csv_path = Path(os.getenv("ANOVA_INPUT_CSV", str(DEFAULT_INPUT)))   # 기본: 시뮬레이터 결과 CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {csv_path.resolve()}")
    data = CreditPreprocessor().process(pd.read_csv(csv_path))
    return data.X_train, data.y_train


# =========================================================================== #
# 검증 1 / 검증 2
# =========================================================================== #
def run_feature_group_test(
    X: pd.DataFrame,
    y: pd.Series,
    traditional_features: Sequence[str],
    alternative_features: Sequence[str],
    estimator,
    config: Optional[AnovaConfig] = None,
) -> Dict[str, Any]:
    """검증 1: 동일 알고리즘으로 피처 그룹만 바꿔 AUC를 비교한다.

    핵심: 알고리즘은 고정하고 피처만 바꾼다.
    그래야 AUC 차이를 "대안 데이터의 효과"로 해석할 수 있다.
    """
    config = config or AnovaConfig()   # config 안 주면 기본값(환경 변수) 사용
    # 두 피처 목록의 합집합에서 실제 데이터 컬럼을 빼면 → 없는 컬럼만 남는다
    missing = set(traditional_features) | set(alternative_features)
    missing -= set(X.columns)
    if missing:   # 오타 등을 빨리 발견
        raise KeyError(f"데이터에 없는 컬럼: {sorted(missing)}")

    feature_sets = {
        FEATURE_GROUP_NAMES[0]: list(traditional_features),   # 전통모델: 기존 금융 변수만
        FEATURE_GROUP_NAMES[1]: list(alternative_features),   # 대안모델: 대안 변수만
        FEATURE_GROUP_NAMES[2]: list(traditional_features) + list(alternative_features),   # 통합모델: 둘 다
    }
    group_scores = {}
    for name, cols in feature_sets.items():
        logger.info("[ANOVA-1] %s (%d features) CV 진행 중...", name, len(cols))
        # 같은 모델, 다른 피처로 fold 별 AUC 계산
        group_scores[name] = cross_val_auc_scores(estimator, X[cols], y, config)

    result = one_way_anova(group_scores, "피처 그룹별 AUC 비교 (One-way ANOVA)", config.alpha)
    # 원본 fold 점수도 저장 (대시보드에서 박스플롯 그릴 때 유용)
    result["fold_scores"] = {k: [_to_float(v, 4) for v in s] for k, s in group_scores.items()}
    return result


def run_algorithm_test(
    X: pd.DataFrame,
    y: pd.Series,
    estimators: Mapping[str, Any],
    config: Optional[AnovaConfig] = None,
) -> Dict[str, Any]:
    """검증 2: 동일 피처(통합)로 알고리즘별 AUC를 비교한다.

    검증 1 과 반대로 피처를 고정하고 알고리즘만 바꾼다.
    """
    config = config or AnovaConfig()
    group_scores = {}
    for name, est in estimators.items():   # {"LogisticRegression": 모델, ...} 순회
        logger.info("[ANOVA-2] %s CV 진행 중...", name)
        group_scores[name] = cross_val_auc_scores(est, X, y, config)

    result = one_way_anova(group_scores, "알고리즘별 AUC 비교 (One-way ANOVA)", config.alpha)
    result["fold_scores"] = {k: [_to_float(v, 4) for v in s] for k, s in group_scores.items()}
    return result


def run_all_anova(
    X: pd.DataFrame,
    y: pd.Series,
    traditional_features: Optional[Sequence[str]] = None,
    alternative_features: Optional[Sequence[str]] = None,
    feature_group_estimator=None,
    algorithm_estimators: Optional[Mapping[str, Any]] = None,
    config: Optional[AnovaConfig] = None,
    save: bool = True,
    log_mlflow: bool = False,
) -> Dict[str, Any]:
    """두 검증을 모두 수행하고 /anova API 응답 형식의 dict 를 반환한다.

    인자를 생략하면 model.py 의 결과물을 자동으로 사용한다.
      - 피처 그룹   : ALT_COLS 기준으로 자동 분리
      - 검증 2 모델 : models/ 의 최종 모델 3개
      - 검증 1 모델 : model_meta 의 best_model
    """
    config = config or AnovaConfig()

    # 피처 그룹을 안 주면 전처리된 컬럼에서 자동으로 나눈다
    if traditional_features is None or alternative_features is None:
        groups = split_feature_groups(list(X.columns))
        traditional_features = traditional_features or groups["traditional"]
        alternative_features = alternative_features or groups["alternative"]

    # 모델을 안 주면 model.py 가 저장한 최종 모델 3개를 불러온다
    best_name = None
    if algorithm_estimators is None:
        algorithm_estimators = load_final_estimators()
        best_name = load_best_model_name()

    # 검증 1 에 쓸 모델: 실제 서비스에 쓰는 최종 모델과 같은 설정이어야
    # "대안 데이터가 우리 모델에서 효과가 있다"고 말할 수 있다.
    # 최종 모델 이름을 모르면(모델을 직접 넘긴 경우) 첫 번째 모델 사용.
    # next(iter(...)) = "첫 번째 값 꺼내기"
    if feature_group_estimator is None:
        feature_group_estimator = algorithm_estimators.get(
            best_name, next(iter(algorithm_estimators.values()))
        )

    all_features = list(traditional_features) + list(alternative_features)
    results = {
        # 검증 1
        "test_1_feature_group": run_feature_group_test(
            X, y, traditional_features, alternative_features, feature_group_estimator, config
        ),
        # 검증 2 는 통합 피처로
        "test_2_algorithm": run_algorithm_test(X[all_features], y, algorithm_estimators, config),
        # 어떤 조건으로 실험했는지 기록
        "meta": {
            "metric": "ROC-AUC",
            "cv": f"RepeatedStratifiedKFold(n_splits={config.n_splits}, n_repeats={config.n_repeats})",
            "alpha": config.alpha,
            "n_samples": int(len(y)),
            "feature_group_model": best_name or "user-provided",
            "n_traditional_features": len(traditional_features),
            "n_alternative_features": len(alternative_features),
            # 통계적 한계 명시: fold 끼리 학습 데이터가 겹쳐 p값이 다소 낙관적일 수 있음
            "note": "fold 간 학습 데이터가 겹쳐 관측치가 완전히 독립은 아니므로 p-value 해석 시 유의",
        },
    }

    # 제약사항: "최소 1개 이상 검증에서 p < 0.05"
    any_sig = any(results[k]["is_significant"] for k in ("test_1_feature_group", "test_2_algorithm"))
    results["meta"]["requirement_any_significant"] = any_sig
    if not any_sig:
        logger.warning("[ANOVA] 두 검증 모두 p >= %s → 제약사항 'ANOVA 유의성' 미충족", config.alpha)

    if save:
        save_results(results, config.output_path)   # JSON 파일로 저장
    if log_mlflow:
        log_results_to_mlflow(results)              # 옵션이면 MLflow 에도 기록
    return results


# =========================================================================== #
# 저장 / 로드 / MLflow
#   ANOVA 는 모델 3개 × 15 fold 학습을 두 번 하므로 몇 분이 걸릴 수 있다.
#   /anova 호출마다 새로 계산하면 너무 느리므로
#   "학습 단계에서 미리 계산 → JSON 저장 → API 는 읽기만" 하는 구조.
#   json 은 파이썬 기본 내장이라 설치가 필요 없고, API 응답도 원래 JSON 이라 가장 자연스럽다.
# =========================================================================== #
def save_results(results: Dict[str, Any], path: Optional[Path] = None) -> Path:
    path = Path(path or AnovaConfig().output_path)   # 경로 안 주면 config 기본 경로
    path.parent.mkdir(parents=True, exist_ok=True)   # reports/ 폴더가 없으면 생성 (이미 있어도 에러 없음)
    # ensure_ascii=False: 한글이 \uc804... 로 깨지지 않게
    # indent=2: 사람이 읽기 좋게 들여쓰기
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[ANOVA] 결과 저장: %s", path)
    return path


def load_results(path: Optional[Path] = None) -> Dict[str, Any]:
    """FastAPI /anova 에서 사전 계산된 결과를 읽을 때 사용."""
    path = Path(path or AnovaConfig().output_path)
    # 아직 ANOVA 를 안 돌렸으면 명확한 에러 → API 에서 404 로 변환
    if not path.exists():
        raise FileNotFoundError(f"ANOVA 결과 파일이 없습니다: {path} (먼저 anova.py 를 실행하세요)")
    return json.loads(path.read_text(encoding="utf-8"))   # 텍스트 파일 → 파이썬 dict


def log_results_to_mlflow(results: Dict[str, Any], run_name: str = "anova_validation") -> None:
    try:
        import mlflow
        # model.py 와 같은 서버·같은 실험("credit-scoring")에 기록한다.
        # 서버가 꺼져 있으면 model.py 와 똑같이 로컬 sqlite 로 대체된다.
        from src.models.model import DEFAULT_TRACKING_URI, setup_mlflow
    except ImportError:
        # mlflow 가 없어도 전체가 죽지 않게 경고만 남기고 넘어감
        logger.warning("mlflow 미설치 → MLflow 로깅 생략")
        return

    setup_mlflow(DEFAULT_TRACKING_URI)

    # MLflow 실험 run 하나 시작 (with 블록이 끝나면 자동 종료)
    with mlflow.start_run(run_name=run_name):
        for key in ("test_1_feature_group", "test_2_algorithm"):
            r = results[key]
            prefix = "anova1" if key.startswith("test_1") else "anova2"
            for metric in ("f_statistic", "p_value", "effect_size_eta_squared"):
                if r.get(metric) is not None:
                    mlflow.log_metric(f"{prefix}_{metric}", r[metric])   # 예: anova1_p_value
        # 실험 조건 기록
        mlflow.log_params({"alpha": results["meta"]["alpha"], "cv": results["meta"]["cv"]})
        # 전체 결과를 artifact 로 저장
        mlflow.log_dict(results, "anova_results.json")


# =========================================================================== #
# 명령줄 실행부
#   사용법: python -m src.statistics.anova [--mlflow]
#   순서  : python -m src.models.model (모델 학습) → python -m src.statistics.anova
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="ANOVA 통계 검증 (Train split 전용)")
    # 플래그만 주면 True
    parser.add_argument("--mlflow", action="store_true", help="MLflow 에 결과 기록")
    args = parser.parse_args()   # 실제로 터미널 인자를 읽음

    # INFO 이상 로그를 시간과 함께 출력
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # model.py 와 같은 전처리 → Train 만 사용
    X_train, y_train = load_train_data()
    results = run_all_anova(X_train, y_train, log_mlflow=args.mlflow)

    for key in ("test_1_feature_group", "test_2_algorithm"):
        print(f"\n[{key}] {results[key]['interpretation']}")   # 터미널에 해석 문장 출력


# python -m src.statistics.anova 로 "직접 실행했을 때만" main() 을 호출한다.
# API 처럼 다른 파일에서 import 할 때는 실행되지 않는다.
if __name__ == "__main__":
    main()