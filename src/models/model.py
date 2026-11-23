"""
src/models/model.py
XAI 기반 대안신용평가 시스템 - 신용평가 모델 학습 / 평가 / MLflow 기록

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

[변경점] SHAP이 KernelExplainer로 넘어가던 문제 대응 (멘토링 피드백 1번)
  - 문제: SMOTE 모델은 ImbPipeline 객체로 저장됨 -> shap이 트리 모델로 인식 못 함
          -> KernelExplainer(샘플링 기반)로 넘어가서 실행할 때마다 거절 사유가 바뀜
  - 해결: unwrap_model()로 Pipeline 안의 LGBMClassifier만 꺼내 TreeExplainer에 넘김
          (SMOTE는 fit 때만 동작하고 predict 때는 아무것도 안 하므로 X는 그대로 넣으면 됨)
  - 확인: check_tree_explainer()가 같은 입력을 30번 설명해 거절 사유(상위 3개)가
          한 번도 안 바뀌는지, SHAP 합 + base value == 모델 로그오즈인지 검사
          -> 결과는 model_meta_v{버전}.json 의 "shap_check" 에 저장
  - API(main.py)에는 같은 함수를 따로 구현해 사용함
          (이 파일은 mlflow 등 학습용 라이브러리를 불러와, API 환경에서 import 하면 오류가 남)

[실행 옵션] 최종 모델은 v1.3 (class_weight만, 데이터 정제 반영). SMOTE 비교는 옵션으로 남겨 둠
  python -m src.models.model                -> v1.3: class_weight만 사용 (실험 3개, 최종)
  python -m src.models.model --with-smote   -> v1.1: class_weight vs SMOTE 비교 (실험 6개, 기록용)
  - 버전별로 모델 pkl, 메타 json, 전처리기 파일명이 달라서 서로 덮어쓰지 않음
  - 기본 실행(v1.3)은 imbalanced-learn 없이도 실행됨
  - --with-smote 결과는 Model Registry에 Staging으로만 등록 (Production을 덮어쓰지 않음)

[최종 모델 선정 기준] Validation AUC -> Train 5-Fold CV AUC 평균
  - Val AUC 차이(0.0007)가 CV 표준편차(±0.005)보다 작아 순위가 노이즈에 좌우됐음
  - CV 평균은 5개 fold 평균이라 더 안정적

체크리스트 대응
  [모델 학습]
  1) 3개 알고리즘          : Logistic Regression, XGBoost, LightGBM
  2) 5-Fold Stratified CV  : 각 실험마다 Train 70% 안에서 StratifiedKFold(5) AUC 계산
  3) 불균형 처리 2개 비교  : class_weight='balanced' vs SMOTE
                             (SMOTE는 imblearn Pipeline으로 CV 각 fold 안에서만 적용 -> 누수 없음)
  4) 모델 저장             : 프로젝트 루트 models/ 폴더에 .pkl
  5) 버전 포함 파일명      : xgboost_v{버전}.pkl, lightgbm_v{버전}.pkl, logistic_regression_v{버전}.pkl
  [성능 목표]
  - Test AUC-ROC >= 0.78, Test KS >= 0.28, 씬파일러 AUC 향상 >= +0.03, PSI < 0.1
    -> 실행 마지막에 PASS/FAIL 표로 출력, models/model_meta_v{버전}.json 에 저장 (/metrics API용)
  [승인 판정 기준] 추가
  - 승인율 70% (시뮬레이터 add_approval_label의 과거 승인율과 동일)
  - Validation 부도확률의 하위 70% 지점을 approval_cutoff 로 저장
    -> /predict 에서 "부도확률 <= approval_cutoff 이면 승인"
    -> /fairness 에서도 같은 기준으로 승인 여부를 만들어야 과거 승인 기록과 비교 가능
  - 기존 threshold(F1 최대 지점)는 F1 평가용으로만 남겨 둠
  [MLflow]
  - 실험(알고리즘 x 불균형 기법)을 각각 run으로 기록 (기본 3개, --with-smote 6개): params, metrics(AUC/KS/F1), artifacts(모델 파일)
  - 최종 모델을 Model Registry에 등록하고 Staging/Production 태그 부여

데이터 사용 원칙
  - Train(70%) : 5-Fold CV + 최종 학습
  - Train(70%) 5-Fold CV 평균 : 실험 중 최종 모델 선택
  - Validation(15%) : F1 임계값·승인 기준값·등급 경계 결정
  - Test(15%) : 최종 성능 보고에만 사용 (선택·기준 결정에 쓰지 않음)

필요 패키지: scikit-learn>=1.2, xgboost, lightgbm, imbalanced-learn, mlflow, joblib
실행: 프로젝트 최상위 폴더에서  python -m src.models.model
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.data.preprocessor import (DEFAULT_INPUT, DEFAULT_SAVE, PROJECT_ROOT,
                                   RANDOM_STATE, CreditPreprocessor, SplitData)
from src.data.simulator import ALT_COLS   # 대안변수 목록은 시뮬레이터 한 곳에서 관리

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
# 파일명에 들어가는 버전. main()에서 --with-smote 여부에 따라 아래 둘 중 하나로 정해짐
VERSION_WITH_SMOTE = "1.1"    # class_weight vs SMOTE 비교 (기록용)
VERSION_NO_SMOTE = "1.3"      # class_weight만 (최종)
MODEL_VERSION = VERSION_NO_SMOTE
MODEL_DIR = PROJECT_ROOT / "models"   # 모델 저장 폴더

# 씬파일러 여부: preprocessor가 피처가 아닌 sensitive_* 에 담아서 돌려줌
THIN_FILER_COL = "is_thin_filer"

ALGORITHMS = ["logistic_regression", "xgboost", "lightgbm"]
IMBALANCE_METHODS = ["class_weight"]   # --with-smote 이면 main()에서 ["class_weight", "smote"]로 바뀜
N_FOLDS = 5

# 추가: 승인율. 시뮬레이터 add_approval_label(approval_rate=0.7)과 반드시 같은 값으로 맞출 것
#   -> 모델 승인 결과와 과거 승인 기록을 같은 승인율 조건에서 비교하기 위함
APPROVAL_RATE = 0.7
# 신용 등급 기준: Validation 부도확률 분포의 분위수로 4개 경계를 정함 (부도확률이 낮을수록 좋은 등급)
#   하위 0~20% → A, 20~45% → B, 45~70% → C, 70~85% → D, 85~100% → E
#   0.7 경계 = APPROVAL_RATE 와 같아야 "A~C 승인 / D~E 거절" 이 승인 판정과 일치함
GRADE_QUANTILES = [0.2, 0.45, 0.7, 0.85]
GRADE_LABELS = ["A", "B", "C", "D", "E"]
assert APPROVAL_RATE in GRADE_QUANTILES, "등급 경계에 승인율(0.7)이 포함돼야 승인 판정과 등급이 일치함"

TARGETS = {  # 성능 목표 (비교 방향, 목표값)
    "test_auc": (">=", 0.78),
    "test_ks": (">=", 0.28),
    "thin_filer_auc_uplift": (">=", 0.03),
    "psi": ("<", 0.10),
}

MLFLOW_EXPERIMENT = "credit-scoring"           # MLflow UI에 보일 실험 이름
REGISTERED_MODEL_NAME = "credit_scoring_model" # Model Registry 등록 이름
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ----------------------------------------------------------------------
# 1. 모델 생성
# ----------------------------------------------------------------------
def base_params(algo: str) -> dict:
    """알고리즘별 하이퍼파라미터 (MLflow params로 기록됨)"""
    if algo == "logistic_regression":
        return {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
    if algo == "xgboost":
        return {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                "eval_metric": "auc"}
    if algo == "lightgbm":
        return {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
    raise ValueError(algo)


def build_model(algo: str, imbalance: str, y_train: pd.Series):
    """
    imbalance="class_weight": 소수 클래스(부도)에 가중치를 더 줌
        - LR, LightGBM: class_weight='balanced'
        - XGBoost: class_weight 인자가 없어서 같은 효과인 scale_pos_weight = 정상 수 / 부도 수
    imbalance="smote": 학습 데이터에 부도 샘플을 합성해서 늘림 (가중치는 사용 안 함)
        - imblearn Pipeline 안에 넣어서 CV 시 각 fold의 학습 부분에만 적용됨
    """
    p = base_params(algo)
    balanced = imbalance == "class_weight"
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())  # 정상 수 / 부도 수

    if algo == "logistic_regression":
        clf = LogisticRegression(**p, class_weight="balanced" if balanced else None,
                                 random_state=RANDOM_STATE)
    elif algo == "xgboost":
        clf = XGBClassifier(**p, scale_pos_weight=pos_weight if balanced else 1.0,
                            n_jobs=-1, random_state=RANDOM_STATE)
    else:
        clf = LGBMClassifier(**p, class_weight="balanced" if balanced else None,
                             n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)

    if imbalance == "smote":
        # SMOTE를 쓸 때만 import (기본 실행에서는 imbalanced-learn 불필요)
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])
    return clf


def unwrap_model(model):
    """ImbPipeline이면 안쪽 분류기만 꺼냄 (SMOTE는 예측 시 동작 안 하므로 X 변환 불필요)"""
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


# ----------------------------------------------------------------------
# 1-1. SHAP 일관성 검사 (피드백 1번: "30번 중 22번 거절 사유 바뀜" 재현 테스트)
# ----------------------------------------------------------------------
TREE_ALGORITHMS = {"xgboost", "lightgbm"}   # TreeExplainer 대상 (LR은 LinearExplainer 사용)


def _positive_class_shap(explainer, X) -> tuple[np.ndarray, float]:
    """shap 버전에 따라 이진분류 출력 모양이 달라서 부도(1) 클래스 기준으로 맞춤"""
    sv = explainer.shap_values(X)
    base = explainer.expected_value
    if isinstance(sv, list):          # 구버전: [정상, 부도] 리스트
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # 신버전 일부: (n, 피처, 클래스)
        sv = sv[:, :, 1]
    base = np.atleast_1d(np.asarray(base, dtype=float))[-1]
    return sv, float(base)


def check_tree_explainer(model, X, n_repeat: int = 30, top_k: int = 3,
                         n_rows: int = 200) -> dict:
    """
    저장할 모델로 TreeExplainer가 제대로 도는지 검사
      1) explainer_type : 실제로 TreeExplainer가 만들어졌는지 (Kernel로 넘어가지 않았는지)
      2) reason_change_rate : 같은 입력을 n_repeat번 설명했을 때 거절 사유 top_k가 바뀐 비율
                              (KernelExplainer 시절 22/30 -> 여기서는 0이어야 정상)
      3) additivity_max_err : SHAP 합 + base value 와 모델 로그오즈의 최대 차이 (~0이어야 정상)
    거절 사유 = 부도확률을 올린 방향(SHAP > 0)으로 큰 변수 순서
    """
    import shap   # 학습만 할 때는 shap 없이도 돌도록 함수 안에서 import

    X = X.iloc[:n_rows] if hasattr(X, "iloc") else X[:n_rows]
    clf = unwrap_model(model)
    explainer = shap.TreeExplainer(clf)

    first_sv, base = _positive_class_shap(explainer, X)
    first_top = np.argsort(-first_sv, axis=1)[:, :top_k]

    changed = 0
    for _ in range(n_repeat - 1):
        sv, _ = _positive_class_shap(shap.TreeExplainer(clf), X)
        top = np.argsort(-sv, axis=1)[:, :top_k]
        changed += int(not np.array_equal(top, first_top))

    # 가법성: TreeExplainer(기본 설정)는 로그오즈 단위로 설명함
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    add_err = float(np.max(np.abs(first_sv.sum(axis=1) + base - logit)))

    return {
        "explainer_type": type(explainer).__name__,
        "n_repeat": n_repeat,
        "n_rows": int(len(X)),
        "top_k": top_k,
        "reason_change_rate": changed / max(n_repeat - 1, 1),
        "additivity_max_err": add_err,
        "passed": bool(changed == 0 and add_err < 1e-3),
    }


# ----------------------------------------------------------------------
# 2. 평가 지표
# ----------------------------------------------------------------------
def ks_statistic(y_true, y_score) -> float:
    """KS = max(TPR - FPR): 부도/정상 점수 분포가 가장 크게 벌어지는 지점의 차이"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    """
    PSI (Population Stability Index): 학습 때 점수 분포(expected) 대비
    다른 데이터 점수 분포(actual)가 얼마나 달라졌는지. < 0.1 이면 안정.
    구간은 expected의 분위수로 나눈다.
    """
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def best_f1_threshold(y_true, y_score) -> float:
    """
    Validation에서 F1이 최대가 되는 분류 임계값 (Test에는 이 값을 그대로 적용)
    ※ F1 평가용 기준. 실제 승인 판정은 아래 approval_cutoff_from()의 70% 기준을 사용한다.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def approval_cutoff_from(y_score, approval_rate: float = APPROVAL_RATE) -> float:
    """
    추가: 승인 기준값 계산
    부도확률을 낮은 순으로 줄 세웠을 때 하위 approval_rate(70%) 지점의 값.
    이 값 '이하'인 사람을 승인하면 전체 승인율이 약 70%가 된다.

    - 반드시 Validation 예측값으로 계산 (Test로 정하면 Test 결과가 기준 결정에 새어 들어감)
    - 한 번 정해 저장해 두면, 신청자가 1명씩 들어오는 /predict에서도 비교만 하면 됨
      (1명만으로는 '상위 몇 %'를 계산할 수 없으므로 미리 고정값이 필요)
    """
    return float(np.quantile(np.asarray(y_score), approval_rate))

def grade_cutoffs_from(y_score, quantiles=GRADE_QUANTILES) -> list[float]:
    """Validation 부도확률의 분위수 → 등급 경계값 4개 (approval_cutoff 와 같은 원리)"""
    return [float(v) for v in np.quantile(np.asarray(y_score), quantiles)]


def assign_grade(y_score, cutoffs, labels=GRADE_LABELS) -> np.ndarray:
    """부도확률 → 등급. 경계값과 같으면 좋은 쪽 등급 (승인 규칙 '부도확률 <= cutoff' 와 일치)
    예) cutoffs=[0.08, 0.20, 0.42, 0.61] 일 때 0.05→A, 0.42→C(승인), 0.43→D(거절)"""
    idx = np.searchsorted(cutoffs, np.asarray(y_score), side="left")
    return np.asarray(labels)[idx]

def evaluate(y_true, y_score, threshold: float, prefix: str) -> dict:
    """AUC, KS, F1, Precision, Recall 계산. '부도확률 >= threshold 이면 부도' 판정 기준"""
    y_pred = (y_score >= threshold).astype(int)
    return {
        f"{prefix}_auc": float(roc_auc_score(y_true, y_score)),
        f"{prefix}_ks": ks_statistic(y_true, y_score),
        f"{prefix}_f1": float(f1_score(y_true, y_pred)),
        # zero_division=0: 부도로 예측한 사람이 0명이어도 에러 없이 0으로 처리
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

def subgroup_auc(y_true: pd.Series, y_score, mask) -> float:
    """특정 집단(예: 씬파일러)만 골라서 AUC 계산. 한쪽 클래스만 있으면 NaN"""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0 or y_true[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], y_score[mask]))


def thin_filer_mask(sensitive: pd.DataFrame) -> np.ndarray:
    """씬파일러인 행만 True (preprocessor의 sensitive_* 에서 가져옴)"""
    if THIN_FILER_COL not in sensitive.columns:
        raise KeyError(f"씬파일러 컬럼 '{THIN_FILER_COL}'이 없습니다. "
                       "preprocessor.py의 SEGMENT_COLS를 확인하세요.")
    return (sensitive[THIN_FILER_COL].astype(int) == 1).to_numpy()


# ----------------------------------------------------------------------
# 3. 실험 1개 = (알고리즘, 불균형 기법)
# ----------------------------------------------------------------------
def run_experiment(algo: str, imbalance: str, data: SplitData) -> dict:
    model = build_model(algo, imbalance, data.y_train)

    # (1) 5-Fold Stratified CV (Train 70% 안에서)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(clone(model), data.X_train, data.y_train,
                             cv=cv, scoring="roc_auc", n_jobs=1)

    # (2) Train 전체로 학습 후 세 세트의 부도확률 예측
    model.fit(data.X_train, data.y_train)
    p_train = model.predict_proba(data.X_train)[:, 1]
    p_val = model.predict_proba(data.X_val)[:, 1]
    p_test = model.predict_proba(data.X_test)[:, 1]

    # (3) F1 임계값은 Validation으로 결정 -> Test에 그대로 적용
    threshold = best_f1_threshold(data.y_val, p_val)

    metrics = {"cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
               "threshold": threshold, "psi": psi(p_train, p_test)}
    metrics.update(evaluate(data.y_val, p_val, threshold, "val"))
    metrics.update(evaluate(data.y_test, p_test, threshold, "test"))
    metrics["test_auc_thin_filer"] = subgroup_auc(data.y_test, p_test, thin_filer_mask(data.sensitive_test))

    print(f"  {algo:20s} {imbalance:12s} CV AUC {metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f}"
          f"  Val AUC {metrics['val_auc']:.4f}  Test AUC {metrics['test_auc']:.4f}"
          f"  KS {metrics['test_ks']:.4f}")
    return {"algo": algo, "imbalance": imbalance, "model": model,
            "params": base_params(algo), "metrics": metrics}


# ----------------------------------------------------------------------
# 4. 씬파일러 AUC 향상: 대안데이터 포함 모델 vs 미포함 모델
# ----------------------------------------------------------------------
def thin_filer_uplift(algo: str, imbalance: str, df: pd.DataFrame,
                      with_alt_auc: float) -> dict:
    """
    같은 알고리즘/불균형 설정으로 대안변수 5종을 뺀 모델을 학습해 비교.
    preprocessor는 같은 행·같은 RANDOM_STATE로 분할하므로 Test 세트 구성이 동일하다.
    """
    pre_base = CreditPreprocessor(drop_cols=ALT_COLS)
    print("\n[model] 대안데이터 미포함 모델 학습 (씬파일러 비교용)")
    base = pre_base.process(df)
    model = build_model(algo, imbalance, base.y_train)
    model.fit(base.X_train, base.y_train)
    p_test = model.predict_proba(base.X_test)[:, 1]

    without_alt = subgroup_auc(base.y_test, p_test, thin_filer_mask(base.sensitive_test))
    return {"test_auc_thin_filer_without_alt": without_alt,
            "test_auc_thin_filer_with_alt": with_alt_auc,
            "thin_filer_auc_uplift": with_alt_auc - without_alt,
            "test_auc_without_alt": float(roc_auc_score(base.y_test, p_test))}


# ----------------------------------------------------------------------
# 5. MLflow
# ----------------------------------------------------------------------
def setup_mlflow(uri: str) -> str:
    """MLflow 서버(docker-compose)에 연결. 꺼져 있으면 로컬 DB로 대체해서 학습은 계속한다."""
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3)
        except Exception:
            fallback = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
            print(f"[model] MLflow 서버({uri})에 연결할 수 없어 로컬에 기록합니다: {fallback}\n"
                  f"        -> docker-compose up 으로 서버를 켠 뒤 다시 실행하면 UI에 기록됩니다.")
            uri = fallback
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return uri


def log_run(result: dict, model_path: Path | None, preprocessor_path: Path) -> str:
    """실험 1개를 MLflow run 1개로 기록 (params / metrics / artifacts)"""
    name = f"{result['algo']}_{result['imbalance']}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"algorithm": result["algo"], "imbalance": result["imbalance"],
                           "cv_folds": N_FOLDS, "model_version": MODEL_VERSION,
                           **result["params"]})
        mlflow.log_metrics({k: v for k, v in result["metrics"].items() if np.isfinite(v)})
        mlflow.sklearn.log_model(
            result["model"],
            "model",
            serialization_format="pickle",   # skops 대신 pickle로 저장 (Registry 등록용)
        )
        if model_path is not None:
            mlflow.log_artifact(str(model_path), artifact_path="model_file")  # .pkl 원본
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        return run.info.run_id


def register_best(run_id: str, extra_metrics: dict, promote: bool = True) -> None:
    """최종 모델 run에 추가 지표를 기록하고 Model Registry에 등록"""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k: v for k, v in extra_metrics.items() if np.isfinite(v)})
        mlflow.set_tag("final_model", "true")

    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client = MlflowClient()
    stage = "Production" if (promote and extra_metrics.get("all_targets_passed")) else "Staging"
    # 버전 태그 + alias (최신 MLflow 방식)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "stage", stage)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, stage.lower(), mv.version)
    # Stage (구버전 방식, MLflow UI의 Stage 칸에 표시). 버전에 따라 경고가 날 수 있음
    try:
        client.transition_model_version_stage(REGISTERED_MODEL_NAME, mv.version, stage)
    except Exception as e:  # 최신 버전에서 지원이 빠진 경우 태그/alias로 대체
        print(f"[model] Stage 지정은 건너뜀(태그/alias로 대체): {e}")
    print(f"[model] Model Registry 등록: {REGISTERED_MODEL_NAME} v{mv.version} -> {stage}")


# ----------------------------------------------------------------------
# 6. 결과 정리
# ----------------------------------------------------------------------
def check_targets(m: dict) -> dict:
    """성능 목표 4개를 PASS/FAIL 판정"""
    out = {}
    for key, (op, goal) in TARGETS.items():
        v = m.get(key, float("nan"))
        ok = np.isfinite(v) and (v >= goal if op == ">=" else v < goal)
        out[key] = {"value": v, "target": f"{op} {goal}", "pass": bool(ok)}
    return out


def print_targets(checks: dict) -> None:
    print("\n[model] 성능 목표 확인")
    for key, c in checks.items():
        print(f"  {key:24s} {c['value']:>8.4f}  (목표 {c['target']:>7s})  "
              f"{'PASS' if c['pass'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="신용평가 모델 학습 + MLflow 기록")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_INPUT), help="시뮬레이터 결과 CSV")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow 서버 주소")
    parser.add_argument("--with-smote", action="store_true",
                    help=f"class_weight vs SMOTE 비교 실험 (v{VERSION_WITH_SMOTE}, Registry는 Staging까지만)")
    args = parser.parse_args()

    global MODEL_VERSION, IMBALANCE_METHODS
    if args.with_smote:
        MODEL_VERSION, IMBALANCE_METHODS = VERSION_WITH_SMOTE, ["class_weight", "smote"]
    else:
        MODEL_VERSION, IMBALANCE_METHODS = VERSION_NO_SMOTE, ["class_weight"]
    print(f"[model] 버전 v{MODEL_VERSION} / 불균형 기법: {', '.join(IMBALANCE_METHODS)}")

    if not Path(args.csv).exists():
        raise SystemExit(f"[model] 입력 파일을 찾을 수 없습니다: {Path(args.csv).resolve()}")

    # 1) 전처리 (대안데이터 포함) + 전처리기 저장 (API에서 같은 변환을 쓰기 위해)
    df = pd.read_csv(args.csv)
    pre = CreditPreprocessor()
    data = pre.process(df)
    # 전처리기도 버전별 파일로 저장 (예: preprocessor.joblib -> preprocessor_v1.3.joblib)
    pre_path = Path(DEFAULT_SAVE)
    pre_path = pre_path.with_name(f"{pre_path.stem}_v{MODEL_VERSION}{pre_path.suffix}")
    pre.save(pre_path)

    # 2) 실험 (기본 3개, --with-smote 6개)
    print(f"\n[model] 실험 {len(ALGORITHMS) * len(IMBALANCE_METHODS)}개 "
          f"(알고리즘 {len(ALGORITHMS)} x 불균형 기법 {len(IMBALANCE_METHODS)}, {N_FOLDS}-Fold Stratified CV)")
    results = [run_experiment(a, i, data) for a in ALGORITHMS for i in IMBALANCE_METHODS]

    # 3) 알고리즘별로 CV AUC(Train 5-Fold 평균)가 더 좋은 불균형 기법의 모델을 models/에 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for algo in ALGORITHMS:
        best_of_algo = max((r for r in results if r["algo"] == algo),
                           key=lambda r: r["metrics"]["cv_auc_mean"])
        path = MODEL_DIR / f"{algo}_v{MODEL_VERSION}.pkl"
        joblib.dump(best_of_algo["model"], path)
        saved[id(best_of_algo)] = path

    # 4) 최종 모델 = CV AUC(Train 5-Fold 평균) 1위 (Val/Test는 선택에 쓰지 않음)
    best = max(results, key=lambda r: r["metrics"]["cv_auc_mean"])
    print(f"\n[model] 최종 모델: {best['algo']} + {best['imbalance']} "
          f"(CV AUC {best['metrics']['cv_auc_mean']:.4f}, Val AUC {best['metrics']['val_auc']:.4f})")

    # 4-1) 승인 기준값 (승인율 70%)
    #   Validation 부도확률로 기준값을 정하고, Test에 적용해 실제 승인율이 70% 근처인지 확인
    p_val_best = best["model"].predict_proba(data.X_val)[:, 1]     # Validation 부도확률
    approval_cutoff = approval_cutoff_from(p_val_best)             # 하위 70% 지점
    p_test_best = best["model"].predict_proba(data.X_test)[:, 1]   # Test 부도확률 (확인용)
    test_approval_rate = float((p_test_best <= approval_cutoff).mean())  # Test 실제 승인율
    print(f"[model] 승인 기준값(approval_cutoff) {approval_cutoff:.4f}  "
          f"목표 승인율 {APPROVAL_RATE:.0%} / Test 실제 승인율 {test_approval_rate:.2%}")

    # 4-2) 신용 등급 경계 (승인 기준값과 같은 방식: Validation 으로 정하고 Test 로 확인)
    grade_cutoffs = grade_cutoffs_from(p_val_best)
    test_grades = pd.Series(assign_grade(p_test_best, grade_cutoffs), name="grade")
    grade_check = (pd.DataFrame({"grade": test_grades, "default": data.y_test.to_numpy()})
                   .groupby("grade")["default"].agg(비율="size", 부도율="mean"))
    grade_check["비율"] = grade_check["비율"] / len(test_grades)
    print("[model] 등급 경계값:", [round(c, 4) for c in grade_cutoffs])
    print("[model] Test 등급 분포 (부도율은 A→E 로 갈수록 높아져야 정상)")
    print(grade_check.round(4).to_string())

    # 5) 씬파일러 AUC 향상
    uplift = thin_filer_uplift(best["algo"], best["imbalance"], df,
                               best["metrics"]["test_auc_thin_filer"])
    final_metrics = {**best["metrics"], **uplift}
    checks = check_targets(final_metrics)
    print_targets(checks)
    final_metrics["all_targets_passed"] = float(all(c["pass"] for c in checks.values()))

    # 5-1) SHAP 일관성 검사: 저장된 트리 모델(XGBoost, LightGBM) 각각에 대해
    print("\n[model] SHAP 일관성 검사 (같은 입력 30회 설명 -> 거절 사유 top3 변화율)")
    shap_check = {}
    for r in results:
        if r["algo"] in TREE_ALGORITHMS and id(r) in saved:
            try:
                c = check_tree_explainer(r["model"], data.X_test)
            except Exception as e:   # shap 미설치 등
                c = {"error": str(e), "passed": False}
            shap_check[r["algo"]] = {"imbalance": r["imbalance"], **c}
            if "error" in c:
                print(f"  {r['algo']:10s} 검사 실패: {c['error']}")
            else:
                print(f"  {r['algo']:10s} ({r['imbalance']:12s}) {c['explainer_type']:15s} "
                      f"사유 변화율 {c['reason_change_rate']:.0%}  "
                      f"가법성 오차 {c['additivity_max_err']:.2e}  "
                      f"{'PASS' if c['passed'] else 'FAIL'}")

    # 6) /metrics API, 보고서용 메타데이터
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": {
            "algorithm": best["algo"],
            "imbalance": best["imbalance"],
            "file": saved[id(best)].name,
            "threshold": best["metrics"]["threshold"],   # F1 최대 임계값 (평가용)
            # 추가: 승인 판정용 값. /predict, /fairness 에서는 이 값을 사용
            #   승인 규칙: 부도확률 <= approval_cutoff  ->  승인
            "approval_rate": APPROVAL_RATE,
            "approval_cutoff": approval_cutoff,
            "test_approval_rate": test_approval_rate,
            # 신용 등급 기준 (/predict 의 risk_grade 에서 사용)
            "grade_quantiles": GRADE_QUANTILES,
            "grade_labels": GRADE_LABELS,
            "grade_cutoffs": grade_cutoffs,
            
        },
        "preprocessor_file": pre_path.name,   # 이 버전 모델과 짝이 되는 전처리기 (API에서 사용)
        "feature_names": pre.feature_names,   # 전처리 후 피처 이름 (SHAP용)
        # SHAP은 unwrap_model(모델)로 TreeExplainer를 만들어야 함 (Pipeline째 넣으면 Kernel로 넘어감)
        "shap_check": shap_check,
        "final_metrics": final_metrics,
        "targets": checks,
        "experiments": [{"algorithm": r["algo"], "imbalance": r["imbalance"], **r["metrics"]}
                        for r in results],
    }
    meta_path = MODEL_DIR / f"model_meta_v{MODEL_VERSION}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[model] 저장: {', '.join(p.name for p in saved.values())}, {meta_path.name} -> {MODEL_DIR}")

    # 7) MLflow 기록 + Registry 등록 (승인 기준값도 최종 모델 run에 함께 기록)
    uri = setup_mlflow(args.tracking_uri)
    run_ids = {id(r): log_run(r, saved.get(id(r)), pre_path) for r in results}
    register_best(run_ids[id(best)], {
        **uplift,
        "all_targets_passed": final_metrics["all_targets_passed"],
        "approval_cutoff": approval_cutoff,          # 추가
        "test_approval_rate": test_approval_rate,    # 추가
        
    }, promote=not args.with_smote)
    print(f"[model] MLflow 기록 완료 ({uri}, 실험명 '{MLFLOW_EXPERIMENT}')")


if __name__ == "__main__":
    main()