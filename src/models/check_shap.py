"""
src/models/check_shap.py
멘토링 피드백 1번 검증: "고객 30명 중 22명의 거절 사유가 반복 분석 때마다 바뀜"

같은 거절 고객 30명을 여러 번 설명해서 거절 사유(top3)가 바뀌는 사람 수를 셈
  - Before : KernelExplainer (기존 방식 재현, 샘플링 기반)  -> 여러 명 바뀜
  - After  : TreeExplainer   (v1.2 LightGBM, 정확 계산)     -> 0/30 이어야 정상
추가로 "설명이 실제 판단 근거와 맞는지"도 확인
  - 가법성: SHAP 합 + base value == 모델 로그오즈 (설명이 모델 출력을 그대로 분해했는지)
  - 거절 사유 1순위 빈도: 무담보 한도 사용률(RevolvingUtilization...) 하나에 쏠리는지

실행: 프로젝트 최상위 폴더에서  python -m src.models.check_shap
      (먼저 python -m src.models.model 로 v1.2 모델 파일이 만들어져 있어야 함)
"""
from __future__ import annotations

import json
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocessor import DEFAULT_INPUT, CreditPreprocessor
from src.models.model import MODEL_DIR, _positive_class_shap, unwrap_model

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "1.2"
N_CUSTOMERS = 30      # 멘토링 때와 같은 30명
N_REPEAT = 10         # TreeExplainer 반복 횟수
N_REPEAT_KERNEL = 5   # KernelExplainer는 느려서 5번만
KERNEL_NSAMPLES = 100 # KernelExplainer 샘플 수 (작을수록 결과가 더 흔들림)
TOP_K = 3
SEED = 0


def top_reasons(sv_row: np.ndarray, names: list[str], k: int = TOP_K) -> tuple:
    """거절 사유 = 부도확률을 올린 방향(SHAP > 0)으로 큰 변수 k개"""
    idx = np.argsort(-sv_row)[:k]
    return tuple(names[i] for i in idx if sv_row[i] > 0)


def count_changed(explain_fn, names: list[str], n_repeat: int):
    """explain_fn()을 n_repeat번 호출해 한 번이라도 거절 사유가 바뀐 고객 인덱스를 반환"""
    first = [top_reasons(r, names) for r in explain_fn()]
    changed: set[int] = set()
    for _ in range(n_repeat - 1):
        cur = [top_reasons(r, names) for r in explain_fn()]
        changed |= {i for i, (a, b) in enumerate(zip(first, cur)) if a != b}
    return first, changed


def main() -> None:
    meta = json.loads((MODEL_DIR / f"model_meta_v{VERSION}.json").read_text(encoding="utf-8"))
    model = joblib.load(MODEL_DIR / meta["best_model"]["file"])
    cutoff = meta["best_model"]["approval_cutoff"]
    print(f"[check_shap] 모델 {meta['best_model']['file']} / 승인 기준값 {cutoff:.4f}")

    # model.py 와 같은 방식으로 분할 -> 같은 Test 세트
    data = CreditPreprocessor().process(pd.read_csv(DEFAULT_INPUT))
    X_test = data.X_test if hasattr(data.X_test, "iloc") else pd.DataFrame(data.X_test, columns=meta["feature_names"])
    X_train = data.X_train if hasattr(data.X_train, "iloc") else pd.DataFrame(data.X_train, columns=meta["feature_names"])
    names = list(X_test.columns)

    # 거절 고객 30명 샘플링 (부도확률 > 승인 기준값)
    p = model.predict_proba(X_test)[:, 1]
    rejected = np.where(p > cutoff)[0]
    pick = np.random.default_rng(SEED).choice(rejected, N_CUSTOMERS, replace=False)
    Xs = X_test.iloc[pick]
    print(f"[check_shap] Test 거절 고객 {len(rejected):,}명 중 {N_CUSTOMERS}명 샘플링\n")

    # ---- Before: KernelExplainer (기존 방식 재현) ----
    background = shap.sample(X_train, 100, random_state=SEED)
    kexp = shap.KernelExplainer(lambda z: model.predict_proba(pd.DataFrame(z, columns=names))[:, 1],
                                background)
    _, k_changed = count_changed(lambda: np.asarray(kexp.shap_values(Xs, nsamples=KERNEL_NSAMPLES, silent=True)),
                                 names, N_REPEAT_KERNEL)
    print(f"[Before] KernelExplainer ({N_REPEAT_KERNEL}회 반복): "
          f"거절 사유가 바뀐 고객 {len(k_changed)}/{N_CUSTOMERS}명")

    # ---- After: TreeExplainer (v1.2) ----
    clf = unwrap_model(model)
    explainer = shap.TreeExplainer(clf)
    first, t_changed = count_changed(lambda: _positive_class_shap(shap.TreeExplainer(clf), Xs)[0],
                                     names, N_REPEAT)
    print(f"[After ] TreeExplainer   ({N_REPEAT}회 반복): "
          f"거절 사유가 바뀐 고객 {len(t_changed)}/{N_CUSTOMERS}명  "
          f"{'PASS' if not t_changed else 'FAIL'}")

    # ---- 실제 판단 근거와 맞는지 ----
    sv, base = _positive_class_shap(explainer, Xs)
    pp = np.clip(model.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
    add_err = float(np.max(np.abs(sv.sum(axis=1) + base - np.log(pp / (1 - pp)))))
    print(f"\n[가법성] SHAP 합 + base value vs 모델 로그오즈 최대 오차 {add_err:.2e}  "
          f"{'PASS (설명 = 모델의 실제 계산)' if add_err < 1e-3 else 'FAIL'}")

    top1 = Counter(r[0] for r in first if r)
    print("\n[거절 사유 1순위 빈도] 한 변수에 너무 쏠리면 멘토링 피드백(무담보 한도 사용률) 재검토")
    for name, cnt in top1.most_common():
        print(f"  {name:45s} {cnt:2d}명 ({cnt / N_CUSTOMERS:.0%})")

    print("\n[예시 3명] 부도확률 / 거절 사유 top3 (변수=값)")
    for j in range(3):
        row = Xs.iloc[j]
        reasons = ", ".join(f"{n}={row[n]:.3g}" for n in first[j])
        print(f"  고객 {Xs.index[j]}: {pp[j]:.3f} / {reasons}")


if __name__ == "__main__":
    main()