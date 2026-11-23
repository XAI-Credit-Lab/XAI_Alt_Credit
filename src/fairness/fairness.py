"""
src/fairness/fairness.py
공정성 검증 모듈 (DI Ratio / Equalized Odds / 편향 완화)

[기존 유지] 시뮬레이터(src/data/simulator.py)가 import하는 이름은 그대로 둠
  AGE_BINS, AGE_LABELS, AGE_REFERENCE, gender_di, age_di, fmt_age_di

[추가] 공정성 검증
  1) DI Ratio        : 성별(여성/남성), 연령대(각 연령대/35~54)
  2) WARNING 판정    : 속성마다 따로 판정 (성별이 정상이어도 연령이 위반이면 경고)
  3) Equalized Odds  : ΔTPR, ΔFPR (집단 간 최대 차이)
  4) 편향 완화       : Threshold Adjustment (집단별 승인 기준값)
  5) API 연결        : model_fairness() -> (non_miti, miti)  ※ /fairness, /fairness/mitigation

값의 출처 (코드 안에 임의의 값을 넣지 않음)
  - 승인 규칙·승인율(70%)·승인 기준값 : models/model_meta_v*.json (model.py가 저장)
  - 모델·데이터                       : models/*.pkl, data/processed/{X,y,sensitive}_{val,test}.csv
  - 판정 기준(DI 0.8~1.25, EO 0.10, 집단별 부도 100건) : 프로젝트 요구사항 값. 아래 상수로만 존재

정의 (보고서에도 같은 문장으로 쓸 것)
  - 긍정 결과(favorable) = 승인.  승인 규칙: 부도확률 <= approval_cutoff
  - TPR = P(승인 | 실제 정상)   ← 갚을 사람을 얼마나 승인하는가
  - FPR = P(승인 | 실제 부도)   ← 못 갚을 사람을 얼마나 승인하는가
  - ΔTPR/ΔFPR = 집단 간 (최대 - 최소). 성별은 2집단이라 |여성 - 남성|과 같다.
  - 성별 코딩: gender 1 = 여성, 0 = 남성 (gender_di 정의와 동일)

한계 (보고서에 명시)
  - 성별·편향은 시뮬레이터가 가정한 값으로 만든 것이라 실제 대출 데이터의 결과가 아니다.
  - Threshold Adjustment는 집단별로 다른 기준값을 쓰므로, 실제 서비스에서는 보호 속성을
    판정에 쓰는 것이 허용되는지 법적·정책적 검토가 필요하다.
"""
from __future__ import annotations

import json
import re
import os
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 보호 속성: 연령대 구간 (시뮬레이터의 add_age_group이 이 값으로 age_group 생성)
# ----------------------------------------------------------------------
AGE_BINS   = [0, 34, 54, 200]
AGE_LABELS = ["~34", "35~54", "55+"]
AGE_REFERENCE = "35~54"   # 연령 DI의 기준(분모) 집단


# 시뮬레이터 리포트와 공정성 검증이 공통으로 사용하는 DI 계산 함수
def gender_di(df, approved):
    """성별 DI = 여성 승인율 / 남성 승인율"""
    approved = np.asarray(approved, dtype=float)
    g = df["gender"].to_numpy()
    return approved[g == 1].mean() / approved[g == 0].mean()


def age_di(df, approved, reference=AGE_REFERENCE):
    """연령대 DI = 각 연령대 승인율 / 기준 연령대(35~54) 승인율 (기준 집단 제외한 dict)"""
    rates = pd.Series(np.asarray(approved, dtype=float), index=df.index).groupby(df["age_group"]).mean()
    return {grp: rates[grp] / rates[reference] for grp in AGE_LABELS if grp != reference}


def fmt_age_di(di_dict):
    """age_di 결과를 출력용 문자열로 변환"""
    return ", ".join(f"{grp}/{AGE_REFERENCE} {v:.3f}" + ("(0.8 미만)" if v < 0.8 else "") for grp, v in di_dict.items())


# ----------------------------------------------------------------------
# 판정 기준 (프로젝트 요구사항 값)
# ----------------------------------------------------------------------
DI_LOWER = 0.8        # DI < 0.8  이면 WARNING
DI_UPPER = 1.25       # DI > 1.25 이면 WARNING
EO_THRESHOLD = 0.10   # ΔTPR, ΔFPR 이 값을 넘으면 기준 초과
MIN_DEFAULTS_PER_GROUP = 100   # 집단별 (테스트) 부도 건수가 이보다 적으면 통계가 불안정

DEFAULT_MITIGATION_BY = ("gender", "age_group")   # 완화 기법이 맞추는 집단 (성별 x 연령대)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_PATH = ROOT_DIR / "reports" / "fairness_report.json"


# ----------------------------------------------------------------------
# 1. 승인 판정
# ----------------------------------------------------------------------
def approve(proba, cutoff) -> np.ndarray:
    """승인 규칙: 부도확률 <= 승인 기준값  ->  1(승인).  model.py / /predict 와 같은 규칙"""
    return (np.asarray(proba, dtype=float) <= cutoff).astype(int)


# ----------------------------------------------------------------------
# 2. DI + 속성별 WARNING
# ----------------------------------------------------------------------
def _safe(x):
    """JSON에 넣을 수 있게 float 변환 (NaN/inf -> None)"""
    x = float(x)
    return x if np.isfinite(x) else None


def di_status(value) -> str:
    """DI 값 하나의 판정. 계산 불가(None)면 'UNKNOWN'"""
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    return "WARNING" if (value < DI_LOWER or value > DI_UPPER) else "OK"


def compute_di(sens: pd.DataFrame, approved) -> dict:
    """성별 DI 1개 + 연령대 DI(기준 집단 제외)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = _safe(gender_di(sens, approved))
        a = {k: _safe(v) for k, v in age_di(sens, approved).items()}
    return {"gender": g, "age": a}


def check_di_warnings(di: dict, tag: str = "") -> list[dict]:
    """
    속성마다 따로 판정한다. 성별이 정상이어도 연령이 위반이면 그 항목만 WARNING.
    반환: [{"attribute": "gender"|"age_group", "group": ..., "di": ..., "status": "OK"|"WARNING"|"UNKNOWN"}]
    위반 항목마다 logging.WARNING 을 남긴다.
    """
    rows = [("gender", "여성/남성", di["gender"])]
    rows += [("age_group", f"{k}/{AGE_REFERENCE}", v) for k, v in di["age"].items()]

    out = []
    for attr, grp, val in rows:
        st = di_status(val)
        out.append({"attribute": attr, "group": grp, "di": val, "status": st})
        prefix = f"[공정성{'/' + tag if tag else ''}] "
        if st == "WARNING":
            logger.warning("%s%s DI(%s) = %.4f -> 기준 %.2f~%.2f 벗어남",
                           prefix, attr, grp, val, DI_LOWER, DI_UPPER)
        elif st == "UNKNOWN":
            logger.warning("%s%s DI(%s) 계산 불가 (기준 집단 승인율이 0이거나 집단이 비어 있음)",
                           prefix, attr, grp)
    return out


# ----------------------------------------------------------------------
# 3. Equalized Odds
# ----------------------------------------------------------------------
def group_rates(y_true, approved, groups) -> pd.DataFrame:
    """
    집단별 표: n, 부도 건수, 승인율, TPR(정상 중 승인), FPR(부도 중 승인)
    y_true: 1 = 부도, 0 = 정상
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(approved).astype(int)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    rows = []
    for name in pd.unique(g):
        m = g == name
        good, bad = m & (y == 0), m & (y == 1)
        rows.append({
            "group": name,
            "n": int(m.sum()),
            "n_default": int(bad.sum()),
            "approval_rate": float(a[m].mean()),
            "tpr": float(a[good].mean()) if good.any() else float("nan"),
            "fpr": float(a[bad].mean()) if bad.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def equalized_odds(y_true, approved, groups) -> dict:
    """ΔTPR, ΔFPR = 집단 간 (최대 - 최소). 집단이 2개면 절대 차이와 같다."""
    t = group_rates(y_true, approved, groups)
    return {
        "tpr_diff": _safe(t["tpr"].max() - t["tpr"].min()),
        "fpr_diff": _safe(t["fpr"].max() - t["fpr"].min()),
        "table": t,
        "small_groups": t.loc[t["n_default"] < MIN_DEFAULTS_PER_GROUP, "group"].tolist(),
    }


# ----------------------------------------------------------------------
# 4. 편향 완화: Threshold Adjustment (집단별 승인 기준값)
# ----------------------------------------------------------------------
def _group_keys(sens: pd.DataFrame, by) -> pd.Series:
    by = list(by)
    return sens[by].astype(str).agg("|".join, axis=1)


def fit_group_cutoffs(p_val, sens_val: pd.DataFrame, target_rate: float,
                      by=DEFAULT_MITIGATION_BY) -> dict:
    """
    Validation 부도확률로 집단마다 '하위 target_rate' 지점을 승인 기준값으로 정한다.
    -> 모든 집단의 승인율이 target_rate 가 되도록 맞춤. (Test 정보는 쓰지 않음 = 누수 없음)
    target_rate 는 model_meta 의 approval_rate(승인율) 값을 그대로 쓴다.
    """
    keys = _group_keys(sens_val, by).to_numpy()
    p = np.asarray(p_val, dtype=float)
    return {k: float(np.quantile(p[keys == k], target_rate)) for k in pd.unique(keys)}


def apply_group_cutoffs(p, sens: pd.DataFrame, cutoffs: dict, fallback: float,
                        by=DEFAULT_MITIGATION_BY) -> np.ndarray:
    """집단별 기준값으로 승인 판정. Validation에 없던 집단은 전체 기준값(fallback) 사용"""
    keys = _group_keys(sens, by)
    cut = keys.map(cutoffs).astype(float).fillna(fallback).to_numpy()
    return (np.asarray(p, dtype=float) <= cut).astype(int)


# ----------------------------------------------------------------------
# 5. 한 번에 계산 (승인 결과 -> API 응답 dict)
# ----------------------------------------------------------------------
def fairness_summary(y_true, approved, sens: pd.DataFrame, tag: str = "") -> dict:
    """
    승인 결과 하나에 대한 공정성 요약.
    API가 쓰는 키(di_ratio, equalized_odds_tpr_diff, equalized_odds_fpr_diff)는 성별 기준.
    나머지(연령, 경고, 집단표)는 로그·보고서·대시보드용 추가 정보.
    """
    approved = np.asarray(approved).astype(int)
    di = compute_di(sens, approved)
    warns = check_di_warnings(di, tag)

    eo_g = equalized_odds(y_true, approved, sens["gender"].map({0: "남성", 1: "여성"}).to_numpy())
    eo_a = equalized_odds(y_true, approved, sens["age_group"].to_numpy())
    for name, eo in (("gender", eo_g), ("age_group", eo_a)):
        for key in ("tpr_diff", "fpr_diff"):
            v = eo[key]
            if v is not None and v > EO_THRESHOLD:
                logger.warning("[공정성%s] %s Equalized Odds Δ%s = %.4f > %.2f",
                               f"/{tag}" if tag else "", name, key[:3].upper(), v, EO_THRESHOLD)
        if eo["small_groups"]:
            logger.warning("[공정성%s] %s 집단 %s: 부도 건수 %d건 미만 (Δ값이 불안정)",
                           f"/{tag}" if tag else "", name, eo["small_groups"], MIN_DEFAULTS_PER_GROUP)

    return {
        # --- API 응답용 (성별 기준) ---
        "di_ratio": di["gender"],
        "equalized_odds_tpr_diff": eo_g["tpr_diff"],
        "equalized_odds_fpr_diff": eo_g["fpr_diff"],
        # --- 추가 정보 ---
        "age_di": di["age"],
        "age_equalized_odds_tpr_diff": eo_a["tpr_diff"],
        "age_equalized_odds_fpr_diff": eo_a["fpr_diff"],
        "approval_rate": float(approved.mean()),
        # 완화의 비용 확인용: 승인된 사람 중 실제 부도 비율 (낮을수록 좋음)
        "approved_default_rate": _safe(np.asarray(y_true).astype(int)[approved == 1].mean()) if approved.any() else None,
        "warnings": warns,                                   # 속성별 판정
        "overall_status": "WARNING" if any(w["status"] != "OK" for w in warns) else "OK",
        "gender_table": eo_g["table"].to_dict("records"),
        "age_table": eo_a["table"].to_dict("records"),
    }


# ----------------------------------------------------------------------
# 6. 저장된 모델·데이터로 실행 (API / CLI / 대시보드 공용)
# ----------------------------------------------------------------------
def _version_key(path: Path):
    """model_meta_v1.10.json > model_meta_v1.2.json 이 되도록 버전을 숫자로 비교 (문자열 정렬은 1.10 < 1.2)"""
    m = re.search(r"_v([\d.]+)\.json$", path.name)
    return tuple(int(x) for x in m.group(1).split(".") if x.isdigit()) if m else ()


def _latest_meta(model_dir: Path) -> Path:
    """사용할 model_meta 파일. 환경변수 MODEL_VERSION(예: 1.2)이 있으면 그 버전, 없으면 가장 높은 버전.
    ※ API(main.py)가 읽는 메타 버전과 같아야 /fairness 숫자와 /predict 승인 기준이 일치한다."""
    want = os.getenv("MODEL_VERSION")
    if want:
        f = model_dir / f"model_meta_v{want}.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} 가 없습니다 (환경변수 MODEL_VERSION={want})")
        return f
    files = sorted(model_dir.glob("model_meta_v*.json"), key=_version_key)
    if not files:
        raise FileNotFoundError(f"model_meta_v*.json 이 없습니다: {model_dir}  (python -m src.models.model 먼저 실행)")
    return files[-1]


def _read_split(data_dir: Path, name: str):
    X = pd.read_csv(data_dir / f"X_{name}.csv", index_col=0)
    y = pd.read_csv(data_dir / f"y_{name}.csv", index_col=0).iloc[:, 0]
    s = pd.read_csv(data_dir / f"sensitive_{name}.csv", index_col=0)
    if not (X.index.equals(y.index) and X.index.equals(s.index)):
        raise ValueError(f"{name} 세트의 X / y / sensitive 행 순서가 다릅니다. "
                         "preprocessor를 다시 실행해 세 파일을 함께 만드세요.")
    return X, y, s


def run_fairness(model_dir=MODEL_DIR, data_dir=DATA_DIR, by=DEFAULT_MITIGATION_BY) -> dict:
    """
    저장된 최종 모델로 Test 세트의 공정성을 계산한다.
      non_miti : 승인 기준값 1개 (model_meta 의 approval_cutoff) - 완화 전
      miti     : 집단별 승인 기준값 (Validation으로 결정)       - 완화 후
    반환: {"non_miti": {...}, "miti": {...}, "info": {...}}
    """
    import joblib
    from sklearn.metrics import roc_auc_score

    model_dir, data_dir = Path(model_dir), Path(data_dir)
    meta_path = _latest_meta(model_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    best = meta["best_model"]
    cutoff, rate = best["approval_cutoff"], best["approval_rate"]

    model = joblib.load(model_dir / best["file"])
    X_val, y_val, s_val = _read_split(data_dir, "val")
    X_te, y_te, s_te = _read_split(data_dir, "test")
    # 학습 때 피처 순서로 맞춤 (sensitive 전용 컬럼은 X에 없음). 없는 피처가 있으면 즉시 에러
    feats = meta.get("feature_names")
    if feats:
        missing = [c for c in feats if c not in X_val.columns]
        if missing:
            raise KeyError(f"X_val에 모델 피처가 없습니다: {missing}")
        X_val, X_te = X_val[feats], X_te[feats]
    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    non_miti = fairness_summary(y_te, approve(p_te, cutoff), s_te, tag="완화 전")

    cuts = fit_group_cutoffs(p_val, s_val, rate, by)
    miti = fairness_summary(y_te, apply_group_cutoffs(p_te, s_te, cuts, cutoff, by), s_te, tag="완화 후")

    info = {
        "model": best["algorithm"], "model_file": best["file"], "model_version": meta.get("model_version"),
        "trained_at": meta.get("trained_at"),
        "approval_rule": "부도확률 <= approval_cutoff -> 승인",
        "approval_cutoff": cutoff, "target_approval_rate": rate,
        "mitigation": "Threshold Adjustment", "mitigation_by": list(by),
        "group_cutoffs": cuts,
        "test_auc": float(roc_auc_score(y_te, p_te)),   # 점수는 그대로라 완화 전/후 AUC 동일
        "n_test": int(len(y_te)),
    }
    return {"non_miti": non_miti, "miti": miti, "info": info}


@lru_cache(maxsize=1)
def _cached_run() -> dict:
    return run_fairness()


def model_fairness():
    """
    model.py 의 fairness() 가 호출하는 함수.  main.py 가 기대하는 순서: (non_miti, miti)
        non_miti, miti = model.fairness()
    """
    r = _cached_run()
    return r["non_miti"], r["miti"]


def _json_safe(o):
    """NaN/inf -> None, numpy 타입 -> 파이썬 타입 (FastAPI JSON 응답은 NaN을 허용하지 않음)"""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _safe(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def model_fairness_detail() -> dict:
    """
    /fairness/detail 용: 대시보드가 HTTP로만 받아 갈 수 있도록 상세 결과 전체를 반환.
    (연령 DI, 속성별 판정, 집단표, 완화 기준값, 판정 기준값)
    """
    r = _cached_run()
    return _json_safe({
        "thresholds": {"di_lower": DI_LOWER, "di_upper": DI_UPPER, "eo_threshold": EO_THRESHOLD,
                       "min_defaults_per_group": MIN_DEFAULTS_PER_GROUP},
        "info": r["info"], "non_miti": r["non_miti"], "miti": r["miti"],
    })


def comparison_table(result: dict | None = None) -> pd.DataFrame:
    """보고서용 '완화 전/후 공정성 지표 비교표' (요구사항: 리포트에 비교 표 포함)"""
    r = result or _cached_run()
    b, a = r["non_miti"], r["miti"]
    rows = [("성별 DI (여/남)", "di_ratio", f"{DI_LOWER}~{DI_UPPER}"),
            ("성별 ΔTPR", "equalized_odds_tpr_diff", f"<= {EO_THRESHOLD}"),
            ("성별 ΔFPR", "equalized_odds_fpr_diff", f"<= {EO_THRESHOLD}"),
            ("연령 ΔTPR", "age_equalized_odds_tpr_diff", f"<= {EO_THRESHOLD}"),
            ("연령 ΔFPR", "age_equalized_odds_fpr_diff", f"<= {EO_THRESHOLD}")]
    out = [{"지표": n, "완화 전": b[k], "완화 후": a[k], "기준": std} for n, k, std in rows]
    for grp in AGE_LABELS:
        if grp != AGE_REFERENCE:
            out.append({"지표": f"연령 DI {grp}/{AGE_REFERENCE}", "완화 전": b["age_di"][grp],
                        "완화 후": a["age_di"][grp], "기준": f"{DI_LOWER}~{DI_UPPER}"})
    out.append({"지표": "전체 승인율", "완화 전": b["approval_rate"], "완화 후": a["approval_rate"], "기준": "-"})
    out.append({"지표": "승인자 부도율", "완화 전": b["approved_default_rate"], "완화 후": a["approved_default_rate"], "기준": "-"})
    return pd.DataFrame(out)


def write_report(path=REPORT_PATH, **kw) -> Path:
    """대시보드·보고서용 JSON 저장"""
    r = run_fairness(**kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# 7. 실행: python -m src.fairness.fairness
# ----------------------------------------------------------------------
def _print(r: dict) -> None:
    info = r["info"]
    print(f"모델 {info['model']} v{info['model_version']} (학습 {info['trained_at']}), Test {info['n_test']:,}행, "
          f"AUC {info['test_auc']:.4f}")
    print(f"승인 규칙: {info['approval_rule']} (목표 승인율 {info['target_approval_rate']:.0%}, "
          f"기준값 {info['approval_cutoff']:.4f})")
    print(f"완화: {info['mitigation']} by {info['mitigation_by']}\n")
    print(f"{'':16s}{'완화 전':>12s}{'완화 후':>12s}")
    rows = [("성별 DI", "di_ratio"), ("성별 ΔTPR", "equalized_odds_tpr_diff"), ("성별 ΔFPR", "equalized_odds_fpr_diff"),
            ("연령 ΔTPR", "age_equalized_odds_tpr_diff"), ("연령 ΔFPR", "age_equalized_odds_fpr_diff"),
            ("승인율", "approval_rate"), ("승인자 부도율", "approved_default_rate")]
    for label, k in rows:
        print(f"{label:14s}{r['non_miti'][k]:>12.4f}{r['miti'][k]:>12.4f}")
    for grp in AGE_LABELS:
        if grp == AGE_REFERENCE:
            continue
        print(f"{'연령 DI ' + grp:14s}{r['non_miti']['age_di'][grp]:>12.4f}{r['miti']['age_di'][grp]:>12.4f}")
    print(f"\n판정: 완화 전 {r['non_miti']['overall_status']} / 완화 후 {r['miti']['overall_status']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = run_fairness()
    _print(res)
    out = write_report()
    print(f"\n리포트 저장: {out}")
    csv_path = out.with_name("fairness_comparison.csv")
    comparison_table(res).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"비교표 저장 (보고서용): {csv_path}")
