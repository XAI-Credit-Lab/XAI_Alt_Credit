"""
src/data/audit_data.py
데이터 클렌징 점검 (수정 전 진단용) - 데이터와 코드는 아무것도 바꾸지 않고 '확인'만 한다.

멘토링 3가지 관점
  1) 라벨-값 일치 : 컬럼 이름(정의)과 실제 들어 있는 값이 맞는지
                    예) DebtRatio(비율)인데 금액이 들어 있는 경우, 횟수인데 소수점, 비율인데 1 초과
  2) 결측치       : 결측이 어떻게 채워져 있는지 (NaN 그대로 / 0 / 중앙값 / 96·98 같은 코드값)
                    클렌징 전 단계에서 결측이 다른 값으로 숨어 있을 수 있음
  3) 단위·스케일  : 각 값의 실제 단위가 무엇인지, 스케일링이 어떻게 되어 있는지
                    한 컬럼 안에 단위가 섞이면 XGBoost 분할·SHAP·설명 문장이 모두 왜곡됨

실행 (프로젝트 루트, .venv 켠 상태):
  python -m src.data.audit_data --raw data/raw/cs-training.csv
  python -m src.data.audit_data --raw data/raw/cs-training.csv > reports/data_audit_v1.2.txt   # 결과 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
INCOME = "MonthlyIncome"
DEBT = "DebtRatio"
REVOLV = "RevolvingUtilizationOfUnsecuredLines"
PAST_DUE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# 컬럼별 정의: (정의, 단위, 종류, 정상 하한, 정상 상한)
#   종류: ratio(비율) / count(횟수·개수, 정수) / amount(금액) / binary(0·1)
#   GMSC 원본은 Kaggle 'Data Dictionary.xls', 시뮬레이터 컬럼은 미션 정의 기준
SPEC = {
    TARGET:                              ("2년 내 90일 이상 연체 여부 (타깃)", "0/1", "binary", 0, 1),
    REVOLV:                              ("무담보 회전 한도 잔액 / 신용 한도 합계", "비율", "ratio", 0, 1),
    "age":                               ("나이", "세", "count", 18, 110),
    PAST_DUE[0]:                         ("최근 2년 30~59일 연체", "횟수", "count", 0, 24),
    PAST_DUE[1]:                         ("최근 2년 60~89일 연체", "횟수", "count", 0, 12),
    PAST_DUE[2]:                         ("90일 이상 연체", "횟수", "count", 0, 20),
    DEBT:                                ("월 부채상환액·생활비 / 월 총소득", "비율", "ratio", 0, 1),
    INCOME:                              ("월 소득", "금액(달러/월)", "amount", 0, None),
    "NumberOfOpenCreditLinesAndLoans":   ("보유 대출·신용카드 수", "개", "count", 0, None),
    "NumberRealEstateLoansOrLines":      ("부동산 대출 수", "개", "count", 0, None),
    "NumberOfDependents":                ("부양 가족 수", "명", "count", 0, 20),
    # 시뮬레이터 대안 변수 (단위는 simulator.py 생성 로직과 맞는지 꼭 대조)
    "card_history_months":               ("신용카드 이용 기간", "개월", "count", 0, 360),
    "telecom_payment_rate":              ("통신비 정상 납부율", "비율", "ratio", 0, 1),
    "utility_payment_rate":              ("공과금 납부율", "비율", "ratio", 0, 1),
    "spending_consistency":              ("소비 일관성 점수", "점(0~100)", "score", 0, 100),
    "regular_payment_count":             ("정기 결제 건수", "건", "count", 0, None),
    "app_login_frequency":               ("앱 로그인 빈도", "회 (기간 확인 필요)", "count", 0, None),
    "is_thin_filer":                     ("씬파일러 여부", "0/1", "binary", 0, 1),
}
SENTINELS = [-1, 96, 98, 99, 999, 9999]       # 결측·이상을 숫자로 기록하는 흔한 코드값
Q = [0.01, 0.5, 0.99, 0.999]


def title(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def income_group(df: pd.DataFrame) -> pd.Series:
    inc = df[INCOME]
    return pd.Series(np.select([inc.isna(), inc == 0, inc == 1, inc > 1],
                               ["소득 결측", "소득 = 0", "소득 = 1", "소득 > 1"], "기타"), index=df.index)


# ======================================================================
# 1) 라벨-값 일치
# ======================================================================
def check_label_value(raw: pd.DataFrame, sim: pd.DataFrame) -> None:
    title("1) 라벨-값 일치: 컬럼 이름(정의)과 실제 값이 맞는가")

    for name, df in [("원본", raw), ("시뮬레이터 결과", sim)]:
        rows = []
        for c in [c for c in df.columns if c in SPEC]:
            desc, unit, kind, lo, hi = SPEC[c]
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            r = {"컬럼": c, "정의": desc, "단위": unit,
                 "정상범위": f"{lo}~{'' if hi is None else hi}",
                 "실제 min": s.min(), "실제 max": s.max(),
                 "하한미만": int((s < lo).sum()),
                 "상한초과": int((s > hi).sum()) if hi is not None else 0}
            r["정수아님"] = int((s % 1 != 0).sum()) if kind in ("count", "binary") else "-"
            rows.append(r)
        unknown = [c for c in df.select_dtypes("number").columns if c not in SPEC]
        print(f"\n[{name}] (행 {len(df):,})")
        print(pd.DataFrame(rows).set_index("컬럼").to_string(float_format=lambda v: f"{v:,.3f}"))
        if unknown:
            print(f"  ※ 정의가 등록되지 않은 수치형 컬럼 -> 단위 직접 확인 필요: {unknown}")

    print("\n해석 가이드")
    print("  - 비율(ratio)인데 상한초과가 많음  -> 비율이 아닌 값(금액 등)이 섞였을 가능성")
    print("  - 횟수(count)인데 정수아님 > 0       -> 다른 단위이거나 이미 가공(평균·대체)된 값")
    print("  - 하한미만(음수 등)                  -> 코드값(-1 등)으로 결측을 기록했을 가능성 (2번에서 확인)")

    # 비율 컬럼은 '어떤 행에서' 정의가 깨지는지 소득 상태별로 확인 (DebtRatio = 부채 / 소득)
    if INCOME in sim:
        g = income_group(sim)
        print("\n[소득 상태별 비율 컬럼 분포]  중앙값이 수백~수천이면 그 그룹은 '비율'이 아니라 '금액'")
        for c in [DEBT, REVOLV]:
            if c in sim:
                t = sim.groupby(g)[c].describe(percentiles=[.5, .99])[["count", "min", "50%", "99%", "max"]]
                t["1 초과 비율"] = sim.groupby(g)[c].apply(lambda s: (s > 1).mean())
                print(f"\n  {c}")
                print(t.rename(columns={"50%": "중앙값"}).to_string(float_format=lambda v: f"{v:,.3f}"))
        print(f"\n  소득 상태별 부도율: {sim.groupby(g)[TARGET].mean().round(4).to_dict()}")

    # 시뮬레이터가 붙인 라벨이 규칙과 일치하는지
    if "is_thin_filer" in sim:
        cols = [c for c in PAST_DUE if c in sim]
        rule = sim[cols].isna().sum(axis=1) >= 2
        if "card_history_months" in sim:
            rule |= sim["card_history_months"] < 12
        label = sim["is_thin_filer"].astype(str).str.lower().isin(["1", "true", "1.0"])
        print(f"\n[씬파일러 라벨] 비율 {label.mean():.2%} | 규칙(연체 컬럼 2개 이상 결측 OR 카드 이력 < 12개월) "
              f"재계산과 불일치 {int((rule != label).sum()):,}행")

    added = [c for c in sim.columns if c not in raw.columns]
    print(f"[시뮬레이터가 추가한 컬럼] {added}")
    if len(raw) == len(sim):
        common = [c for c in raw.columns if c in sim.columns]
        changed = {c: int((raw[c].fillna(-999) != sim[c].fillna(-999)).sum()) for c in common}
        print(f"[원본 대비 값이 바뀐 컬럼 (행 수)] {({c: n for c, n in changed.items() if n}) or '없음'}")


# ======================================================================
# 2) 결측치
# ======================================================================
def check_missing(raw: pd.DataFrame, sim: pd.DataFrame, processed_dir: Path) -> None:
    title("2) 결측치: 어디에 있고, 무엇으로 채워져 있는가 (NaN / 0 / 중앙값 / 코드값)")

    # 2-1 결측 현황
    tbl = pd.DataFrame({"원본 NaN": raw.isna().sum(), "원본 %": raw.isna().mean() * 100}).join(
        pd.DataFrame({"시뮬 NaN": sim.isna().sum(), "시뮬 %": sim.isna().mean() * 100}), how="outer")
    print("[2-1 NaN 현황]")
    print(tbl[(tbl.fillna(0)[["원본 NaN", "시뮬 NaN"]] > 0).any(axis=1)].to_string(float_format=lambda v: f"{v:,.2f}"))

    # 2-2 숨은 결측: 코드값·0
    print("\n[2-2 숨은 결측: 숫자로 기록된 코드값 / 0]")
    rows = []
    for c in [c for c in sim.columns if c in SPEC and SPEC[c][2] != "binary"]:
        s = sim[c]
        r = {"컬럼": c, **{f"={v}": int((s == v).sum()) for v in SENTINELS}, "=0": int((s == 0).sum())}
        if any(v for k, v in r.items() if k not in ("컬럼", "=0")) or SPEC[c][2] in ("amount", "ratio"):
            rows.append(r)
    print(pd.DataFrame(rows).set_index("컬럼").to_string())
    print("  - 연체 횟수의 96/98   : GMSC '기록 이상' 코드 (현재 전처리에서 past_due_code_flag 후 결측 처리)")
    print("  - 금액·비율 컬럼의 0  : 진짜 0인지 '모름'을 0으로 기록한 것인지 구분 필요 (예: 소득 = 0)")

    # 2-3 원본 NaN 행이 무엇으로 바뀌었나 (원본 -> 시뮬레이터)
    print("\n[2-3 원본에서 NaN이던 행이 시뮬레이터 결과에서 무엇이 되었나]")
    targets = [c for c in raw.columns if raw[c].isna().any() and c in sim]
    if len(raw) == len(sim) and targets:
        rows = []
        for c in targets:
            med, s = raw[c].median(), sim.loc[raw[c].isna(), c]
            rows.append({"컬럼": c, "원본 NaN": int(raw[c].isna().sum()), "원본 중앙값": med,
                         "→ NaN 그대로": int(s.isna().sum()), "→ 0": int((s == 0).sum()),
                         "→ 중앙값": int((s == med).sum()),
                         "→ 기타 값": int((s.notna() & (s != 0) & (s != med)).sum())})
        print(pd.DataFrame(rows).set_index("컬럼").to_string())
        print("  - 'NaN 그대로'가 원본 NaN과 같아야 정상 (그래야 전처리의 결측 규칙·지시변수가 작동)")
        print("  - '→ 0' 또는 '→ 중앙값'이 있으면 시뮬레이터 단계에서 이미 채워진 것 -> 전처리 규칙이 이 행을 못 찾음")
    else:
        print("  원본과 행 수가 달라 행 단위 추적 생략 -> 2-1의 NaN 수 비교로 판단")

    # 2-4 전처리 후 (save_splits 결과): raw_* 는 변환 전, X_* 는 변환 후
    raw_tr, x_tr = processed_dir / "raw_train.csv", processed_dir / "X_train.csv"
    print("\n[2-4 전처리 후 결측이 어떻게 채워졌나 (raw_train.csv -> X_train.csv)]")
    if raw_tr.exists() and x_tr.exists():
        rt = pd.read_csv(raw_tr, index_col=0)
        xt = pd.read_csv(x_tr, index_col=0).loc[rt.index]
        for c in [c for c in rt.columns if rt[c].isna().any() and c in xt]:
            nan_rows = rt[c].isna()
            filled = xt.loc[nan_rows, c]
            ind = f"missingindicator_{c}"
            ind_ok = ind in xt and int((xt.loc[nan_rows, ind] > xt[ind].min()).sum()) == int(nan_rows.sum())
            how = "단일 상수(중앙값 대체)" if filled.nunique() == 1 else f"값 {filled.nunique()}종류 (확인 필요)"
            print(f"  {c:40s} 결측 {int(nan_rows.sum()):>7,}행 -> {how}, 남은 NaN {int(filled.isna().sum())}, "
                  f"지시변수 {'있음' if ind_ok else '없음'}")
    else:
        print(f"  {raw_tr.name} / {x_tr.name} 없음 -> 생략")


# ======================================================================
# 3) 단위·스케일링
# ======================================================================
def check_unit_scale(sim: pd.DataFrame, preprocessor_path: str | None) -> None:
    title("3) 단위·스케일링: 값의 실제 단위와 스케일링 방식")

    num = sim.select_dtypes("number").drop(columns=[TARGET], errors="ignore")
    d = num.describe(percentiles=Q).T[["min", "1%", "50%", "mean", "99%", "99.9%", "max"]]
    d.insert(0, "단위", [SPEC.get(c, ("", "(미등록)"))[1] for c in d.index])
    d["평균/중앙값"] = d["mean"] / d["50%"].replace(0, np.nan)
    d["max/99%"] = d["max"] / d["99%"].replace(0, np.nan)
    print("[3-1 원래 단위의 분포]")
    print(d.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("  - 평균/중앙값이 1에서 크게 벗어남 -> 꼬리가 긴 분포 (평균은 대표값으로 부적절, 중앙값 사용)")
    print("  - max/99% 가 수십~수천 배       -> 소수 극단값 또는 단위가 다른 값이 섞임")

    if not (preprocessor_path and Path(preprocessor_path).exists()):
        print(f"\n[3-2] 전처리기 {preprocessor_path} 없음 -> 생략")
        return
    import joblib
    pre = joblib.load(preprocessor_path)
    num_pipe = pre.transformer_.named_transformers_["num"]
    print(f"\n[3-2 현재 전처리기 {Path(preprocessor_path).name}의 스케일링 순서]")
    print("  " + " -> ".join(f"{k}({type(v).__name__})" for k, v in num_pipe.named_steps.items()))
    clip, imp, sc = (num_pipe.named_steps[k] for k in ("clip", "imputer", "scaler"))
    names = list(getattr(imp, "feature_names_in_", pre.numeric_cols_))
    t = pd.DataFrame({"단위": [SPEC.get(n, ("", "(미등록)"))[1] for n in names],
                      "clip 상한(99%)": clip.upper_, "imputer 중앙값": imp.statistics_}, index=names)
    t = t.join(pd.DataFrame({"scaler 평균": sc.mean_, "scaler 표준편차": sc.scale_}, index=sc.feature_names_in_))
    print(t.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("  - 'clip 상한'이 그 단위에서 말이 안 되는 값(예: 비율인데 수천)  -> 상한 처리가 사실상 작동 안 함")
    print("  - 'scaler 평균'이 'imputer 중앙값'과 크게 다름               -> 표준화 기준이 극단값에 끌려감")
    print("  - 거절 사유 문장의 '평균'은 'scaler 평균' 열 (DebtRatio 308의 출처)")
    print("  - 트리 모델(XGBoost)은 스케일 자체엔 강하지만, 한 컬럼에 단위가 섞이면 분할 기준이 왜곡되고")
    print("    SHAP 기여도·Dependence Plot·설명 문장도 그 섞인 값 기준으로 계산됨")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="데이터 클렌징 점검 (라벨-값 일치 / 결측치 / 단위·스케일)")
    p.add_argument("--raw", required=True, help="GMSC 원본 CSV (cs-training.csv)")
    p.add_argument("--sim", default=str(root / "data" / "processed" / "결과_시나리오.csv"), help="시뮬레이터 결과 CSV")
    p.add_argument("--pre", default=str(root / "artifacts" / "preprocessor_v1.2.joblib"), help="현재 전처리기")
    a = p.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    raw, sim = load(a.raw), load(a.sim)
    check_label_value(raw, sim)
    check_missing(raw, sim, Path(a.sim).parent)
    check_unit_scale(sim, a.pre)


if __name__ == "__main__":
    main()