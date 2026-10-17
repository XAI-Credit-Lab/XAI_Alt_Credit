import logging # loader의 로그(컬럼명·타입·결측치) 출력용
from pathlib import Path #파일 경로를 다루는 객체
import numpy as np # 난수 생성, 수치 계산
import pandas as pd # CSV/데이터프레임 처리

if __package__:
    from .loader import load_gmsc # 모듈 실행 시 같은 패키지의 로더 사용
else:
    from loader import load_gmsc # 파일 직접 실행 시 같은 폴더의 로더 사용


# 대안 변수 생성 : telecom_payment_rate, utility_payment_rate, spending_consistency
#   regular_payment_count, app_login_frequency
ALT_COLS = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
            "regular_payment_count", "app_login_frequency"]

# 연체 관련 컬럼 (씬파일러 판정에 사용)
DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

TRAD_INFO_COLS = DELINQ_COLS + [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]
THIN_INFO_LOSS = 0.8   # 씬파일러의 기존 변수 값 중 80%를 정보 없는 값으로 교체 (가정값)

NON_FEATURE_COLS = ["approved", "gender", "persona", "age", "age_group"]

CONTROL_BIAS_RATIO  = 0.0
SCENARIO_BIAS_RATIO = 1.0

AGE_BINS   = [0, 34, 54, 200]
AGE_LABELS = ["~34", "35~54", "55+"]
AGE_REFERENCE = "35~54"   # [3차-2] 연령 DI의 기준(분모) 집단

YOUNG_ADULT_AGE = (15, 29)

MIN_TEST_DEFAULTS = 100
TEST_SIZE = 0.15

BIAS_PENALTY = {"female": 1.0, "~34": 1.0, "55+": 0.5}

'''
# 기존 코드
FEATURE_BIAS = {
    "telecom_payment_rate":  {"female": 0.75, "55+": 0.2},               # 통신비가 배우자·자녀 명의라 기록 불리
    "utility_payment_rate":  {"female": 0.75, "~34": 0.5},                    # 공과금이 본인 명의 아님(주부·부모와 동거)
    "spending_consistency":  {"female": 0.6, "~34": 0.4, "55+": 0.3},  # 경력단절·사회초년생·은퇴로 소비 변동
    "regular_payment_count": {"~34": -0.3},                                         # 부도와 +상관이라 불리하게 = 늘림
    "app_login_frequency":   {"55+": 0.4},                                    # 고령층 디지털 격차 (0.8에서 축소)
}

RHO = {
    "telecom_payment_rate":  -0.35,   # 2차 -0.40 / 최초 -0.31
    "utility_payment_rate":  -0.33,   # 2차 -0.38 / 최초 -0.31
    "spending_consistency":  -0.35,   # 2차 -0.41 / 최초 -0.32
    "regular_payment_count": 0.48,    # 2차 0.56 / 최초 0.45 (Poisson 변환에서 상관이 많이 약해짐)
    "app_login_frequency":   -0.57,   # 2차 -0.69 / 최초 -0.53 (Poisson 변환에서 상관이 많이 약해짐)
}
'''

# TODO : FEATURE_BIAS와 RHO 값을 기존 코드로 할지 새로운 코드로 할지 검토 부탁!

# 대안변수 편향 (가정값): 불리한 집단의 잠재값을 몇 표준편차 낮출지. 실제 = bias_ratio × 값
# 부도와 무관하게 그룹만 보고 낮춤 → 모델이 부도 라벨로 학습해도 이 편향을 배우게 됨
# female 편향은 5개 변수 모두에 같은 방향(불리한 쪽)으로 넣음
# → 일부 변수에만 넣으면 모델이 "이 변수들만 낮은 패턴"으로 성별을 알아채 편향을 보정해 버림
'''새로운 코드'''

FEATURE_BIAS = {
    "telecom_payment_rate":  {"female": 0.5, "55+": 0.2},               # 통신비가 배우자·자녀 명의라 기록 불리
    "utility_payment_rate":  {"female": 0.5, "~34": 0.5},               # 공과금이 본인 명의 아님(주부·부모와 동거)
    "spending_consistency":  {"female": 0.5, "~34": 0.4, "55+": 0.3},   # 경력단절·사회초년생·은퇴로 소비 변동
    "regular_payment_count": {"female": -0.5, "~34": -0.3},             # 부도와 +상관이라 불리하게 = 늘림 (가계 고정지출을 본인 명의로 결제)
    "app_login_frequency":   {"female": 0.5, "55+": 0.4},               # 가계 금융을 배우자 명의 앱으로 관리 / 고령층 디지털 격차
}

RHO = {
    "telecom_payment_rate":  -0.34,
    "utility_payment_rate":  -0.32,
    "spending_consistency":  -0.35,
    "regular_payment_count":  0.48,   # Poisson 변환에서 상관이 많이 약해져 크게 잡음
    "app_login_frequency":   -0.59,   # Poisson 변환에서 상관이 많이 약해져 크게 잡음
}

# 페르소나 코드 (CSV 저장용): 1: 일반, 2: 사회초년생, 3: 주부
PERSONA_CODE = {"일반": 1, "사회초년생": 2, "주부": 3}

# 씬파일러 선택 가중치 (가정값): 사회초년생·주부가 일반보다 5배 잘 뽑힘
PERSONA_WEIGHT = {PERSONA_CODE["일반"]: 1.0,
                  PERSONA_CODE["사회초년생"]: 5.0,
                  PERSONA_CODE["주부"]: 5.0}


# t = 타깃, rho = 원하는 상관계수, f = 모든 변수가 공유하는 공통 요인(대안변수끼리 상관 만듦)
# rng = 난수 생성기, lam = 공통 요인이 노이즈에서 차지하는 비중
def _corr_latent(t, rho, f, rng, lam=0.3):
    s = (t - t.mean()) / t.std()                                           # 표준화한 타깃 (평균 0, 분산 1)
    e = np.sqrt(lam) * f + np.sqrt(1 - lam) * rng.standard_normal(len(t))  # 노이즈 (lam 비중은 공통 요인 f, 나머지는 개별 난수)
    return rho * s + np.sqrt(1 - rho**2) * e                               # 타깃과의 상관 ≈ rho

def _group_shift(df, feature):
    """해당 대안변수에 대해 각 사람이 받는 편향 크기(표준편차 단위)"""
    shift = np.zeros(len(df))
    for group, v in FEATURE_BIAS.get(feature, {}).items():
        mask = (df["gender"] == 1) if group == "female" else (df["age_group"] == group)
        shift += mask.to_numpy() * v
    return shift

def is_thin_filer(df):
    """씬파일러 판정: 연체 관련 컬럼 2개 이상 결측 또는 카드 거래이력 12개월 미만"""
    delinq_missing = df[DELINQ_COLS].isna().sum(axis=1) >= 2
    short_history = df["card_history_months"] < 12
    return delinq_missing | short_history

# 나이를 AGE_BINS·AGE_LABELS 기준 3구간(~34, 35~54, 55+)으로 잘라 문자열로 저장
def add_age_group(df):
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df

# 주부 판정·여성 확률 상향에 같이 쓰는 조건 (add_gender·add_persona가 공유)
def _homemaker_conditions(df):
    income_missing = df["MonthlyIncome"].isna() # 월 소득 결측 여부
    has_dependents = df["NumberOfDependents"].fillna(0) >= 1 # 부양가족 1명 이상 (결측은 0으로 간주)
    mid_age = df["age"].between(30, 59) # 30~59세 여부
    return income_missing, has_dependents, mid_age

def add_gender(df, rng):
    income_missing, has_dependents, mid_age = _homemaker_conditions(df)

    # 여성일 확률: 기본 50%, 소득 결측+부양가족이면 상향 (가정값, 보고서에 근거 명시)
    p_female = np.full(len(df), 0.50)
    p_female[(income_missing & has_dependents & mid_age).to_numpy()] = 0.80  # 소득 결측, 부양가족있음, 30~59세를 모두 만족하면 80%로 올림
    df["gender"] = (rng.random(len(df)) < p_female).astype(int)   # 0: 남, 1: 여
    return df

def add_persona(df):
    income_missing, has_dependents, mid_age = _homemaker_conditions(df)

    # 페르소나 라벨 (숫자 코드)
    df["persona"] = PERSONA_CODE["일반"] # 기본값: 일반(1)
    df.loc[df["age"].between(*YOUNG_ADULT_AGE), "persona"] = PERSONA_CODE["사회초년생"] # 통계청 청년층(15~29세)은 사회초년생(2)
    df.loc[(df["gender"] == 1) & mid_age & has_dependents
           & (income_missing | (df["MonthlyIncome"] < 2000)), "persona"] = PERSONA_CODE["주부"] # 여성, 30~59세, 부양가족 있음, 소득 결측 또는 2000 미만일 경우 주부(3)
    return df


def inject_thin_filers(df, rng, thin_filer_ratio, thin_info_loss=THIN_INFO_LOSS):
    n = len(df)
    k = int(round(n * thin_filer_ratio))

    # 페르소나별 선택 가중치 (가정값): 사회초년생·주부가 일반보다 5배 잘 뽑힘
    w = df["persona"].map(PERSONA_WEIGHT).to_numpy()
    assert not np.isnan(w).any(), f"PERSONA_WEIGHT에 없는 persona 값: {df['persona'].unique()}"
    idx = rng.choice(n, size=k, replace=False, p=w / w.sum())
    selected = np.zeros(n, dtype=bool)
    selected[idx] = True

    # 카드 거래이력(개월): 일반은 나이에 비례, 씬파일러는 0~11개월
    months = np.clip((df["age"].to_numpy() - 18) * 12 * rng.uniform(0.3, 0.9, n), 12, 360)
    months = months.astype(int)
    months[idx] = rng.integers(0, 12, size=k)
    df["card_history_months"] = months

    donors = np.flatnonzero(~selected)
    for col in TRAD_INFO_COLS:
        replace = selected & (rng.random(n) < thin_info_loss)
        donor_vals = df[col].to_numpy()[rng.choice(donors, size=replace.sum())]
        df.loc[replace, col] = donor_vals

    # 씬파일러의 60%는 연체 컬럼 3개 중 2개를 결측 처리
    df[DELINQ_COLS] = df[DELINQ_COLS].astype(float)
    mask_rows = selected & (rng.random(n) < 0.6)
    keep = rng.integers(0, len(DELINQ_COLS), n)       # 결측시키지 않고 남길 컬럼
    for j, col in enumerate(DELINQ_COLS):
        df.loc[mask_rows & (keep != j), col] = np.nan

    df["is_thin_filer"] = is_thin_filer(df).astype(int)
    return df


def add_approval_label(df, t, f, rng, bias_ratio, approval_rate=0.7):
    """과거 승인 결과(approved)를 만든다. bias_ratio가 클수록 그룹 간 승인율 격차가 커진다."""
    base = _corr_latent(t, -0.6, f, rng)              # 부도자일수록 낮은 신용 점수

    age_pen = df["age_group"].map(BIAS_PENALTY).fillna(0).to_numpy()   # ~34: 1.0, 55+: 0.5, 35~54: 0
    penalty = (df["gender"] == 1).to_numpy() * BIAS_PENALTY["female"] + age_pen
    score = base - bias_ratio * penalty

    # 전체 승인율은 approval_rate로 고정하고, 그룹별 승인율만 달라지게 함
    thr = np.quantile(score, 1 - approval_rate)
    df["approved"] = (score >= thr).astype(int)
    return df



# 전체 파이프라인: 성별·페르소나·연령대 → 대안 변수 생성(편향 포함) → 씬파일러 판정 → 승인 여부 생성
def add_alt_features(df, seed=42, thin_filer_ratio=0.2, bias_ratio=0.0, approval_rate=0.7,
                     thin_info_loss=THIN_INFO_LOSS):
    assert 0 <= thin_filer_ratio <= 1 and 0 <= bias_ratio <= 1 and 0 <= thin_info_loss <= 1
    rng = np.random.default_rng(seed)
    df = df.copy()
    t = df["SeriousDlqin2yrs"].to_numpy()
    f = rng.standard_normal(len(df))                  # 공통 '금융 성실도' 요인

    df = add_gender(df, rng)                          # 성별 (대안변수 편향에 필요해서 먼저 실행)
    df = add_age_group(df)                            # 연령대 (편향 주입·리포트 공통 구간)
    df = add_persona(df)                              # 페르소나 라벨 (성별이 먼저 있어야 함)

    def z(feature, rho):                              # 부도와 상관된 잠재값 - 그룹 편향(bias_ratio × FEATURE_BIAS)
        return _corr_latent(t, rho, f, rng) - bias_ratio * _group_shift(df, feature)

    df["telecom_payment_rate"]  = np.clip(0.88 + 0.10 * z("telecom_payment_rate", RHO["telecom_payment_rate"]), 0, 1)   # 평균 0.88, 표준편차 약 0.1, 부도와 음의 상관
    df["utility_payment_rate"]  = np.clip(0.90 + 0.09 * z("utility_payment_rate", RHO["utility_payment_rate"]), 0, 1)   # 평균 0.90, 표준편차 약 0.09, 음의 상관
    df["spending_consistency"]  = np.clip(70 + 15 * z("spending_consistency", RHO["spending_consistency"]), 0, 100)     # 평균 70, 표준편차 15, 음의 상관, 0~100으로 자름
    df["regular_payment_count"] = rng.poisson(np.exp(1.2 + 0.35 * z("regular_payment_count", RHO["regular_payment_count"])))  # 정기결제 건수: 평균 약 3.5건, 부도와 양의 상관(할부·상환 부담 가정)
    df["app_login_frequency"]   = rng.poisson(np.exp(2.6 + 0.40 * z("app_login_frequency", RHO["app_login_frequency"])))   # 앱 로그인 빈도: 평균 약 14.6회, 부도와 음의 상관

    df = inject_thin_filers(df, rng, thin_filer_ratio, thin_info_loss)
    df = add_approval_label(df, t, f, rng, bias_ratio, approval_rate)
    return df


# 성별 DI와 연령대 DI를 따로 계산하는 함수 (두 리포트가 공통으로 사용)
def gender_di(df, approved):
    """성별 DI = 여성 승인율 / 남성 승인율"""
    approved = np.asarray(approved, dtype=float)
    g = df["gender"].to_numpy()
    return approved[g == 1].mean() / approved[g == 0].mean()


def age_di(df, approved, reference=AGE_REFERENCE):
    """연령대 DI = 각 연령대 승인율 / 기준 연령대(35~54) 승인율 (기준 집단 제외한 dict)"""
    rates = pd.Series(np.asarray(approved, dtype=float), index=df.index).groupby(df["age_group"]).mean()
    return {grp: rates[grp] / rates[reference] for grp in AGE_LABELS if grp != reference}


def _fmt_age_di(di_dict):
    return ", ".join(f"{grp}/{AGE_REFERENCE} {v:.3f}" + ("(0.8 미만)" if v < 0.8 else "") for grp, v in di_dict.items())


def approval_report(df):
    """성별·연령대별 승인율과 DI를 출력 (approved 라벨 기준 → BIAS_PENALTY 영향). 반환값은 성별 DI"""
    # 자체 6구간 대신 age_group 컬럼 사용
    g = df.assign(성별=df["gender"].map({0: "남", 1: "여"}))
    by_gender = g.groupby("성별")["approved"].mean()
    by_age = g.groupby("age_group")["approved"].mean()

    print("  [라벨 기준] 성별 승인율:", {k: f"{v:.1%}" for k, v in by_gender.items()})
    print("  [라벨 기준] 연령대별 승인율:", {k: f"{v:.1%}" for k, v in by_age.items()})
    di = gender_di(df, df["approved"])                                      # 성별 DI 함수 사용
    print(f"  [라벨 기준] 성별 DI(여/남): {di:.3f}")
    print(f"  [라벨 기준] 연령대 DI: {_fmt_age_di(age_di(df, df['approved']))}")  # 연령대 DI 추가
    return di   # 테스트에서 쓰도록 DI 반환


# FEATURE_BIAS가 대안변수 기반 승인 결과에 주는 성별 DI 확인
# 모델 학습은 모델 단계(LR·XGBoost·LightGBM)에서 하므로, 여기서는 학습 없이 간이 점수로만 점검
def feature_di_report(df, target="SeriousDlqin2yrs", approval_rate=0.7, verbose=True):
    """대안변수 간이 점수 → 위험 낮은 상위 approval_rate 승인 → 성별 DI(여/남)

    간이 점수 = Σ (대안변수 표준화값 × 그 변수와 타깃의 상관계수). 클수록 부도 위험.
    부도 라벨과의 상관으로 가중치를 정하므로, 부도 라벨로 학습한 모델이 받는 편향을 근사함.
    """
    X = df[ALT_COLS].astype(float)
    Z = (X - X.mean()) / X.std()                       # 표준화
    w = X.corrwith(df[target])                         # 각 대안변수와 부도의 상관 (가중치)
    risk = Z.to_numpy() @ w.to_numpy()                 # 부도 위험 간이 점수

    approved = risk <= np.quantile(risk, approval_rate)   # 전체 승인율을 approval_rate로 고정
    g = df["gender"].to_numpy()
    rate_m = approved[g == 0].mean()
    rate_f = approved[g == 1].mean()
    di = gender_di(df, approved)                          # 성별 DI 함수 사용

    if verbose:
        print(f"  [대안변수 기준] 승인율 남 {rate_m:.1%} / 여 {rate_f:.1%}, "
              f"성별 DI(여/남): {di:.3f}" + ("  ← 0.8 미만" if di < 0.8 else ""))
        print(f"  [대안변수 기준] 연령대 DI: {_fmt_age_di(age_di(df, approved))}")   # 연령대 DI 추가
    return di

def feature_bias_sweep(df, scales=(0.5, 0.75, 1, 1.25, 1.5), bias_ratio=SCENARIO_BIAS_RATIO, groups=("female",)):
    """FEATURE_BIAS의 groups 항목을 scale배 했을 때의 대안변수 기준 성별 DI 출력 (끝나면 원래 값으로 복구)"""
    global FEATURE_BIAS
    original = FEATURE_BIAS
    try:
        for s in scales:
            FEATURE_BIAS = {feat: {grp: (v * s if grp in groups else v) for grp, v in d.items()}
                            for feat, d in original.items()}
            r = add_alt_features(df, bias_ratio=bias_ratio)
            di = feature_di_report(r, verbose=False)
            print(f"  FEATURE_BIAS({','.join(groups)}) ×{s}: 성별 DI = {di:.3f}" + ("  ← 0.8 미만" if di < 0.8 else ""))
    finally:
        FEATURE_BIAS = original

def calibrate_rho(df, bias_ratio=SCENARIO_BIAS_RATIO, target=0.33, grid=np.arange(0.25, 0.76, 0.01),
                  key="SeriousDlqin2yrs"):
    """RHO 추천값(dict)을 반환하고 통제/시나리오 상관을 출력 (끝나면 RHO 원래 값으로 복구)"""
    global RHO
    original = RHO
    sign = {c: np.sign(v) for c, v in original.items()}
    rows = []
    try:
        for r in grid:
            RHO = {c: sign[c] * r for c in ALT_COLS}            # 모든 변수를 같은 크기 r로 두고 한 번에 측정
            ctrl = add_alt_features(df, bias_ratio=CONTROL_BIAS_RATIO)[ALT_COLS].corrwith(df[key]).abs()
            scen = add_alt_features(df, bias_ratio=bias_ratio)[ALT_COLS].corrwith(df[key]).abs()
            rows.append(pd.DataFrame({"rho": r, "통제": ctrl, "시나리오": scen}))
    finally:
        RHO = original

    table = pd.concat(rows).rename_axis("변수").reset_index()
    table["낮은쪽"] = table[["통제", "시나리오"]].min(axis=1)              # [3차-4] 중간값 → 낮은 쪽 기준
    best = table.loc[(table["낮은쪽"] - target).abs().groupby(table["변수"]).idxmin()].set_index("변수")
    suggestion = {c: round(float(sign[c] * best.loc[c, "rho"]), 2) for c in ALT_COLS}

    print(f"[rho 재보정] bias_ratio={bias_ratio}, 목표: 낮은 쪽 |r|={target}")
    print(best[["rho", "통제", "시나리오"]].round(3).to_string())
    print("  추천 RHO =", suggestion)
    return suggestion

def alt_feature_report(df):
    """성별·연령대별 대안변수 평균 출력"""
    by_gender = df.groupby(df["gender"].map({0: "남", 1: "여"}))[ALT_COLS].mean()
    by_age = df.groupby("age_group")[ALT_COLS].mean()
    print(pd.concat([by_gender, by_age]).round(3).to_string())

def sample_size_report(df, target="SeriousDlqin2yrs", test_size=TEST_SIZE, min_test_defaults=MIN_TEST_DEFAULTS):
    """성별·연령대·페르소나·씬파일러 집단별 인원/부도 건수와 테스트셋 예상 부도 건수 출력 (부족하면 경고)"""
    need = int(np.ceil(min_test_defaults / test_size))          # 전체에서 필요한 부도 건수 (100 / 0.15 ≈ 667)
    groups = {
        "성별": df["gender"].map({0: "남", 1: "여"}),
        "연령대": df["age_group"],
        "페르소나": df["persona"].map({v: k for k, v in PERSONA_CODE.items()}),
        "씬파일러": df["is_thin_filer"].map({0: "비씬", 1: "씬"}),
    }
    rows = []
    for kind, labels in groups.items():
        stat = df.groupby(labels)[target].agg(인원="size", 부도="sum")
        stat.insert(0, "구분", kind)
        rows.append(stat)
    table = pd.concat(rows).rename_axis("집단")
    table["테스트 예상 부도"] = (table["부도"] * test_size).round().astype(int)
    table["충족"] = np.where(table["부도"] >= need, "O", "X")

    print(f"\n[표본 하한] 테스트 {test_size:.0%} 기준 부도 {min_test_defaults}건 이상 → 집단별 전체 부도 {need}건 이상 필요")
    print(table.to_string())
    for name, row in table[table["충족"] == "X"].iterrows():
        logging.warning(f"표본 부족: {row['구분']} '{name}' 부도 {row['부도']}건 (< {need}건) → AUC·TPR/FPR 불안정")
    return table


# [스펙 확인] 대안변수와 타깃(SeriousDlqin2yrs) 간 상관계수가 0.3~0.5 범위인지 df.corr()로 확인
def corr_report(df, target="SeriousDlqin2yrs", low=0.3, high=0.5):
    """대안변수 + 타깃의 상관행렬(df.corr())과 타깃 상관의 스펙 충족 여부를 출력"""
    # 상관행렬: 대안변수 5개 + 타깃 (6×6)
    corr_matrix = df[ALT_COLS + [target]].corr()
    print("[상관행렬 df.corr()] 대안변수 + 타깃")
    print(corr_matrix.round(3).to_string())

    # 타깃 열만 꺼내서 스펙(|r| 0.3~0.5) 충족 여부 판정
    corr = corr_matrix[target].drop(target)
    report = pd.DataFrame({
        "상관계수": corr.round(3),
        "|상관계수|": corr.abs().round(3),
        "방향": np.where(corr > 0, "+", "-"),
        "스펙 충족": np.where(corr.abs().between(low, high), "O", "X"),
    })
    print(f"\n[대안변수-타깃 상관계수 리포트] 스펙: |r| {low}~{high}")
    print(report.to_string())
    print(f"  충족: {(report['스펙 충족'] == 'O').sum()}/{len(ALT_COLS)}개")

    for col, r in corr.items():
        if not low <= abs(r) <= high:
            logging.warning(f"스펙 위반: {col}의 |상관계수| {abs(r):.3f}가 {low}~{high} 범위를 벗어남")
    return corr_matrix, report

def load_base():
    df = load_gmsc()                    # loader.py와 같은 폴더의 cs-training.csv
    return df[df["age"] > 0]            # 나이 0인 입력 오류 행 제거 (age_group이 "nan"이 되는 것 방지)

def main(thin_filer_ratio=0.2, tune=False):
    # loader는 logger.info로 출력하므로, 이 설정이 있어야 컬럼명·타입·결측치 요약이 보임
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_base()
    print(f"읽기 완료: {len(df):,}행", flush=True)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {"통제": CONTROL_BIAS_RATIO, "시나리오": SCENARIO_BIAS_RATIO}
    for name, br in scenarios.items():
        print(f"\n===== [{name}] bias_ratio={br} =====", flush=True)
        result = add_alt_features(df, thin_filer_ratio=thin_filer_ratio, bias_ratio=br)
        print(f"씬파일러 비율: {result['is_thin_filer'].mean():.1%}", flush=True)
        if br == CONTROL_BIAS_RATIO:
            sample_size_report(result)   # [3차-3] 집단 구성은 두 데이터가 같으므로 통제용에서 한 번만

        # [스펙 확인] 상관행렬 + 스펙 충족 리포트 (위반 시 경고만), 보고서용 CSV 저장
        corr_matrix, corr_table = corr_report(result)
        corr_matrix.round(3).to_csv(out_dir / f"상관행렬_{name}.csv", encoding="utf-8-sig")

        approval_report(result)      # 라벨 기준 DI (BIAS_PENALTY 영향)
        feature_di_report(result)    # 대안변수 기준 DI (FEATURE_BIAS 영향)
        alt_feature_report(result)   # 그룹별 대안변수 평균

        output_path = out_dir / f"결과_{name}.csv"
        print(f"결과 저장 중: {output_path}", flush=True)
        result.to_csv(output_path, index=False)
        print(f"완료: {len(result):,}행 저장됨", flush=True)

    # 보정용 탐색 (값을 다시 정할 때만 tune=True)
    if tune:
        print(f"\n[FEATURE_BIAS female 배율 탐색, bias_ratio={SCENARIO_BIAS_RATIO}]")
        feature_bias_sweep(df)
        calibrate_rho(df)


# 직접 실행할 때만 동작 (preprocessor.py 등에서 import할 때는 실행 안 됨)
# 실행: 프로젝트 루트에서 python -m src.data.simulator
if __name__ == "__main__":
    main()