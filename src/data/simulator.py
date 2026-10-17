from pathlib import Path #파일 경로를 다루는 객체
import numpy as np # 난수 생성, 수치 계산
import pandas as pd # CSV/데이터프레임 처리


# 대안 변수 생성 : telecom_payment_rate, utility_payment_rate, spending_consistency
#   regular_payment_count, app_login_frequency

# 연체 관련 컬럼 (씬파일러 판정에 사용)
DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# 연령대 구간 (스펙): 편향 주입·대안변수 편향·리포트가 모두 이 age_group 하나를 사용
AGE_BINS   = [0, 34, 54, 200]
AGE_LABELS = ["~34", "35~54", "55+"]

# [피드백1] 변경: 키를 AGE_LABELS와 같은 이름으로 (age_group 컬럼으로 바로 찾기 위해)
# 편향 주입 시 그룹별 감점 (점수 표준편차 단위, 가정값). 실제 감점 = bias_ratio × 값
BIAS_PENALTY = {"female": 1.0, "~34": 1.0, "55+": 0.5}

# [피드백2·3] 변경: 대안변수 편향을 bias_ratio 하나로 조절 (feature_bias_ratio 삭제) 
# 대안변수 편향 (가정값): 불리한 집단의 잠재값을 몇 표준편차 낮출지. 실제 = bias_ratio × 값
# 부도(타깃)와 무관하게 '그룹'만 보고 낮춤 → 모델이 부도 라벨로 학습해도 이 편향을 배우게 됨
FEATURE_BIAS = {
    "telecom_payment_rate":  {"female": 0.3},               # 통신비가 배우자 명의라 기록 불리
    "utility_payment_rate":  {"female": 0.3, "~34": 0.5},   # 공과금이 본인 명의 아님(주부·부모와 동거)
    "spending_consistency":  {"~34": 0.4},                  # 소득 변동이 큰 사회초년생
    "regular_payment_count": {"~34": -0.3},                 # 부도와 +상관이라 불리하게 = 늘림
    "app_login_frequency":   {"55+": 0.8},                  # 고령층 디지털 격차
}

# 페르소나 코드 (CSV 저장용): 1: 일반, 2: 사회초년생, 3: 주부
PERSONA_CODE = {"일반": 1, "사회초년생": 2, "주부": 3}

# 씬파일러 선택 가중치 (가정값): 사회초년생·주부가 일반보다 5배 잘 뽑힘
PERSONA_WEIGHT = {PERSONA_CODE["일반"]: 1.0,
                  PERSONA_CODE["사회초년생"]: 5.0,
                  PERSONA_CODE["주부"]: 5.0}

csv_path = Path(__file__).with_name("cs-training.csv")
print(f"CSV 읽는 중: {csv_path}", flush=True)

df = pd.read_csv(csv_path, index_col=0)   # 첫 열(Unnamed: 0, 행 번호)을 인덱스로 사용

df = df[df["age"] > 0] # [추가] 나이 0인 입력 오류 행 제거 (age_group이 "nan"이 되는 것 방지)
print(f"읽기 완료: {len(df):,}행", flush=True)


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
'''공정성 검증'''
def add_age_group(df):
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df

def add_gender_and_persona(df, rng):
    income_missing = df["MonthlyIncome"].isna() # 월 소득 결측 여부
    has_dependents = df["NumberOfDependents"].fillna(0) >= 1 # 부양가족 1명 이상 (결측은 0으로 간주)
    mid_age = df["age"].between(30, 59) # 30~59세 여부

    # 여성일 확률: 기본 50%, 소득 결측+부양가족이면 상향 (가정값, 보고서에 근거 명시)
    p_female = np.full(len(df), 0.50)
    p_female[(income_missing & has_dependents & mid_age).to_numpy()] = 0.80  # 소득 결측, 부양가족있음, 30~59세를 모두 만족하면 80%로 올림
    df["gender"] = (rng.random(len(df)) < p_female).astype(int)   # 0: 남, 1: 여

    # 페르소나 라벨 (숫자 코드)
    df["persona"] = PERSONA_CODE["일반"] # 기본값: 일반(1)
    df.loc[df["age"].between(20, 29), "persona"] = PERSONA_CODE["사회초년생"] # 20~29세는 사회초년생(2)
    df.loc[(df["gender"] == 1) & mid_age & has_dependents
           & (income_missing | (df["MonthlyIncome"] < 2000)), "persona"] = PERSONA_CODE["주부"] # 여성, 30~59세, 부양가족 있음, 소득 결측 또는 2000 미만일 경우 주부(3)
    return df


def inject_thin_filers(df, rng, thin_filer_ratio):
    """전체의 thin_filer_ratio 만큼을 씬파일러로 만든다 (사회초년생·주부 우선)."""
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

    # [피드백1] 변경: 나이 숫자(29/60) 대신 age_group 컬럼으로 감점 
    age_pen = df["age_group"].map(BIAS_PENALTY).fillna(0).to_numpy()   # ~34: 1.0, 55+: 0.5, 35~54: 0
    penalty = (df["gender"] == 1).to_numpy() * BIAS_PENALTY["female"] + age_pen
    score = base - bias_ratio * penalty

    # 전체 승인율은 approval_rate로 고정하고, 그룹별 승인율만 달라지게 함
    thr = np.quantile(score, 1 - approval_rate)
    df["approved"] = (score >= thr).astype(int)
    return df



# 전체 파이프라인: 성별·페르소나·연령대 → 대안 변수 생성(편향 포함) → 씬파일러 판정 → 승인 여부 생성
# [피드백3] 변경: feature_bias_ratio 삭제, bias_ratio 하나가 승인 감점 + 대안변수 편향을 함께 조절 
def add_alt_features(df, seed=42, thin_filer_ratio=0.2, bias_ratio=0.0, approval_rate=0.7):
    assert 0 <= thin_filer_ratio <= 1 and 0 <= bias_ratio <= 1
    rng = np.random.default_rng(seed)
    df = df.copy()
    t = df["SeriousDlqin2yrs"].to_numpy()
    f = rng.standard_normal(len(df))                  # 공통 '금융 성실도' 요인

    df = add_gender_and_persona(df, rng)              # 대안변수 편향에 성별·연령대가 필요해서 먼저 실행
    df = add_age_group(df)

    def z(feature, rho):                              # 부도와 상관된 잠재값 - 그룹 편향(bias_ratio × FEATURE_BIAS)
        return _corr_latent(t, rho, f, rng) - bias_ratio * _group_shift(df, feature)

    # [피드백4] 변경
    df["telecom_payment_rate"]  = np.clip(0.88 + 0.10 * z("telecom_payment_rate", -0.31), 0, 1)   # 평균 0.88, 표준편차 약 0.1, 부도와 음의 상관(-0.31)
    df["utility_payment_rate"]  = np.clip(0.90 + 0.09 * z("utility_payment_rate", -0.31), 0, 1)   # 평균 0.90, 표준편차 약 0.09, 상관 -0.31
    df["spending_consistency"]  = np.clip(70 + 15 * z("spending_consistency", -0.32), 0, 100)     # 평균 70, 표준편차 15, 상관 -0.32, 0~100으로 자름
    df["regular_payment_count"] = rng.poisson(np.exp(1.2 + 0.35 * z("regular_payment_count", 0.45)))  # 정기결제 건수: 평균 약 3.3건, 부도와 양의 상관(+0.45, 할부·상환 부담 가정)
    df["app_login_frequency"]   = rng.poisson(np.exp(2.6 + 0.40 * z("app_login_frequency", -0.53)))   # 앱 로그인 빈도: 평균 약 13.5회, 부도와 음의 상관(-0.53)

    df = inject_thin_filers(df, rng, thin_filer_ratio)
    df = add_approval_label(df, t, f, rng, bias_ratio, approval_rate)
    return df


def approval_report(df):
    """성별·연령대별 승인율과 성별 Disparate Impact(여/남)를 출력"""
    # [피드백1] 변경: 자체 6구간 대신 age_group 컬럼 사용 
    g = df.assign(성별=df["gender"].map({0: "남", 1: "여"}))
    by_gender = g.groupby("성별")["approved"].mean()
    by_age = g.groupby("age_group")["approved"].mean()

    print("  성별 승인율:", {k: f"{v:.1%}" for k, v in by_gender.items()})
    print("  연령대별 승인율:", {k: f"{v:.1%}" for k, v in by_age.items()})
    print(f"  Disparate Impact(여/남): {by_gender['여'] / by_gender['남']:.3f}")


# [피드백2·3] 추가: 대안변수가 그룹별로 얼마나 낮아졌는지 확인
def alt_feature_report(df):
    """성별·연령대별 대안변수 평균 출력"""
    cols = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
            "regular_payment_count", "app_login_frequency"]
    by_gender = df.groupby(df["gender"].map({0: "남", 1: "여"}))[cols].mean()
    by_age = df.groupby("age_group")[cols].mean()
    print(pd.concat([by_gender, by_age]).round(3).to_string())


# [스펙 확인] 대안변수와 타깃(SeriousDlqin2yrs) 간 상관계수가 0.3~0.5 범위인지 df.corr()로 확인
def corr_report(df, target="SeriousDlqin2yrs", low=0.3, high=0.5):
    """대안변수 + 타깃의 상관행렬(df.corr())과 타깃 상관의 스펙 충족 여부를 출력"""
    cols = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
            "regular_payment_count", "app_login_frequency"]

    # 1) 상관행렬: 대안변수 5개 + 타깃 (6×6)
    corr_matrix = df[cols + [target]].corr()
    print("[상관행렬 df.corr()] 대안변수 + 타깃")
    print(corr_matrix.round(3).to_string())

    # 2) 타깃 열만 꺼내서 스펙(|r| 0.3~0.5) 충족 여부 판정
    corr = corr_matrix[target].drop(target)
    report = pd.DataFrame({
        "상관계수": corr.round(3),
        "|상관계수|": corr.abs().round(3),
        "방향": np.where(corr > 0, "+", "-"),
        "스펙 충족": np.where(corr.abs().between(low, high), "O", "X"),
    })
    print(f"\n[대안변수-타깃 상관계수 리포트] 스펙: |r| {low}~{high}")
    print(report.to_string())
    print(f"  충족: {(report['스펙 충족'] == 'O').sum()}/{len(cols)}개")

    # 스펙 미충족이면 여기서 실행을 멈춤 (어느 변수가 벗어났는지 같이 출력)
    for col, r in corr.items():
        assert low <= abs(r) <= high, f"스펙 위반: {col}의 |상관계수| {abs(r):.3f}가 {low}~{high} 범위를 벗어남"
    return corr_matrix, report


print("대안 변수 생성 중...", flush=True)
result = add_alt_features(df, thin_filer_ratio=0.2, bias_ratio=0.0)
print(f"씬파일러 비율: {result['is_thin_filer'].mean():.1%}", flush=True)

# ★ [스펙 확인] 상관행렬 + 스펙 충족 리포트 (bias_ratio=0 기준), 보고서용 CSV 저장
corr_matrix, corr_table = corr_report(result)
corr_matrix.round(3).to_csv(Path(__file__).with_name("상관행렬.csv"), encoding="utf-8-sig")

output_path = Path(__file__).with_name("결과.csv")
print(f"결과 저장 중: {output_path}", flush=True)
result.to_csv(output_path, index=False)
print(f"완료: {len(result):,}행 저장됨", flush=True)

# 확인용: bias_ratio에 따라 승인율 격차와 대안변수 격차가 어떻게 변하는지
for br in (0.0, 0.5, 1.0):
    print(f"[bias_ratio={br}]")
    r = add_alt_features(df, bias_ratio=br)
    approval_report(r)
    alt_feature_report(r)   # ★ [피드백2·3] 추가