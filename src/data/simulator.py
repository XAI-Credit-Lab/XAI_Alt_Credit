
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

# 편향 주입 시 그룹별 감점 (점수 표준편차 단위, 가정값). 실제 감점 = bias_ratio × 값
BIAS_PENALTY = {"female": 1.0, "age_young": 1.0, "age_senior": 0.5}

# 페르소나 코드 (CSV 저장용): 1: 일반, 2: 사회초년생, 3: 주부
PERSONA_CODE = {"일반": 1, "사회초년생": 2, "주부": 3}

# 씬파일러 선택 가중치 (가정값): 사회초년생·주부가 일반보다 5배 잘 뽑힘
PERSONA_WEIGHT = {PERSONA_CODE["일반"]: 1.0,
                  PERSONA_CODE["사회초년생"]: 5.0,
                  PERSONA_CODE["주부"]: 5.0}

csv_path = Path(__file__).with_name("cs-training.csv")
print(f"CSV 읽는 중: {csv_path}", flush=True)
df = pd.read_csv(csv_path)
print(f"읽기 완료: {len(df):,}행", flush=True)


# rho = 원하는 상관계수, t = 타깃, f = 모든 변수가 공유하는 공통 요인
# rng = 난수 생성기, lam = 공통 요인이 노이즈에서 차지하는 비중
def _corr_latent(t, rho, f, rng, lam=0.3):
    s = (t - t.mean()) / t.std()                                           # 표준화한 타깃
    e = np.sqrt(lam) * f + np.sqrt(1 - lam) * rng.standard_normal(len(t))  # 노이즈
    return rho * s + np.sqrt(1 - rho**2) * e                               # 타깃과의 상관 ≈ rho


def is_thin_filer(df):
    """씬파일러 판정: 연체 관련 컬럼 2개 이상 결측 또는 카드 거래이력 12개월 미만"""
    delinq_missing = df[DELINQ_COLS].isna().sum(axis=1) >= 2
    short_history = df["card_history_months"] < 12
    return delinq_missing | short_history


def add_gender_and_persona(df, rng):
    income_missing = df["MonthlyIncome"].isna()
    has_dependents = df["NumberOfDependents"].fillna(0) >= 1
    mid_age = df["age"].between(30, 59)

    # 여성일 확률: 기본 50%, 소득 결측+부양가족이면 상향 (가정값, 보고서에 근거 명시)
    p_female = np.full(len(df), 0.50)
    p_female[(income_missing & has_dependents & mid_age).to_numpy()] = 0.80
    df["gender"] = (rng.random(len(df)) < p_female).astype(int)   # 0: 남, 1: 여

        # 페르소나 라벨 (숫자 코드)
    df["persona"] = PERSONA_CODE["일반"]
    df.loc[df["age"].between(20, 25), "persona"] = PERSONA_CODE["사회초년생"]
    df.loc[(df["gender"] == 1) & mid_age & has_dependents
           & (income_missing | (df["MonthlyIncome"] < 2000)), "persona"] = PERSONA_CODE["주부"]
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

    penalty = (
        (df["gender"] == 1).to_numpy() * BIAS_PENALTY["female"]
        + (df["age"] <= 25).to_numpy() * BIAS_PENALTY["age_young"]
        + (df["age"] >= 60).to_numpy() * BIAS_PENALTY["age_senior"]
    )
    score = base - bias_ratio * penalty

    # 전체 승인율은 approval_rate로 고정하고, 그룹별 승인율만 달라지게 함
    thr = np.quantile(score, 1 - approval_rate)
    df["approved"] = (score >= thr).astype(int)
    return df


def add_alt_features(df, seed=42, thin_filer_ratio=0.2, bias_ratio=0.0, approval_rate=0.7):
    assert 0 <= thin_filer_ratio <= 1 and 0 <= bias_ratio <= 1
    rng = np.random.default_rng(seed)
    df = df.copy()
    t = df["SeriousDlqin2yrs"].to_numpy()
    f = rng.standard_normal(len(df))                  # 공통 '금융 성실도' 요인

    # 대안변수 생성
    df["telecom_payment_rate"] = np.clip(0.88 + 0.10 * _corr_latent(t, -0.45, f, rng), 0, 1) # 평균 0.88, 표준편차 약 0.1, 부도와 음의 상관(-0.45)
    df["utility_payment_rate"] = np.clip(0.90 + 0.09 * _corr_latent(t, -0.40, f, rng), 0, 1) # 평균 0.90, 표준편차 약 0.09, 상관 -0.40
    df["spending_consistency"] = np.clip(70 + 15 * _corr_latent(t, -0.35, f, rng), 0, 100) # 평균 70, 표준편차 15, 상관 -0.35, 0~100으로 자름
    df["regular_payment_count"] = rng.poisson(np.exp(1.2 + 0.35 * _corr_latent(t, 0.60, f, rng)))
    df["app_login_frequency"]   = rng.poisson(np.exp(2.6 + 0.40 * _corr_latent(t, -0.60, f, rng)))

    df = add_gender_and_persona(df, rng)
    df = inject_thin_filers(df, rng, thin_filer_ratio)
    df = add_approval_label(df, t, f, rng, bias_ratio, approval_rate)
    return df


def approval_report(df):
    """성별·연령대별 승인율과 성별 Disparate Impact(여/남)를 출력"""
    g = df.assign(
        성별=df["gender"].map({0: "남", 1: "여"}),
        연령대=pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 200],
                      labels=["~25", "26~35", "36~45", "46~55", "56~65", "66~"]),
    )
    by_gender = g.groupby("성별")["approved"].mean()
    by_age = g.groupby("연령대", observed=True)["approved"].mean()
    print("  성별 승인율:", {k: f"{v:.1%}" for k, v in by_gender.items()})
    print("  연령대별 승인율:", {k: f"{v:.1%}" for k, v in by_age.items()})
    print(f"  Disparate Impact(여/남): {by_gender['여'] / by_gender['남']:.3f}")


print("대안 변수 생성 중...", flush=True)
result = add_alt_features(df, thin_filer_ratio=0.2, bias_ratio=0.0)
print(f"씬파일러 비율: {result['is_thin_filer'].mean():.1%}", flush=True)

output_path = Path(__file__).with_name("결과.csv")
print(f"결과 저장 중: {output_path}", flush=True)
result.to_csv(output_path, index=False)
print(f"완료: {len(result):,}행 저장됨", flush=True)

# 확인용: bias_ratio에 따라 승인율 격차가 어떻게 변하는지
for br in (0.0, 0.5, 1.0):
    print(f"[bias_ratio={br}]")
    approval_report(add_alt_features(df, bias_ratio=br))