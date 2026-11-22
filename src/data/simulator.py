"""
src/data/simulator.py
XAI 기반 대안신용평가 시스템 - 대안 데이터 시뮬레이터

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

체크리스트 대응
  1) 대안 변수 5개 생성   : add_alt_features()
                            telecom_payment_rate(통신비 정상납부율), utility_payment_rate(공과금 납부율),
                            spending_consistency(소비 일관성 점수), regular_payment_count(정기결제 건수),
                            app_login_frequency(앱 로그인 빈도)
  2) 타깃 상관 0.3~0.5    : corr_report()에서 df.corr()로 상관행렬 출력 + 변수별 스펙 충족(O/X) 표시,
                            범위를 벗어나면 WARNING 로그, 상관행렬은 CSV로 저장
  3) thin_filer_ratio     : add_alt_features(thin_filer_ratio=...) → inject_thin_filers()
                            thin_filer_ratio_check()로 여러 값에서 목표 비율과 실제 씬파일러 비율 비교
  4) bias_ratio           : add_alt_features(bias_ratio=...)
                            - 승인 라벨: add_approval_label()에서 BIAS_PENALTY × bias_ratio 만큼 점수 감점
                            - 대안 변수: FEATURE_BIAS × bias_ratio 만큼 불리한 집단의 잠재값을 낮춤
                            통제(0.0)/시나리오를 각각 생성해 approval_report(), feature_di_report()로
                            성별·연령대별 승인율과 DI 확인
                            bias_ratio_check()로 여러 값에서 성별·연령대 DI가 감소하는지 확인
                            calibrate_bias_ratio()로 목표 성별 DI에 맞는 bias_ratio 탐색 (tune=True)
  5) 씬파일러 판정        : is_thin_filer()
                            연체 관련 컬럼 3개 중 2개 이상 결측 또는 카드 거래이력 12개월 미만
                            (씬파일러 AUC uplift 평가는 모델 단계에서 수행)

구현 방법
  - 상관 생성: _alt_latent() = 몸통 + 꼬리
      * 몸통: 표준화한 타깃과 독립 난수를 선형결합해 약한 상관(RHO_BODY ±0.22)을 줌
        (2변수 Cholesky 분해와 같은 형태, 힌트의 방법 2에 해당)
      * 꼬리: _tail_flag_maker()가 '납부에 어려움을 겪는 사람'을 뽑아 위험 방향으로 TAIL_DELTA만큼 이동
          - 꼬리 확률은 기존 연체 기록 기준(TAIL_P_DELINQ / TAIL_P_CLEAN), 부도자만 TAIL_DEFAULT_BOOST배
          - 정상 고객도 꼬리에 들어가므로 부도 정답 표시가 되지 않음
          - 씬파일러는 전통 연체 정보가 지워지므로 대안변수가 이를 되살림 (씬파일러 uplift의 원천)
      * 타깃 상관(|r| 0.3~0.5)은 TAIL_DEFAULT_BOOST와 RHO_BODY로 조절
        (TAIL_DELTA는 키워도 |r|이 corr(꼬리, 부도) 근처에서 멈추므로 고정값으로 둠)
      * 모든 대안 변수가 공통 요인 f('금융 성실도', 비중 LAM)와 꼬리 대상자(TAIL_SHARE)를 공유
        → 대안 변수끼리 상관이 높아, 합쳐도 AUC가 과하게 오르지 않음
      * 승인 라벨(approved)의 신용 점수는 _corr_latent()(꼬리 없는 단순 버전, rho=-0.6)로 생성
  - 분포 변환: 비율·점수는 clip, 건수·빈도는 Poisson으로 변환
      * 변환 과정에서 상관이 약해지므로, 상수를 바꾸면 corr_report()로 매번 다시 확인
        (시나리오 최소 |r| 0.325 → 하한 0.3까지 여유 0.025)
  - 검증 체크: 생성 후 corr_report()로 상관 확인, bias_ratio 변경 시 집단별 승인율·DI 변화 확인

설계 원칙
  - 편향 주입 (가정값)
      * 보호 속성은 성별(gender)과 연령대(age_group: ~34 / 35~54 / 55+)
      * 편향은 부도 여부와 무관하게 집단만 보고 넣음 → 부도 라벨로 학습한 모델도 편향을 배우게 됨
      * female 편향은 5개 변수 모두 불리한 방향으로 넣음
        (일부 변수에만 넣으면 모델이 그 패턴으로 성별을 알아채 편향을 보정해 버림)
  - 페르소나: 일반/사회초년생/주부 라벨을 만들고, 사회초년생·주부가 씬파일러로 5배 잘 뽑히게 함
  - 공정성 지표 공유
      * 연령 구간(AGE_BINS, AGE_LABELS)과 DI 계산 함수는 공정성 모듈(src/fairness/fairness.py)에서 가져옴
      * 시뮬레이터 리포트와 공정성 검증이 같은 정의를 써야 보고서의 DI 값이 어긋나지 않음
  - 재현성: seed=42 고정 → bias_ratio·thin_filer_ratio를 바꿔도 다른 조건은 동일
  - 표본 점검: sample_size_report()로 집단별 부도 건수가 테스트셋 평가에 충분한지 확인

출력
  data/processed/결과_통제.csv, 결과_시나리오.csv         : 대안 변수·보호 속성·승인 라벨 포함 데이터
  data/processed/상관행렬_통제.csv, 상관행렬_시나리오.csv : 보고서용 상관행렬

실행
  직접 실행     : python src/data/simulator.py
  모듈 실행     : 프로젝트 루트에서 python -m src.data.simulator
  값 보정 포함  : python -c "from src.data.simulator import main; main(tune=True)"
                  → FEATURE_BIAS 배율 탐색(~34, 55+), 목표 성별 DI 기준 bias_ratio 추천값 출력
"""
import logging              # loader의 로그(컬럼명·타입·결측치)와 스펙 위반 경고 출력용
from pathlib import Path    # 파일 경로를 다루는 객체
import sys
import numpy as np          # 난수 생성, 수치 계산
import pandas as pd         # CSV/데이터프레임 처리
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split        

# 연령 구간 정의와 DI 계산 함수는 공정성 모듈에서 가져옴
# → 시뮬레이터 리포트와 공정성 검증이 같은 정의를 써야 보고서의 DI 값이 어긋나지 않음
if __package__:
    from .loader import load_gmsc   # 모듈 실행(python -m) 시 같은 패키지의 로더 사용
    from ..fairness.fairness import AGE_BINS, AGE_LABELS, gender_di, age_di, fmt_age_di
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data.loader import load_gmsc
    from src.fairness.fairness import AGE_BINS, AGE_LABELS, gender_di, age_di, fmt_age_di


# =====================================================================
# 상수 / 가정값
# =====================================================================

# 생성할 대안 변수 5개
ALT_COLS = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
            "regular_payment_count", "app_login_frequency"]

# 연체 관련 컬럼 (씬파일러 판정에 사용)
DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# 씬파일러에게서 정보를 지울 기존 신용 변수 (연체 + 부채·대출 관련)
TRAD_INFO_COLS = DELINQ_COLS + [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]

THIN_INFO_LOSS = 0.5   # 씬파일러의 기존 변수 값 중 50%를 정보 없는 값으로 교체 (가정값)

# DebtRatio 교체 시 기증자(donor)를 받는 사람의 소득 상태에 맞춰 고름
#   - GMSC 원본은 소득이 결측이거나 0~1이면 DebtRatio 칸에 비율이 아니라 '부채 금액'이 들어 있음
#   - 아무나 기증자로 고르면 '소득 있는 사람 + 남의 부채 금액' 조합이 생김 (이전 버전에서 14,977행 발생)
#   - 월 소득은 성별·페르소나(주부 = 소득 결측/2,000 미만) 판정에 쓰이므로 바꾸지 않고,
#     DebtRatio만 '소득 상태·소득 구간이 같은 사람'에게서 가져온다
VALID_INCOME_MIN = 1      # 월 소득이 이 값 이하(0, 1)이거나 결측이면 '소득 정보 없음'으로 봄
DEBT_MATCH_BANDS = 5      # 소득 있는 사람은 소득 5분위 구간이 같은 기증자에게서 DebtRatio를 가져옴
DEBT_MATCH_SEED = 4242    # DebtRatio 기증자 선택 전용 난수 (주 난수열을 건드리지 않아 다른 컬럼·승인 라벨은 이전과 동일)

# 모델 피처가 아닌 컬럼 (라벨·보호 속성·대리변수) - 전처리 단계에서 제외할 때 참고
NON_FEATURE_COLS = ["approved", "gender", "persona", "age", "age_group"]

# 통제 데이터는 편향 없음(0.0), 시나리오 데이터는 편향 최대(1.0)
# 시나리오 값은 calibrate_bias_ratio()로 목표 DI에 맞춰 다시 정할 수 있음
CONTROL_BIAS_RATIO  = 0.0
SCENARIO_BIAS_RATIO = 1.0

YOUNG_ADULT_AGE = (21, 29)   # 사회초년생 페르소나 나이 범위

# 집단별 평가 안정성 기준: 테스트셋(15%)에 부도 100건 이상
MIN_TEST_DEFAULTS = 100
TEST_SIZE = 0.15

# 연령대 라벨은 공정성 모듈의 AGE_LABELS 를 그대로 사용 (모듈마다 표기가 달라도 편향이 사라지지 않게)
AGE_YOUNG, AGE_MID, AGE_OLD = AGE_LABELS   # 예: "~34", "35~54", "55+"

# 승인 라벨 편향 (가정값): 불리한 집단의 신용 점수에서 bias_ratio × 값 만큼 감점
BIAS_PENALTY = {"female": 1.0, AGE_YOUNG: 0.7, AGE_OLD: 0.8}

# 대안변수 편향 (가정값): 불리한 집단의 잠재값을 몇 표준편차 낮출지. 실제 = bias_ratio × 값
# 부도와 무관하게 그룹만 보고 낮춤 → 모델이 부도 라벨로 학습해도 이 편향을 배우게 됨
# female 편향은 5개 변수 모두에 같은 방향(불리한 쪽)으로 넣음
# → 일부 변수에만 넣으면 모델이 "이 변수들만 낮은 패턴"으로 성별을 알아채 편향을 보정해 버림

# female 편향(1.5)은 연령 편향보다 크게 설정 (모델 기준 성별 DI가 0.8 아래로 내려가도록 보정한 값)
FEATURE_BIAS = {
    "telecom_payment_rate":  {"female": 1.5, AGE_OLD: 0.5},               # 통신비가 배우자·자녀 명의라 기록 불리
    "utility_payment_rate":  {"female": 1.5, AGE_YOUNG: 1.0},               # 공과금이 본인 명의 아님(주부·부모와 동거)
    "spending_consistency":  {"female": 1.5, AGE_YOUNG: 0.8, AGE_OLD: 0.7},   # 경력단절·사회초년생·은퇴로 소비 변동
    "regular_payment_count": {"female": -1.5, AGE_YOUNG: -0.6},           # 부도와 +상관이라 불리하게 = 늘림 (가계 고정지출을 본인 명의로 결제)
    "app_login_frequency":   {"female": 1.5, AGE_OLD: 0.6},               # 가계 금융을 배우자 명의 앱으로 관리 / 고령층 디지털 격차
}

LAM = 0.8          # 공통 요인 비중: 대안변수끼리 정보가 겹쳐, 합쳤을 때 AUC가 덜 오름
TAIL_SHARE = 0.85  # 변수끼리 같은 부도자가 꼬리에 들어갈 확률 → 높을수록 합친 AUC가 덜 오름
# 꼬리 확률 (가정값): 부도 여부가 아니라 기존 연체 기록을 기준으로 정함
# → 꼬리 = "납부에 어려움을 겪는 사람". 정상 고객도 들어갈 수 있어 정답 표시가 되지 않음
TAIL_P_DELINQ = 0.5        # 연체 기록 있는 사람이 꼬리에 들어갈 확률 (부도 여부 무관)
TAIL_P_CLEAN = 0.02        # 연체 기록 없는 사람
TAIL_DEFAULT_BOOST = 1.7   # 부도자는 확률을 이만큼 곱함 → 타깃 상관을 조절하는 손잡이

# 몸통 상관 (부도와 음의 상관은 -, 정기결제만 +)
RHO_BODY = {
    "telecom_payment_rate":  -0.22,
    "utility_payment_rate":  -0.22,
    "spending_consistency":  -0.22,
    "regular_payment_count":  0.22,
    "app_login_frequency":   -0.22,
}
# 꼬리 이동 폭(표준편차 단위): 고정값. 보정 손잡이가 아님
#   δ를 키우면 잠재값이 꼬리 여부(flag)에 수렴해 |r|이 corr(flag, 부도) 근처에서 멈춤
#   (1.0배 → 1.4배로 바꿔도 최소 |r| 0.275 → 0.276) → 타깃 상관은 TAIL_DEFAULT_BOOST, RHO_BODY로 조절
#   예외: app_login_frequency만 δ=20 사용 (가정값 + 보정 결과)
#     - 평균 60회/월(하루 약 2회): 주거래 금융앱을 일상적으로 쓰는 고객을 가정 (가정값)
#     - log 스케일 계수 0.15 → 1 SD 변화가 ±16%(약 ±9회)뿐이라 Poisson 잡음(√60 ≈ 7.7회)과 비슷한 크기
#       → 같은 δ(4.5)로는 타깃 상관이 다른 변수보다 약함
#     - 실제 이동량은 계수 × δ = 0.15 × 20 = 3.0 (로그) → 꼬리 대상자는 60 × e^-3 ≈ 3회/월
#       = "앱을 사실상 쓰지 않는 고객"으로 해석
TAIL_DELTA = {c: 4.5 for c in ALT_COLS}
TAIL_DELTA["app_login_frequency"] = 20.0

# 페르소나 코드 (CSV 저장용): 1: 일반, 2: 사회초년생, 3: 주부
PERSONA_CODE = {"일반": 1, "사회초년생": 2, "주부": 3}

# 씬파일러 선택 가중치 (가정값): 사회초년생·주부가 일반보다 5배 잘 뽑힘
PERSONA_WEIGHT = {PERSONA_CODE["일반"]: 1.0,
                  PERSONA_CODE["사회초년생"]: 5.0,
                  PERSONA_CODE["주부"]: 5.0}


# =====================================================================
# 데이터 생성 - 내부 도구 함수
# =====================================================================

# t = 타깃, rho = 원하는 상관계수, f = 모든 변수가 공유하는 공통 요인(대안변수끼리 상관 만듦)
# rng = 난수 생성기, lam = 공통 요인이 노이즈에서 차지하는 비중
def _corr_latent(t, rho, f, rng, lam=0.3):
    """타깃 t와 상관계수가 약 rho인 표준정규 잠재값 생성"""
    s = (t - t.mean()) / t.std()                                           # 표준화한 타깃 (평균 0, 분산 1)
    e = np.sqrt(lam) * -np.sign(rho) * f + np.sqrt(1 - lam) * rng.standard_normal(len(t))  # f를 위험 방향에 맞춰 넣음 (rho가 +인 변수는 -f)
    return rho * s + np.sqrt(1 - rho**2) * e                               # 타깃과의 상관 ≈ rho

def _tail_flag_maker(t, has_delinq, rng, share=TAIL_SHARE):
    """변수마다 '꼬리' 대상자를 뽑는 함수를 반환. share 확률로 공통 대상자를 공유
    꼬리 확률은 기존 연체 기록으로 정하고 부도자만 조금 높임 (정상 고객도 꼬리에 들어감)
    → 비씬파일러: 대안변수가 전통 연체 정보를 비추는 거울이라 정보가 겹침
    → 씬파일러: 전통 연체 정보가 지워지므로 대안변수가 이를 되살림 (uplift의 원천)"""
    n = len(t)
    p = np.where(has_delinq, TAIL_P_DELINQ, TAIL_P_CLEAN) * np.where(t == 1, TAIL_DEFAULT_BOOST, 1.0)
    p = np.clip(p, 0, 1)
    base = rng.random(n) < p                           # 여러 변수에서 공통으로 나쁜 사람
    def next_flag():
        own = rng.random(n) < p                        # 이 변수에서만 나쁜 사람
        return np.where(rng.random(n) < share, base, own)
    return next_flag

def _alt_latent(t, rho_body, delta, flag, f, rng, lam=LAM):
    """몸통(약한 상관) + 꼬리(일부 부도자만 크게 이동) 잠재값"""
    s = (t - t.mean()) / t.std()
    e = np.sqrt(lam) * -np.sign(rho_body) * f + np.sqrt(1 - lam) * rng.standard_normal(len(t))
    z = rho_body * s + np.sqrt(1 - rho_body**2) * e
    return z + np.sign(rho_body) * delta * flag               # 위험한 방향으로 꼬리 이동

def _group_shift(df, feature):
    """해당 대안변수에 대해 각 사람이 받는 편향 크기(표준편차 단위)"""
    shift = np.zeros(len(df))
    for group, v in FEATURE_BIAS.get(feature, {}).items():
        mask = (df["gender"] == 1) if group == "female" else (df["age_group"] == group)
        shift += mask.to_numpy() * v                                       # 여러 집단에 속하면 편향이 더해짐
    return shift


def is_thin_filer(df):
    """씬파일러 판정: 연체 관련 컬럼 2개 이상 결측 또는 카드 거래이력 12개월 미만"""
    delinq_missing = df[DELINQ_COLS].isna().sum(axis=1) >= 2
    short_history = df["card_history_months"] < 12
    return delinq_missing | short_history


# =====================================================================
# 데이터 생성 - 보호 속성·페르소나
# =====================================================================

# 나이를 AGE_BINS·AGE_LABELS(공정성 모듈에서 정의) 기준 3구간(~34, 35~54, 55+)으로 잘라 문자열로 저장
def add_age_group(df):
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df


# 주부 판정·여성 확률 상향에 같이 쓰는 조건 (add_gender·add_persona가 공유)
def _homemaker_conditions(df):
    income_missing = df["MonthlyIncome"].isna()                  # 월 소득 결측 여부
    has_dependents = df["NumberOfDependents"].fillna(0) >= 1     # 부양가족 1명 이상 (결측은 0으로 간주)
    mid_age = df["age"].between(30, 59)                          # 30~59세 여부
    return income_missing, has_dependents, mid_age


def add_gender(df, rng):
    """성별 생성 (0: 남, 1: 여). 원본 데이터에 성별이 없어 확률적으로 부여"""
    income_missing, has_dependents, mid_age = _homemaker_conditions(df)

    # 여성일 확률: 기본 50%, 소득 결측+부양가족+30~59세이면 80%로 상향 (가정값, 보고서에 근거 명시)
    p_female = np.full(len(df), 0.50)
    p_female[(income_missing & has_dependents & mid_age).to_numpy()] = 0.80
    df["gender"] = (rng.random(len(df)) < p_female).astype(int)
    return df


def add_persona(df):
    """페르소나 라벨 생성 (1: 일반, 2: 사회초년생, 3: 주부)"""
    income_missing, has_dependents, mid_age = _homemaker_conditions(df)

    df["persona"] = PERSONA_CODE["일반"]                                                  # 기본값: 일반(1)
    df.loc[df["age"].between(*YOUNG_ADULT_AGE), "persona"] = PERSONA_CODE["사회초년생"]   # 21~29세는 사회초년생(2)
    df.loc[(df["gender"] == 1) & mid_age & has_dependents
           & (income_missing | (df["MonthlyIncome"] < 2000)), "persona"] = PERSONA_CODE["주부"]  # 여성, 30~59세, 부양가족 있음, 소득 결측 또는 2000 미만이면 주부(3)
    return df


# =====================================================================
# 데이터 생성 - 씬파일러·승인 라벨
# =====================================================================

def _income_matched_debt_ratio(df, donors, receivers, bands=DEBT_MATCH_BANDS, seed=DEBT_MATCH_SEED):
    """받는 사람(receivers)마다 소득 상태가 같은 기증자의 DebtRatio를 골라 반환

    - 소득 정보 없음(결측·0~1) → 소득 정보 없는 기증자 (DebtRatio = 금액, 전처리에서 결측 처리됨)
    - 소득 있음               → 소득 있는 기증자 중 소득 분위 구간이 같은 사람 (DebtRatio = 비율)
    """
    match_rng = np.random.default_rng(seed)
    income = df["MonthlyIncome"].to_numpy(dtype=float)
    debt = df["DebtRatio"].to_numpy(dtype=float)
    valid = ~np.isnan(income) & (income > VALID_INCOME_MIN)

    # 소득 구간 경계는 기증자(소득 있음) 분포로 정함 → 받는 사람도 같은 경계로 구간 배정
    d_valid = donors[valid[donors]]
    edges = np.unique(np.quantile(income[d_valid], np.linspace(0, 1, bands + 1)[1:-1]))
    band = np.full(len(df), -1)                       # -1 = 소득 정보 없음
    band[valid] = np.searchsorted(edges, income[valid], side="right")

    out = np.empty(len(receivers))
    for b in np.unique(band[receivers]):
        pool = donors[band[donors] == b]
        if len(pool) == 0:                            # 같은 구간 기증자가 없으면 소득 상태만 맞춤
            pool = d_valid if b >= 0 else donors[~valid[donors]]
        sel = band[receivers] == b
        out[sel] = debt[match_rng.choice(pool, size=sel.sum())]
    return out


def inject_thin_filers(df, rng, thin_filer_ratio, thin_info_loss=THIN_INFO_LOSS):
    """전체의 thin_filer_ratio 만큼 씬파일러로 만들고 is_thin_filer 컬럼 생성"""
    n = len(df)
    k = int(round(n * thin_filer_ratio))   # 씬파일러로 만들 인원 수

    # 페르소나별 선택 가중치로 씬파일러 k명 선택 (사회초년생·주부가 5배 잘 뽑힘)
    w = df["persona"].map(PERSONA_WEIGHT).to_numpy()
    assert not np.isnan(w).any(), f"PERSONA_WEIGHT에 없는 persona 값: {df['persona'].unique()}"
    idx = rng.choice(n, size=k, replace=False, p=w / w.sum())
    selected = np.zeros(n, dtype=bool)
    selected[idx] = True

    # 카드 거래이력(개월): 일반은 나이에 비례(12~360개월), 씬파일러는 0~11개월
    months = np.clip((df["age"].to_numpy() - 18) * 12 * rng.uniform(0.3, 0.9, n), 12, 360)
    months = months.astype(int)
    months[idx] = rng.integers(0, 12, size=k)
    df["card_history_months"] = months

    # 씬파일러의 기존 신용 변수 중 thin_info_loss 비율을 다른 사람(비씬파일러) 값으로 교체
    # → 값은 그럴듯하지만 본인 신용도와 무관해져 정보가 사라짐
    donors = np.flatnonzero(~selected)
    for col in TRAD_INFO_COLS:
        replace = selected & (rng.random(n) < thin_info_loss)
        # 주 난수열 소비량을 이전과 같게 유지 (DebtRatio도 일단 뽑고, 아래에서 소득 맞춤 값으로 덮어씀)
        donor_vals = df[col].to_numpy()[rng.choice(donors, size=replace.sum())]
        if col == "DebtRatio":
            donor_vals = _income_matched_debt_ratio(df, donors, np.flatnonzero(replace))
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

    # 불리한 집단 감점: BIAS_PENALTY 기준 (여성 1.0, ~34 0.7, 55+ 0.8, 35~54 0)
    age_pen = df["age_group"].map(BIAS_PENALTY).fillna(0).to_numpy()
    penalty = (df["gender"] == 1).to_numpy() * BIAS_PENALTY["female"] + age_pen
    score = base - bias_ratio * penalty

    # 전체 승인율은 approval_rate로 고정하고, 그룹별 승인율만 달라지게 함
    thr = np.quantile(score, 1 - approval_rate)
    df["approved"] = (score >= thr).astype(int)
    return df


# =====================================================================
# 데이터 생성 - 전체 파이프라인
# =====================================================================

# 성별·연령대·페르소나 → 대안 변수 생성(편향 포함) → 씬파일러 주입·판정 → 승인 여부 생성
def add_alt_features(df, seed=42, thin_filer_ratio=0.2, bias_ratio=0.0, approval_rate=0.7,
                     thin_info_loss=THIN_INFO_LOSS):
    assert 0 <= thin_filer_ratio <= 1 and 0 <= bias_ratio <= 1 and 0 <= thin_info_loss <= 1
    rng = np.random.default_rng(seed)                 # seed 고정 → 같은 입력이면 같은 결과
    df = df.copy()                                    # 원본 데이터는 건드리지 않음
    t = df["SeriousDlqin2yrs"].to_numpy()             # 타깃(부도 여부)
    f = rng.standard_normal(len(df))                  # 공통 '금융 성실도' 요인

    df = add_gender(df, rng)                          # 성별 (대안변수 편향에 필요해서 먼저 실행)
    df = add_age_group(df)                            # 연령대 (편향 주입·리포트 공통 구간)
    df = add_persona(df)                              # 페르소나 라벨 (성별이 먼저 있어야 함)

    # 꼬리 대상 선택용 연체 기록 여부: 씬파일러 주입(inject_thin_filers) 전 원본 기준
    has_delinq = (df[DELINQ_COLS].fillna(0).sum(axis=1) > 0).to_numpy()
    next_flag = _tail_flag_maker(t, has_delinq, rng)

    def z(feature):   # 몸통+꼬리 잠재값 - 그룹 편향(bias_ratio × FEATURE_BIAS)
        return (_alt_latent(t, RHO_BODY[feature], TAIL_DELTA[feature], next_flag(), f, rng)
                - bias_ratio * _group_shift(df, feature))

    df["telecom_payment_rate"]  = np.clip(0.88 + 0.10 * z("telecom_payment_rate"), 0, 1)        # 평균 0.88, 표준편차 약 0.1, 0~1로 자름
    df["utility_payment_rate"]  = np.clip(0.90 + 0.09 * z("utility_payment_rate"), 0, 1)        # 평균 0.90, 표준편차 약 0.09, 0~1로 자름
    df["spending_consistency"]  = np.clip(70 + 12 * z("spending_consistency"), 0, 100)          # 평균 70, 표준편차 12, 0~100으로 자름
    df["regular_payment_count"] = rng.poisson(np.exp(1.2 + 0.35 * z("regular_payment_count")))  # 정기결제 건수: 평균 약 3.5건, 부도와 양의 상관(할부·상환 부담 가정)
    df["app_login_frequency"] = rng.poisson(np.exp(np.log(60) + 0.15 * z("app_login_frequency"))) # 앱 로그인 빈도: 평균 약 60회/월, 부도와 음의 상관

    df = inject_thin_filers(df, rng, thin_filer_ratio, thin_info_loss)
    df = add_approval_label(df, t, f, rng, bias_ratio, approval_rate)
    return df


# =====================================================================
# 시뮬레이터 점검 리포트 (DI 계산은 공정성 모듈 함수 사용)
# =====================================================================

def approval_report(df):
    """성별·연령대별 승인율과 DI를 출력 (approved 라벨 기준 → BIAS_PENALTY 영향). 반환값은 성별 DI"""
    g = df.assign(성별=df["gender"].map({0: "남", 1: "여"}))
    by_gender = g.groupby("성별")["approved"].mean()
    by_age = g.groupby("age_group")["approved"].mean()

    print("  [라벨 기준] 성별 승인율:", {k: f"{v:.1%}" for k, v in by_gender.items()})
    print("  [라벨 기준] 연령대별 승인율:", {k: f"{v:.1%}" for k, v in by_age.items()})
    di = gender_di(df, df["approved"])                                          # 성별 DI (공정성 모듈)
    print(f"  [라벨 기준] 성별 DI(여/남): {di:.3f}")
    print(f"  [라벨 기준] 연령대 DI: {fmt_age_di(age_di(df, df['approved']))}")  # 연령대 DI (공정성 모듈)
    return di   # 테스트에서 쓰도록 DI 반환


# FEATURE_BIAS가 대안변수 기반 승인 결과에 주는 DI 확인
# 모델 학습은 모델 단계(LR·XGBoost·LightGBM)에서 하므로, 여기서는 학습 없이 간이 점수로만 점검
def feature_di_report(df, target="SeriousDlqin2yrs", approval_rate=0.7, verbose=True):
    """대안변수 간이 점수 → 위험 낮은 상위 approval_rate 승인 → 성별 DI(여/남) 반환

    간이 점수 = Σ (대안변수 표준화값 × 그 변수와 타깃의 상관계수). 클수록 부도 위험.
    부도 라벨과의 상관으로 가중치를 정하므로, 부도 라벨로 학습한 모델이 받는 편향을 근사함.
    verbose=False면 출력 없이 DI만 반환 (bias_ratio_check 등에서 사용)
    """
    X = df[ALT_COLS].astype(float)
    Z = (X - X.mean()) / X.std()                       # 표준화
    w = X.corrwith(df[target])                         # 각 대안변수와 부도의 상관 (가중치)
    risk = Z.to_numpy() @ w.to_numpy()                 # 부도 위험 간이 점수

    approved = risk <= np.quantile(risk, approval_rate)   # 전체 승인율을 approval_rate로 고정
    di = gender_di(df, approved)                          # 성별 DI (공정성 모듈)

    if verbose:
        g = df["gender"].to_numpy()
        rate_m = approved[g == 0].mean()
        rate_f = approved[g == 1].mean()
        print(f"  [대안변수 기준] 승인율 남 {rate_m:.1%} / 여 {rate_f:.1%}, "
              f"성별 DI(여/남): {di:.3f}" + ("  ← 0.8 미만" if di < 0.8 else ""))
        print(f"  [대안변수 기준] 연령대 DI: {fmt_age_di(age_di(df, approved))}")   # 연령대 DI (공정성 모듈)
    return di

def model_di_report(df, target="SeriousDlqin2yrs", approval_rate=0.7, seed=42, verbose=True):
    """기존+대안 변수로 간이 모델 학습 → 테스트셋 승인 → 성별 DI, 연령대 DI 반환
      (model_di_report 만든 이유 : 보정 반복을 빠르게 돌리기 위해)"""
    drop = set(NON_FEATURE_COLS) | {target, "is_thin_filer", "Unnamed: 0"}   # 전처리기 피처 목록과 맞출 것
    feats = [c for c in df.columns if c not in drop]
    tr, te = train_test_split(df.index, test_size=TEST_SIZE, stratify=df[target], random_state=seed)
    m = HistGradientBoostingClassifier(random_state=seed).fit(df.loc[tr, feats], df.loc[tr, target])
    risk = m.predict_proba(df.loc[te, feats])[:, 1]
    approved = risk <= np.quantile(risk, approval_rate)
    d = df.loc[te]
    g, a = gender_di(d, approved), age_di(d, approved)
    if verbose:
        print(f"  [모델 기준] 성별 DI(여/남): {g:.3f}" + ("  ← 0.8 미만" if g < 0.8 else ""))
        print(f"  [모델 기준] 연령대 DI: {fmt_age_di(a)}")
    return g, a

def feature_group_auc_report(df, target="SeriousDlqin2yrs", seed=42):
    """전통/대안/통합 피처 AUC를 전체·씬파일러·비씬별로 비교 (모델 단계 전 빠른 점검)"""
    drop = set(NON_FEATURE_COLS) | {target, "is_thin_filer", "Unnamed: 0"}
    all_feats = [c for c in df.columns if c not in drop]
    groups = {"전통": [c for c in all_feats if c not in ALT_COLS], "대안": ALT_COLS, "통합": all_feats}
    tr, te = train_test_split(df.index, test_size=TEST_SIZE, stratify=df[target], random_state=seed)
    y = df.loc[te, target].to_numpy()
    thin = df.loc[te, "is_thin_filer"].to_numpy() == 1
    rows = {}
    for name, feats in groups.items():
        p = HistGradientBoostingClassifier(random_state=seed).fit(
            df.loc[tr, feats], df.loc[tr, target]).predict_proba(df.loc[te, feats])[:, 1]
        rows[name] = {"전체": roc_auc_score(y, p), "씬": roc_auc_score(y[thin], p[thin]),
                      "비씬": roc_auc_score(y[~thin], p[~thin])}
    table = pd.DataFrame(rows).T
    print(table.round(3).to_string())
   # print(f"  씬파일러 uplift(통합-전통): {table.loc['통합', '씬'] - table.loc['전통', '씬']:+.3f}")
    uplift = float(table.at['통합', '씬']) - float(table.at['전통', '씬'])
    print(f"  씬파일러 uplift(통합-전통): {uplift:+.3f}")
    return table

def alt_feature_report(df):
    """성별·연령대별 대안변수 평균 출력 (편향이 의도한 방향으로 들어갔는지 확인)"""
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


# =====================================================================
# [스펙 확인] 파라미터 조정 시 결과가 바뀌는지 확인 (노션 시뮬레이터 체크리스트)
# =====================================================================

def bias_ratio_check(df, ratios=(0.0, 0.25, 0.5, 0.75, 1.0), tol=0.01):
    """bias_ratio를 바꿔가며 성별·연령대 DI가 의도대로 변하는지 확인"""
    rows = []
    for br in ratios:
        r = add_alt_features(df, bias_ratio=br)          # seed 고정 → bias_ratio만 바뀜
        label_age = age_di(r, r["approved"])
        rows.append({
            "bias_ratio": br,
            "라벨_성별DI": gender_di(r, r["approved"]),
            f"라벨_{AGE_YOUNG}DI": label_age[AGE_YOUNG],
            f"라벨_{AGE_OLD}DI": label_age[AGE_OLD],
            "대안변수_성별DI": feature_di_report(r, verbose=False),
        })

    table = pd.DataFrame(rows).set_index("bias_ratio")
    print("\n[bias_ratio 동작 확인] bias_ratio가 커질수록 DI가 작아져야 함")
    print(table.round(3).to_string())

    # 불리한 집단에 페널티를 주므로 bias_ratio ↑ → DI ↓ 이어야 정상
    for col in table.columns:
        if (table[col].diff().dropna() > tol).any():
            logging.warning(f"bias_ratio 동작 이상: {col}이 bias_ratio 증가에 따라 감소하지 않음")
    return table


def thin_filer_ratio_check(df, ratios=(0.1, 0.2, 0.3, 0.4), tol=0.01):
    """thin_filer_ratio를 바꿔가며 실제 씬파일러 비율이 목표대로 나오는지 확인"""
    persona_name = {v: k for k, v in PERSONA_CODE.items()}
    rows = []
    for tr in ratios:
        r = add_alt_features(df, thin_filer_ratio=tr)        # seed 고정 → thin_filer_ratio만 바뀜
        thin = r[r["is_thin_filer"] == 1]
        mix = thin["persona"].map(persona_name).value_counts(normalize=True)   # 씬파일러 중 페르소나 구성비
        rows.append({
            "목표 비율": tr,
            "실제 비율": r["is_thin_filer"].mean(),
            **{f"씬 중 {p}": mix.get(p, 0.0) for p in PERSONA_CODE},
        })

    table = pd.DataFrame(rows).set_index("목표 비율")
    print("\n[thin_filer_ratio 동작 확인] 실제 비율이 목표 비율과 같아야 함")
    print(table.round(3).to_string())

    for tr, actual in table["실제 비율"].items():
        if abs(actual - tr) > tol:
            logging.warning(f"thin_filer_ratio 동작 이상: 목표 {tr:.0%} / 실제 {actual:.1%}")
    return table


# =====================================================================
# 보정용 탐색 (값을 다시 정할 때만 main(tune=True)로 실행)
# =====================================================================

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
            print(f"  FEATURE_BIAS({','.join(groups)}) ×{s}: 성별 DI = {di:.3f}" + ("  ← 0.8 미만!!" if di < 0.8 else ""))
    finally:
        FEATURE_BIAS = original   # 오류가 나도 원래 값으로 복구

def calibrate_bias_ratio(df, target_di=0.8, metric="label", grid=np.round(np.arange(0, 1.01, 0.05), 2), tol=0.02):
    """목표 성별 DI에 가장 가까운 bias_ratio를 찾아 반환 (SCENARIO_BIAS_RATIO를 정할 때 사용)

    metric="label"   : approved 라벨 기준 DI (BIAS_PENALTY 영향, approval_report와 같은 값)
    metric="feature" : 대안변수 간이 점수 기준 DI (FEATURE_BIAS 영향, feature_di_report와 같은 값)
    bias_ratio 하나가 성별·연령 편향을 같이 움직이므로, 성별 DI를 맞추면 연령 DI는 따라서 정해짐
    """
    rows = []
    for br in grid:
        r = add_alt_features(df, bias_ratio=br)          # seed 고정 → bias_ratio만 바뀜
        di = gender_di(r, r["approved"]) if metric == "label" else feature_di_report(r, verbose=False)
        rows.append({"bias_ratio": br, "성별DI": di})

    table = pd.DataFrame(rows).set_index("bias_ratio")
    best = float((table["성별DI"] - target_di).abs().idxmin())   # 목표 DI와 차이가 가장 작은 bias_ratio
    best_di = table.loc[best, "성별DI"]

    print(f"\n[bias_ratio 보정] 목표 성별 DI={target_di} ({metric} 기준)")
    print(table.round(3).to_string())
    print(f"  추천 bias_ratio = {best} (성별 DI {best_di:.3f}) → SCENARIO_BIAS_RATIO에 반영")

    # bias_ratio는 0~1로 제한되므로, 1.0에서도 목표에 못 미치면 편향 크기 자체를 키워야 함
    if abs(best_di - target_di) > tol:
        logging.warning(f"bias_ratio 0~1 범위에서 목표 DI {target_di}에 도달하지 못함 "
                        f"(가장 가까운 값: bias_ratio {best}, DI {best_di:.3f}) → BIAS_PENALTY 또는 FEATURE_BIAS 조정 필요")
    return best


# =====================================================================
# 실행
# =====================================================================

def load_base():
    """원본 GMSC 데이터 로드"""
    df = load_gmsc()                    # loader.py와 같은 폴더의 cs-training.csv
    return df[df["age"] > 0]            # 나이 0인 입력 오류 행 제거 (age_group이 "nan"이 되는 것 방지)


def main(thin_filer_ratio=0.2, tune=False):
    # 시뮬레이터의 WARNING 로그(스펙 위반, 표본 부족 등) 출력용
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_base()
    print(f"읽기 완료: {len(df):,}행", flush=True)

    # 결과 저장 폴더: 프로젝트 루트/data/processed
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 통제(편향 없음)·시나리오(편향 있음) 데이터를 각각 생성 → 점검 → 저장
    scenarios = {"통제": CONTROL_BIAS_RATIO, "시나리오": SCENARIO_BIAS_RATIO}
    for name, br in scenarios.items():
        print(f"\n===== [{name}] bias_ratio={br} =====", flush=True)
        result = add_alt_features(df, thin_filer_ratio=thin_filer_ratio, bias_ratio=br)
        print(f"씬파일러 비율: {result['is_thin_filer'].mean():.1%}", flush=True)
        if br == CONTROL_BIAS_RATIO:
            sample_size_report(result)   # 집단 구성은 두 데이터가 같으므로 통제용에서 한 번만

        # [스펙 확인] 상관행렬 + 스펙 충족 리포트 (위반 시 경고만), 보고서용 CSV 저장
        corr_matrix, _ = corr_report(result)
        corr_matrix.round(3).to_csv(out_dir / f"상관행렬_{name}.csv", encoding="utf-8-sig")

        approval_report(result)      # 라벨 기준 DI (BIAS_PENALTY 영향)
        feature_di_report(result)    # 대안변수 기준 DI (FEATURE_BIAS 영향)
        model_di_report(result)      # 모델 기준 DI (최종 판단 기준)
        feature_group_auc_report(result)  # 전통/대안/통합 AUC, 씬파일러 uplift
        alt_feature_report(result)   # 그룹별 대안변수 평균

        output_path = out_dir / f"결과_{name}.csv"
        print(f"결과 저장 중: {output_path}", flush=True)
        result.to_csv(output_path, index=False)
        print(f"완료: {len(result):,}행 저장됨", flush=True)

    # [스펙 확인] 파라미터 조정 시 결과가 바뀌는지 확인 (반복문 밖에서 한 번만)
    bias_ratio_check(df)          # bias_ratio ↑ → 성별·연령대 DI ↓
    thin_filer_ratio_check(df)    # thin_filer_ratio 목표 = 실제 씬파일러 비율

    # 보정용 탐색 (값을 다시 정할 때만 tune=True)
    if tune:
        print(f"\n[FEATURE_BIAS 배율 탐색, bias_ratio={SCENARIO_BIAS_RATIO}]")
        feature_bias_sweep(df, groups=(AGE_YOUNG,))
        feature_bias_sweep(df, groups=(AGE_OLD,))   # 참고용 (값은 바꾸지 않음)
        calibrate_bias_ratio(df, target_di=0.8, metric="label")   # 목표 성별 DI 기준 bias_ratio 추천

# 직접 실행할 때만 동작 (preprocessor.py 등에서 import할 때는 실행 안 됨)
if __name__ == "__main__":
    main()