# src/api/schema.py
# pydantic request/response schema 정의
# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다

# 반사실적 설명 엔드포인트 추가

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, List, Dict, Optional
from src.data.feature_spec import field_bounds as fb
from src.data.preprocessor import DEBT_RATIO_MAX, REVOLVING_MAX

# feature_spec의 입력 상한과 전처리기의 결측 처리 기준이 같아야 함 (다르면 서버 시작 시 바로 알림)
assert fb("DebtRatio")["le"] == DEBT_RATIO_MAX, "feature_spec DebtRatio max != preprocessor DEBT_RATIO_MAX"
assert fb("RevolvingUtilizationOfUnsecuredLines")["le"] == REVOLVING_MAX, "feature_spec Revolving max != preprocessor REVOLVING_MAX"

# 1. (GET) 상태, 사용된 머신러닝 모델, 접속 시간 응답
class health_request(BaseModel):
    pass
class health_response(BaseModel):
    status: str 
    model_version: str 
    timestamp: str 


# 2. (POST) 고객 정보 JSON 입력 -> 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
class predict_request(BaseModel): 
    # populate_by_name=True: 요청 JSON에서 변수 이름(past_due_90)과 alias 이름(NumberOfTimes90DaysLate)으로 둘 다 받을 수 있도록 함
    # extra="forbid": 정의되지 않은 변수 입력 거부. 사용자 입력 중 오타(ex) "MonthlyIncom": 2500 ) 있으면 에러
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # 기존 금융 변수 (입력 범위는 feature_spec.py 한 곳에서 관리)
    RevolvingUtilizationOfUnsecuredLines: float = Field(**fb("RevolvingUtilizationOfUnsecuredLines"))
    # 월 소득이 미입력이거나 0~1이면 전처리기에서 반영되지 않음 (중앙값 대체)
    DebtRatio: float = Field(**fb("DebtRatio"))
    MonthlyIncome: Optional[float] = Field(None, **fb("MonthlyIncome"))
    NumberOfOpenCreditLinesAndLoans: int = Field(**fb("NumberOfOpenCreditLinesAndLoans"))
    NumberRealEstateLoansOrLines: int = Field(**fb("NumberRealEstateLoansOrLines"))
    NumberOfDependents: Optional[int] = Field(None, **fb("NumberOfDependents"))
    card_history_months: int = Field(**fb("card_history_months"))

    # 연체 관련 (입력 선택). 컬럼 이름에 '-'가 있어 alias로 원래 이름 연결
    past_due_30_59: Optional[int] = Field(None, alias="NumberOfTime30-59DaysPastDueNotWorse",
                                          **fb("NumberOfTime30-59DaysPastDueNotWorse"))
    past_due_60_89: Optional[int] = Field(None, alias="NumberOfTime60-89DaysPastDueNotWorse",
                                          **fb("NumberOfTime60-89DaysPastDueNotWorse"))
    past_due_90: Optional[int] = Field(None, alias="NumberOfTimes90DaysLate",
                                       **fb("NumberOfTimes90DaysLate"))

    # 대안 변수
    telecom_payment_rate: float = Field(**fb("telecom_payment_rate"))
    utility_payment_rate: float = Field(**fb("utility_payment_rate"))
    spending_consistency: float = Field(**fb("spending_consistency"))
    regular_payment_count: int = Field(**fb("regular_payment_count"))
    app_login_frequency: int = Field(**fb("app_login_frequency"))

    # 씬파일러 여부 (0/1)
    # is_thin_filer: int = Field(ge=0, le=1) -> 서버에서 계산 후 입력

class predict_response(BaseModel):
    credit_score: int 
    approval_status: str
    default_probability: float 
    risk_grade: str 


# 3. (POST) XAI, 호출 시 거절 사유 top 5가 json 형식으로 반환
class explain_request(predict_request): # 입력값-/predict입력값 상속
    pass
class reason(BaseModel):          # 거절 사유 (feature_spec.build_rejection_reasons 형식)
    # 기본값 None: 값이 아직 안 채워져도 오류가 나지 않고, 기존 대시보드에도 영향 없음
    rank: Optional[int] = None                # 순위 (1부터)
    feature: str                              # 원래 변수명
    feature_name_kr: Optional[str] = None     # 화면에 보여줄 이름 (feature_spec 기준, 예: 부채 비율)
    feature_label: Optional[str] = None       # 이전 필드 이름 (호환용, feature_name_kr과 같은 값)
    current_value: Optional[float] = None     # 입력값, 미입력이면 null
    reference_value: Optional[float] = None   # 비교 기준 (Train 중앙값)
    shap_contribution: float                  # SHAP 기여도 (+: 부도 위험을 높이는 방향)
    explanation: str                          # 자연어 설명

class explain_response(BaseModel):
    approval_status: str
    rejection_reasons: List[reason]
    total_rejection_factors: Optional[int] = None   # 기여도가 +인 전체 변수 수
    shap_values: Dict[str, float]


# 3-1. (POST) XAI 이미지: 거절 사유 top 5, SHAP Waterfall, Summary, Dependence plot 이미지 반환
# 이미지 생성 실패시 해당 이미지만 null이 되고, 거절 사유, 나머지 이미지는 반환되도록 Optional 설정
class explain_plots_response(BaseModel):
    approval_status: str
    rejection_reasons: List[reason]
    total_rejection_factors: Optional[int] = None
    waterfall_png: Optional[str] = None
    summary_png: Optional[str] = None
    dependence_png: Optional[Dict[str, str]] = None


# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
class fairness_request(BaseModel):
    pass
class fairness_response(BaseModel): #fairness.py가 None을 반환할 수 있으므로 Optional[float]처리 -> None도 허용
    di_ratio: Optional[float]
    equalized_odds_tpr_diff : Optional[float]
    equalized_odds_fpr_diff : Optional[float]


# 5. (GET) mitigation 적용 전 DI ratio, tpr, fpr/ 적용 후 DI ratio, tpr, fpr
class fairness_mitigation_request(BaseModel):
    pass
class fairness_mitigation_response(BaseModel):
    before : fairness_response # 위의 공정성 검증 결과(di_raio, tpr, fpr)에서
    after : fairness_response # mitigation 여부만 바꿔서 봄


# 5-1. (GET) 공정성 상세 (대시보드용) - 구조가 깊어 dict로 통과시킴 (필드는 fairness.py의 fairness_summary 참고)
class fairness_detail_response(BaseModel):
    thresholds: Dict[str, float]
    info: Dict[str, Any]
    non_miti: Dict[str, Any]
    miti: Dict[str, Any]


# 6. (GET) 모델 정확도
# 최신 버전 메타에서 값 로드
class metrics_request(BaseModel): 
    pass
class metrics_response(BaseModel):
    auc : float
    ks : float
    precision: float
    recall : float
    f1 : float
    thin_filer_auc_uplift: Optional[float] = None   # 씬 파일러 AUC 향상 (대안 데이터 추가 효과, 예: +0.1134)
    psi: Optional[float] = None                     # 점수 분포 안정성 PSI (0.1 미만: 안정)


# 7. (GET) ANOVA-통계 검증
class tukey_result(BaseModel): # 사후검정 한 쌍 비교 결과
    comparison: str
    mean_diff: float
    p_adj: float
    significant: bool
    
class anova_result(BaseModel):
    test_name : str
    groups: List[str]
    group_means : Dict[str , float]
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    is_significant: bool
    post_hoc_tukey: Optional[List[tukey_result]] = None
class anova_request(BaseModel):
    pass
class anova_response(BaseModel):
    test_1_feature_group: anova_result
    test_2_algorithm: anova_result


# # 8. (GET) 반사실적 설명
class counterfactual_request(predict_request):       # 고객 입력은 /predict와 같음 (입력 범위 검사도 그대로)
    total_cfs: int = Field(3, ge=1, le=5)             # 제안 개수 (1~5)

class cf_change(BaseModel):                           # 바꿀 항목 하나
    feature: str                                      # 원래 변수명
    feature_name_kr: str                              # 화면 이름 (예: 무담보 한도 사용률)
    current_value: float                              # 지금 값
    suggested_value: float                            # 제안 값
    current_display: str                              # 단위 붙인 지금 값 (예: "85%")
    suggested_display: str                            # 단위 붙인 제안 값 (예: "42%")

class cf_option(BaseModel):                           # 방안 하나 (여러 항목을 함께 바꿈)
    rank: int                                         # 1부터, 바꾸는 항목이 적은 순
    changes: List[cf_change]
    credit_score: int                                 # 바꾼 뒤 예상 결과
    approval_status: str
    default_probability: float
    risk_grade: str

class counterfactual_response(BaseModel):
    original: predict_response                        # 지금 결과
    counterfactuals: List[cf_option]                  # 빈 리스트 = 이미 승인 또는 방안 없음
    varied_features: List[str]                        # 바꿔 본 변수
    fixed_features: List[str]                         # 바꿀 수 없다고 보고 고정한 변수
    message: str