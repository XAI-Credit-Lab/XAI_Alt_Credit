# src/api/schema.py
# pydantic request/response schema 정의
# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

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

    # 기존 금융 변수 
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0) 
    DebtRatio: float = Field(ge=0) 
    MonthlyIncome: Optional[float] = Field(None, ge=0) # 월 소득 - 입력 선택 (원본 데이터셋에 결측 있음) - 입력 없으면 None
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0) 
    NumberRealEstateLoansOrLines: int = Field(ge=0) 
    NumberOfDependents: Optional[int] = Field(None, ge=0) # 부양 가족 수 - 입력 선택 (원본 데이터셋에 결측 있음) - 입력 없으면 None
    card_history_months: int = Field(ge=0, le = 360)  

    # 기존 금융 변수-연체 관련-입력 선택
    # 컬럼 이름에 '-'가 있어 파이썬 변수명으로 못 씀 -> alias로 원래 이름 연결
    # 최근 2년간 연체 횟수, 최대값: (365 * 2) / 30, (365 * 2) / 60
    past_due_30_59: Optional[int] = Field(None, alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le = 24)
    past_due_60_89: Optional[int] = Field(None, alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le = 12)
    # 위 두 연체와 달리 '최근' 2년 기준이 아님, 원본 데이터셋에서의 이상치 제외 최대값 반영
    past_due_90: Optional[int] = Field(None, alias="NumberOfTimes90DaysLate", ge=0, le = 17) 

    # 대안 변수
    telecom_payment_rate: float = Field(ge=0, le=1)
    utility_payment_rate: float = Field(ge=0, le=1)
    spending_consistency: float = Field(ge=0, le=100)
    regular_payment_count: int = Field(ge=0)
    app_login_frequency: int = Field(ge=0)

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
class reason(BaseModel): # 거절 사유
    feature : str # 변수명
    current_value : Optional[float]   # 미입력이면 null
    shap_contribution : float # shap 기여도
    explanation : str # 자연어 설명
class explain_response(BaseModel):
    approval_status: str
    rejection_reasons : List[reason] # 거절 사유 5개
    shap_values : dict[str, float]


# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
# 모델 -> 공정성 import에서 공정성 -> 모델 import구조로 변경,
# src/fairness/fairness.py 호출
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


# 6. (GET) 모델 정확도
# models/model_meta_v1.0.json에서 값 로드
class metrics_request(BaseModel): 
    pass
class metrics_response(BaseModel):
    auc : float
    ks : float
    precision: float
    recall : float
    f1 : float


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
# class counterfactual_request(BaseModel):
#     pass
# class counterfactual_response(BaseModel):
#     # 구현 방법 필요
#     pass
