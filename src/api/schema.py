# pydantic request/response schema 정의

from pydantic import BaseModel
from typing import List, Dict

# 1. (GET) 서버 상태, 사용된 머신러닝 모델, 접속 시간 응답
class health_request(BaseModel):
    pass # GET 메서드는 조회 기능이라 입력 값이 없을거라 판단
class health_response(BaseModel):
    status: str 
    model_version: str 
    timestamp: str 


# 2. (POST) 고객 정보 JSON 입력 -> 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
# 모델 입력값 대안 변수 5개인 것으로 가정
class predict_request(BaseModel): 
    telecom_payment_rate: float
    utility_payment_rate: float
    spending_consistency: float
    regular_payment_count: int #
    app_login_frequency: int #
class predict_response(BaseModel):
    credit_score: int 
    approval_status: str
    default_probabilty: float 
    risk_grade: str 


# # 3. (POST) XAI, 호출 시 거절 사유 top 5가 json 형식으로 반환
# class explain_request(BaseModel):
#     # 입력값 정의 필요(?)
#     a : int
# class reason(BaseModel): # 거절 사유
#     feature : str # 변수명
#     current_value : float # 현재 값
#     explanation : str # 자연어 설명
#     shap_contribution : float # shap 기여도
# class explain_response(BaseModel):
#     rejection_reason : List[reason] # 거절 사유 5개 리스트로
#     shap_values : dict #?? json으로 응답 받음


# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
# 모델(model.py)에 fairness.py호출하는 함수(fairness()) 있는 것으로 가정
# fairness.py: mitigation한 함수(miti()), 안 한 함수(non_miti())있다고 가정
# 모델(model.py)의 fairness()에서 fairness.py의 miti()호출에서 miti 변수에 할당,
# non_miti()호출에서 non_miti 변수에 할당하고, return miti, non_miti로 결과 반환한다고 가정
class fairness_request(BaseModel):
    pass
class fairness_response(BaseModel):
    di_ratio: float
    equalized_odds_tpr_diff : float
    equalized_odds_fpr_diff : float


# 5. (GET) 4모델 상속(?), 적용 전 DI ratio, tpr, fpr/ 적용 후 DI ratio, tpr, fpr
class fairness_mitigation_request(BaseModel):
    pass
class fairness_mitigation_response(BaseModel):
    before : fairness_response # 위의 공정성 검증 결과(di_raio, tpr, fpr)에서
    after : fairness_response # mitigation 여부만 바꿔서 봄


# 6. (GET) 모델 정확도
# 모델안에 정확도(auc, ks, precision, recall, f1) 계산하는 함수 따로 있다고 가정
class metrics_request(BaseModel): 
    pass
class metrics_response(BaseModel):
    auc : float
    ks : float
    precision: float
    recall : float
    f1 : float


# # 7. (GET) ANOVA-통계 검증
# class anova_result(BaseModel):
#     test_name : str
#     groups: List[str]
#     group_means : Dict[str , float]
#     f_statistic: float
#     p_value: float
#     effect_size_eta_squared: float
#     tukey : str
# class anova_request(BaseModel):
#     pass
# class anova_response(BaseModel):
#     tees_1 : anova_result
#     test_2 : anova_result


# # 8. (GET) 반사실적 설명
# class counterfactual_request(BaseModel):
#     pass
# class counterfactual_response(BaseModel):
#     # 구현 방법 필요
#     pass


# # 테스트용 모델
# class add_request(BaseModel):
#     a : int
#     b : int
# class add_response(BaseModel):
#     result : int