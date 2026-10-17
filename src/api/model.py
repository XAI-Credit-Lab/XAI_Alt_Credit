from pydantic import BaseModel
from typing import List, Dict
# pydantic request/response schema 정의


# 1. (GET) 
class health_request(BaseModel):
    pass # GET 메서드는 조회 기능이라 입력 값이 없을거라 판단
class health_response(BaseModel):
    status: str # healthy
    model_version: str # 사용된 머신러닝 모델
    timestamp: str # 접속 시간


# 2. (POST) 고객 정보 JSON 입력 -> 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
class predict_request(BaseModel): 
    # 입력할 고객 정보, 모델에 들어갈 입력값 정의 필요
    a : int
class predict_response(BaseModel):
    credit_score: int 
    approval_statis: str
    default_probabilty: float 
    risk_grade: str 


# 3. (POST) XAI, 호출 시 거절 사유 top 5가 json 형식으로 반환
class explain_request(BaseModel):
    # 입력값 정의 필요(?)
    a : int
class reason(BaseModel): # 거절 사유
    feature : str # 변수명
    current_value : float # 현재 값
    explanation : str # 자연어 설명
    shap_contribution : float # shap 기여도
class explain_response(BaseModel):
    rejection_reason : List[reason] # 거절 사유 5개 리스트로
    shap_values : dict #?? json으로 응답 받음


# 4. (GET) 공정성 검증
class fairness_request(BaseModel):
    pass
class fairness_response(BaseModel):
    di_ratio: float
    equalized_odds_tpr_diff : float
    equalized_odds_fpr_dirr : float


# 5. (GET) 4결과에 mitigation 적용
class fairness_mitigation_request(BaseModel):
    pass
class fairness_mitigation_response(BaseModel):
    before : fairness_response
    after : fairness_response


# 6. (GET) 모델 정확도
class metrics_request(BaseModel): 
    pass
class metrics_response(BaseModel):
    auc : float
    ks : float
    precision: float
    recall : float
    f1 : float


# 7. (GET) ANOVA-통계 검증
class anova_result(BaseModel):
    test_name : str
    groups: List[str]
    group_means : Dict[str , float]
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    tukey : str
class anova_request(BaseModel):
    pass
class anova_response(BaseModel):
    tees_1 : anova_result
    test_2 : anova_result


# 8. (GET) 반사실적 설명
class counterfactual_request(BaseModel):
    pass
class counterfactual_response(BaseModel):
    # 구현 방법 필요
    pass


# # 테스트용 모델
# class add_request(BaseModel):
#     a : int
#     b : int
# class add_response(BaseModel):
#     result : int