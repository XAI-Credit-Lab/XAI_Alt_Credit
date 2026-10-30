from fastapi import FastAPI, APIRouter
import schema
import loader
import time
from datetime import datetime
import sys, os
from pathlib import Path


app = FastAPI() # FastAPI 객체 생성

router = APIRouter() # 엔드포인트 라우트

app.include_router(router)

ROOT_DIR = Path(__file__).resolve().parents[2] # 프로젝트 루트/상위 폴더 접근하기 위함

#================================================================
# 엔드포인트 8개 정의
#================================================================

# 1. (GET) 서버 상태, 사용된 머신러닝 모델, 접속 시간 응답
@router.get("/health", response_model = schema.health_response)
def health(): # mkflow json에서 import
    return {
        "status" : "healthy",
        "model_version" : loader.load_health_model_version(), # 모델 결정 후 수정 예정
        "timestamp" : str(datetime.fromtimestamp(time.time()))
    }

# 2. (POST) 고객 정보 JSON 입력 -> 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
# 모델 입력값 대안 변수 5개인 것으로 가정
@router.post("/predict", response_model = schema.predict_response) # json 입력받기
def predict(data : schema.predict_request):
    sys.path.insert(0, str(ROOT_DIR))
    from src.data import model # 모델 파일
    result = model.funcname(schema.predict_request) # 입력받은 json 모델에 넣음
    return { # json return
        "credit_score": result["credit_score"],
        "approval_status": result["approval_status"],
        "default_probabilty": result["default_probabilty"],
        "risk_grade": result["risk_grade"]
    }
    
# 3. (POST) XAI, 호출 시 거절 사유 top 5가 json 형식으로 반환
# 모델(model.py)에 /explain 폴더의 XAI.py 호출하는 함수(explain()) 있는 것으로 가정
'''
@router.post("/explain", response_model = schema.explain_response)
def explain(data : schema.explain_request):
    sys.path.insert(0, str(ROOT_DIR)) 
    from src.data import model
    result = model.explain()
    return{
        
    }
'''


# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
# 모델(model.py)에 fairness.py호출하는 함수(fairness()) 있는 것으로 가정
# fairness.py: mitigation한 함수(miti()), 안 한 함수(non_miti())있다고 가정
# 모델(model.py)의 fairness()에서 fairness.py의 miti()호출에서 miti 변수에 할당,
# non_miti()호출에서 non_miti 변수에 할당하고, return miti, non_miti로 결과 반환한다고 가정
@router.get("/fairness", response_model = schema.fairness_response)
def fairness():
    sys.path.insert(0, str(ROOT_DIR)) 
    from src.data import model
    non_miti, miti = model.fairness()
    return{
        "di_ratio": non_miti["di_ratio"],
        "equalized_odds_tpr_diff" : non_miti["equalized_odds_tpr_diff"],
        "equalized_odds_fpr_diff" : non_miti["equalized_odds_fpr_diff"]
    }

# 5. (GET) 4모델 상속(?) 적용 전 DI ratio, tpr, fpr/ 적용 후 DI ratio, tpr, fpr
@router.get("/fairness/mitigation", response_model = schema.fairness_mitigation_response)
def fairness_mitigation():
    sys.path.insert(0, str(ROOT_DIR)) 
    from src.data import model
    non_miti, miti = model.fairness()
    return{
        "before": { # mitigation 적용 전
            "di_ratio": non_miti["di_ratio"],
            "equalized_odds_tpr_diff" : non_miti["equalized_odds_tpr_diff"],
            "equalized_odds_fpr_diff" : non_miti["equalized_odds_fpr_diff"]
        },
        "after": { # mitigation 적용 후
            "di_ratio": miti["di_ratio"],
            "equalized_odds_tpr_diff" : miti["equalized_odds_tpr_diff"],
            "equalized_odds_fpr_diff" : miti["equalized_odds_fpr_diff"]
        }
    }

# 6. (GET) 모델 정확도
# 모델안에 정확도(auc, ks, precision, recall, f1) 계산하는 함수(metric()) 따로 있다고 가정
@router.get("/metrics", response_model = schema.metrics_response)
def metrics():
    sys.path.insert(0, str(ROOT_DIR))
    from src.data import model # 모델 파일
    result = model.metric() # 입력받은 json 모델에 넣음
    return { # json return
        "auc" : result["auc"],
        "ks" : result["ks"],
        "precision": result["precision"],
        "recall" : result["recall"],
        "f1" : result["f1"]
    }
    
# # 7. (GET) ANOVA-통계 검증
# @router.get("/anova", response_model = schema.anova_response)
# def anova():
#     sys.path.insert(0, str(ROOT_DIR)) # 루트
#     from src/statistics import anova # anova.py 호출
#     # 추가 예정
#     return

# @router.get("/counterfactual", response_model = schema.counterfactual_response)
# def counterfactual():
#     # 추가 예정
#     return

# 다른 폴더에 있는 파일에 있는 함수 호출해보기 위한 테스트 엔드포인트
# @router.post("/add", response_model = model.add_response)
# def add(data: model.add_request):
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 폴더 접근
#     from exfunc.add import add_num # exfunc폴더에 있는 add.py파일에 있는 add_num함수 호출
#     res = add_num(data.a, data.b)
#     return model.add_response(result = res)