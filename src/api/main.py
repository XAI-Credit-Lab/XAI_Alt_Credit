from fastapi import FastAPI, APIRouter
import model
import time
from datetime import datetime
import sys, os


app = FastAPI() # FastAPI 객체 생성

router = APIRouter() # 엔드포인트 라우트

app.include_router(router)

# 엔드포인트 8개 정의
@router.get("/health", response_model = model.health_response)
def health():
    return {
        "status" : "healthy",
        "model_version" : "model_version", # 모델 결정 후 수정 예정
        "timestamp" : str(datetime.fromtimestamp(time.time()))
    }

@router.post("/predict", response_model = model.predict_response)
def predict(data : model.predict_request):
    #추가 예정
    return

@router.post("/explain", response_model = model.explain_response)
def explain(data : model.explain_request):
    # 추가 예정
    return

@router.get("/fairness", response_model = model.fairness_response)
def fairness():
    # 추가 예정
    return

@router.get("/fairness/mitigation", response_model = model.fairness_mitigation_response)
def fairness_mitigation():
    # 추가 예정
    return

@router.get("/metrics", response_model = model.metrics_response)
def metrics():
    # 추가 예정
    return

@router.get("/anova", response_model = model.anova_response)
def anova():
    # 추가 예정
    return

@router.get("/counterfactual", response_model = model.counterfactual_response)
def counterfactual():
    # 추가 예정
    return

# 다른 폴더에 있는 파일에 있는 함수 호출해보기 위한 테스트 엔드포인트
# @router.post("/add", response_model = model.add_response)
# def add(data: model.add_request):
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 폴더 접근
#     from exfunc.add import add_num # exfunc폴더에 있는 add.py파일에 있는 add_num함수 호출
#     res = add_num(data.a, data.b)
#     return model.add_response(result = res)