# src/api/main.py
# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

from fastapi import FastAPI, APIRouter, HTTPException
from src.api import schema
from contextlib import asynccontextmanager

import json
import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import PROJECT_ROOT, DEFAULT_SAVE, CreditPreprocessor
from src.data.simulator import is_thin_filer # 사용자 입력에 is_thin_filer feature 추가 용도

import time
from datetime import datetime

from src.explainer.xai import ModelExplainer
from src.fairness.fairness import model_fairness

MODEL_DIR = PROJECT_ROOT / "models"

IS_THIN_FILER = {"is_thin_filer"} # is_thin_filer feature: 사용자가 입력하지 않고 서버가 계산

art: dict = {}

# 서버가 켜질 때 무거운 파일(모델·전처리기·meta)을 딱 한 번 메모리에 올려 두고, 모든 엔드포인트가 그걸 공유해서 쓰게 하는 용도
# 서버가 켜질 때: uvicorn src.api.main:app 실행 시
# /predict든 SHAP용 /explain이든 모든 엔드포인트에서 art["model"], art["pre"], art["meta"]를 그대로 꺼내 쓸 수 있음
@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = json.loads((MODEL_DIR / f"model_meta_v1.0.json").read_text(encoding="utf-8")) # model_meta_v{MODEL_VERSION}.json파일 로드
    pre = CreditPreprocessor.load(DEFAULT_SAVE) # PROJECT_ROOT / "artifacts" / "preprocessor.joblib" 로드

    # {'meta': 메타 파일, 'pre': 전처리기 파일} 형식 됨
    art["meta"] = meta # model_meta.json
    art["model"] = joblib.load(MODEL_DIR / meta["best_model"]["file"]) # json파일에서 best_model의 file (ex: "logistic_regression_v1.0.pkl")
    art["pre"] = pre # 전처리기 파일
    art["num_cols"] = list(pre.numeric_cols_)       # 전처리기에서 수치형 원본 컬럼
    art["cat_cols"] = list(pre.categorical_cols_)   # 범주형 원본 컬럼 (is_thin_filer)
    art["raw_cols"] = art["num_cols"] + art["cat_cols"] # 수치형 컬럼, 범주형 컬럼 리스트 합쳐짐


    # 요청 스키마가 만들어 낼 컬럼 이름들  (alias 있으면 alias, 없으면 변수명) set
    schema_cols = {f.alias or name for name, f in schema.predict_request.model_fields.items()}

    # raw_cols    = 전처리기가 필요로 하는 컬럼 이름들
    # missing(결측 컬럼): 전처리기 입력 컬럼(raw_cols)에는 있는데 요청 스키마(schema_cols, IS_THIN_FILER)에 없음
    missing = [c for c in art["raw_cols"] if c not in schema_cols and c not in IS_THIN_FILER] 
    if missing: # 결측 걸럼 있으면
        print(f"[api] 경고: 요청 스키마에 없는 입력 컬럼(결측으로 처리됨): {missing}")

    # 전처리 결과 피처 = 모델 학습 피처인지 확인
    if pre.feature_names != meta["feature_names"]:
        raise RuntimeError("전처리기와 모델 메타의 feature_names가 다릅니다. 학습을 다시 실행하세요.")

    # SHAP 설명기: 서버 시작시 한 번만 생성
    background = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    background = background[meta["feature_names"]] # # 컬럼 순서를 모델 학습 순서로 맞춤

    # 자연어 설명용 원본 단위 평균·표준편차 (수치형 파이프라인의 StandardScaler에서 가져옴)
    # 결측 지시변수(missingindicator_...) 5개 포함 총 20개 변수
    scaler = pre.transformer_.named_transformers_["num"].named_steps["scaler"]
    scaler_names = list(scaler.feature_names_in_)
    raw_means  = {n: float(m) for n, m in zip(scaler_names, scaler.mean_)}
    raw_scales = {n: float(s) for n, s in zip(scaler_names, scaler.scale_)}

    art["explainer"] = ModelExplainer(
        art["model"],
        background,
        predict_fn = art["model"].predict_proba,
        class_index = 1,
        raw_means = raw_means,     # 자연어 설명의 '평균'을 원본 단위로
        raw_scales = raw_scales,   # 표준화 값을 원본 단위로 되돌리는 용도
        # is_thin_filer: 서버가 계산한 값 / missingindicator_: 사용자 입력이 아닌 결측 지시변수->거절사유 top에서 제외
        exclude_from_reasons = ("is_thin_filer", "missingindicator_")
    )

    # 공정성 지표
    try:
        non_miti, miti = model_fairness()
        art["fairness"] = {"before": non_miti, "after": miti}
    except Exception as e:
        print(f"[api] 경고: 공정성 계산 실패 — /fairness 사용 불가: {e}")
        art["fairness"] = None

    # anova 결과 json파일 로드
    art["anova"] = json.loads((PROJECT_ROOT / "reports" / "anova_results.json").read_text(encoding = "utf-8"))
    
    yield # 서버 실행중인 동안 코드 여기서 정지
    art.clear() # 서버 종료시 메모리에 올라온 객체 삭제(art["model"], art["pre"], art["meta"] 등)


app = FastAPI(lifespan=lifespan) # FastAPI 객체 생성

router = APIRouter() # 엔드포인트 라우트



# 사용자 입력을 모델이 학습 때 본 것과 똑같은 형태로 바꿔 주는 함수
# 결과: 모델에 바로 넣을 수 있는 전처리된 DataFrame
def to_features(req: schema.predict_request) -> pd.DataFrame: # 결과: pd.DataFrame(타입 힌트)
    raw = pd.DataFrame([req.model_dump(by_alias=True)])   # schema객체 -> 딕셔너리 -> 1행짜리 DataFrame

    # 컬럼을 전처리기 기준으로 맞춤(순서 정렬, 사용자 입력에 있는데 전처리기에 없는 컬럼 버림, 전처리기에 있는데 사용자 입력에 없는 컬럼 NaN)
    raw = raw.reindex(columns=art["raw_cols"])  
              
    raw[art["num_cols"]] = raw[art["num_cols"]].apply(pd.to_numeric, errors="coerce")  # None(숫자로 못 바꿈, 에러) -> NaN, float
    raw["is_thin_filer"] = is_thin_filer(raw).astype(int) # is_thin_filer: 시뮬레이터 규칙대로 계산
    return art["pre"].transform(raw)  #preprocessor.py의 transform(): 96/98 결측화, 상한, 대체, 스케일, 원-핫


# 모델 예측 함수 - /predict, /explain 공통 사용
def predict_result(X) -> dict:
    p = float(art["model"].predict_proba(X)[:, 1][0]) # 사용자 입력 모델 예측 결과
    bm = art["meta"]["best_model"]                    # 사용된 모델 정보
    return {
        "credit_score": int(round((1 - p) * 1000)),
        "approval_status": "APPROVED" if p <= bm["approval_cutoff"] else "REJECTED",
        "default_probability": round(p, 4),
        "risk_grade": bm["grade_labels"][int(np.searchsorted(bm["grade_cutoffs"], p, side="left"))]
    }


#================================================================
# 엔드포인트 8개 정의
#================================================================

# 1. (GET) 상태, 사용된 머신러닝 모델, 접속 시간 응답
@router.get("/health", response_model = schema.health_response)
def health():
    return {
        "status" : "healthy",
        "model_version" : art["meta"]["best_model"]["algorithm"],
        "timestamp" : str(datetime.fromtimestamp(time.time()))
    }

#================================================================================================================================
# # 2. (POST) 고객 정보 JSON 입력 -> 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
@router.post("/predict", response_model = schema.predict_response)
def predict(data : schema.predict_request):
    try:
        X = to_features(data)
    except Exception as e: # 전처리 실패 시 에러
        raise HTTPException(status_code = 422, detail = f"전처리 실패: {e}")

    return predict_result(X)  

#================================================================================================================================    
# 3. (POST) XAI, 호출 시 거절 사유 top 5가 json 형식으로 반환
@router.post("/explain", response_model = schema.explain_response)
def explain(data : schema.explain_request):
    try:
        X = pd.DataFrame(to_features(data), columns=art["meta"]["feature_names"]) # /explain 입력 데이터: DataFrame
    except Exception as e:
        raise HTTPException(status_code = 422, detail=f"전처리 실패: {e}")
    pred = predict_result(X)   # /predict
        # 사용자가 입력한 원본 값 (자연어 설명의 '현재 값' 표시용)
    # by_alias=True: 'NumberOfTime30-59DaysPastDueNotWorse' 같은 원래 컬럼 이름으로 키를 맞춤
    raw_input = data.model_dump(by_alias=True)

    try:
        result = art["explainer"].explain(X.iloc[0].to_dict(), top_k = 5, raw_values = raw_input)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"SHAP 설명 실패: {e}")

    return{
        "approval_status": pred["approval_status"],  # 모델 예측 결과
        "rejection_reasons": [{
            "feature": r["feature"],
            "current_value": r["value"],
            "shap_contribution": r["shap_val"],
            "explanation": r["description"]
        }
        for r in result["rejection_reasons"]
        ],
        "shap_values": result["shap_values"]
    }
    

#================================================================================================================================
# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
# src/fairness/fairness.py 호출
@router.get("/fairness", response_model = schema.fairness_response)
def fairness():
    if art["fairness"] is None:
        raise HTTPException(status_code=503, detail="공정성 지표를 계산하지 못했습니다.")
    non_miti = art["fairness"]["before"]
    return {
        "di_ratio": non_miti["di_ratio"],
        "equalized_odds_tpr_diff" : non_miti["equalized_odds_tpr_diff"],
        "equalized_odds_fpr_diff" : non_miti["equalized_odds_fpr_diff"]
    }

#================================================================================================================================
# 5. (GET) mitigation 적용 전 DI ratio, tpr, fpr/ 적용 후 DI ratio, tpr, fpr
@router.get("/fairness/mitigation", response_model = schema.fairness_mitigation_response)
def fairness_mitigation():
    if art["fairness"] is None:
        raise HTTPException(status_code=503, detail="공정성 지표를 계산하지 못했습니다.")
    non_miti = art["fairness"]["before"]
    miti = art["fairness"]["after"]
    return {
        "before": {
            "di_ratio": non_miti["di_ratio"],
            "equalized_odds_tpr_diff" : non_miti["equalized_odds_tpr_diff"],
            "equalized_odds_fpr_diff" : non_miti["equalized_odds_fpr_diff"]
        },
        "after": {
            "di_ratio": miti["di_ratio"],
            "equalized_odds_tpr_diff" : miti["equalized_odds_tpr_diff"],
            "equalized_odds_fpr_diff" : miti["equalized_odds_fpr_diff"]
        }
    }

#================================================================================================================================
# 6. (GET) 모델 정확도
# models/model_meta_v1.0.json에서 값 로드
@router.get("/metrics", response_model = schema.metrics_response)
def metrics():
    result = art["meta"]["final_metrics"]
    return {
        "auc" : result["test_auc"],
        "ks" : result["test_ks"],
        "precision": result["test_precision"],
        "recall" : result["test_recall"],
        "f1" : result["test_f1"]
    }

#================================================================================================================================
# 7. (GET) ANOVA-통계 검증 , response_model_exclude_none: post_hoc_tukey가 None이면 응답에서 키 자체를 뺌
@router.get("/anova", response_model = schema.anova_response, response_model_exclude_none = True)
def anova():
    return art["anova"]


#================================================================================================================================
# 8. (POST 추정) 반사실적 설명
# @router.get("/counterfactual", response_model = schema.counterfactual_response)
# def counterfactual():
#     # 추가 예정
#     return

# 호출시 등록된 엔드포인트만 복사 -> 파일 맨 아래로, 맨 마지막에 호출
app.include_router(router)