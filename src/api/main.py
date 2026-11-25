# src/api/main.py
# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

# 반사실적 설명 엔드포인트 추가

from fastapi import FastAPI, APIRouter, HTTPException
from src.api import schema
from contextlib import asynccontextmanager

import json
import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import PROJECT_ROOT, DEFAULT_SAVE, VALID_INCOME_MIN, CreditPreprocessor
from src.data.simulator import is_thin_filer # 사용자 입력에 is_thin_filer feature 추가 용도
from src.data.feature_spec import format_value, label, FEATURE_SPEC

from datetime import datetime, timezone, timedelta
import threading
import re

from src.explainer.xai import ModelExplainer
from src.fairness.fairness import model_fairness, model_fairness_detail

KST = timezone(timedelta(hours=9))   # 한국 표준시

MODEL_DIR = PROJECT_ROOT / "models"

# model_meta_v{버전}.json  (예: model_meta_v1.3.json -> "1.3")
META_PATTERN = re.compile(r"model_meta_v(\d+(?:\.\d+)*)\.json$")

def latest_meta_path():
    candidates = []
    for p in MODEL_DIR.glob("model_meta_v*.json"):
        m = META_PATTERN.match(p.name)
        if m:
            candidates.append((tuple(int(x) for x in m.group(1).split(".")), p))
    if not candidates:
        raise FileNotFoundError(f"{MODEL_DIR}에 model_meta_v*.json 파일이 없습니다.")
    return max(candidates)[1]

def meta_version(path) -> str:
    # 메타 파일 이름에서 버전 추출: model_meta_v1.3.json -> "v1.3"
    return "v" + META_PATTERN.match(path.name).group(1)

def unwrap_model(model):
    # SMOTE 선택돼도 TreeExplainer 사용 가능하도록
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model

# 사용자가 입력하지 않는 컬럼 (경고 검사에서 제외)
# is_thin_filer: to_features()에서 서버가 계산
# past_due_code_flag: 전처리기 transform()에서 연체 컬럼의 96/98 코드 여부로 계산
SERVER_COMPUTED = {"is_thin_filer", "past_due_code_flag"}

art: dict = {}

# 서버가 켜질 때 무거운 파일(모델·전처리기·meta)을 딱 한 번 메모리에 올려 두고, 모든 엔드포인트가 그걸 공유해서 쓰게 하는 용도
# 서버가 켜질 때: uvicorn src.api.main:app 실행 시
# /predict든 SHAP용 /explain이든 모든 엔드포인트에서 art["model"], art["pre"], art["meta"]를 그대로 꺼내 쓸 수 있음
@asynccontextmanager
async def lifespan(app: FastAPI):
    meta_path = latest_meta_path()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) # 모델
    pre_file = meta.get("preprocessor_file", DEFAULT_SAVE.name)
    pre = CreditPreprocessor.load(DEFAULT_SAVE.parent / pre_file) # 전처리기
    print(f"[api] 모델 메타 {meta_path.name} / 전처리기 {pre_file} / 모델 {meta['best_model']['file']}")
    # {'meta': 메타 파일, 'pre': 전처리기 파일} 형식 됨
    art["meta"] = meta # model_meta.json
    # /health 표시용: "알고리즘 버전" (예: "lightgbm v1.3")
    art["model_version"] = f'{meta["best_model"]["algorithm"]} {meta_version(meta_path)}'
    art["model"] = joblib.load(MODEL_DIR / meta["best_model"]["file"])
    art["pre"] = pre # 전처리기 파일
    art["num_cols"] = list(pre.numeric_cols_)       # 전처리기에서 수치형 원본 컬럼
    art["cat_cols"] = list(pre.categorical_cols_) 
    art["raw_cols"] = art["num_cols"] + art["cat_cols"] # 수치형 컬럼, 범주형 컬럼 리스트 합쳐짐


    # 요청 스키마가 만들어 낼 컬럼 이름들  (alias 있으면 alias, 없으면 변수명) set
    schema_cols = {f.alias or name for name, f in schema.predict_request.model_fields.items()}

    # raw_cols    = 전처리기가 필요로 하는 컬럼 이름들
    # missing(결측 컬럼): 전처리기 입력 컬럼(raw_cols)에는 있는데 요청 스키마(schema_cols)에 없음
    missing = [c for c in art["raw_cols"] if c not in schema_cols and c not in SERVER_COMPUTED]
    if missing: # 결측 걸럼 있으면
        print(f"[api] 경고: 요청 스키마에 없는 입력 컬럼(결측으로 처리됨): {missing}")

    # 전처리 결과 피처 = 모델 학습 피처인지 확인
    if pre.feature_names != meta["feature_names"]:
        raise RuntimeError("전처리기와 모델 메타의 feature_names가 다릅니다. 학습을 다시 실행하세요.")

    background = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv", index_col=0)
    lack = [c for c in meta["feature_names"] if c not in background.columns]
    if lack:
        raise RuntimeError(f"X_train.csv가 현재 전처리기({pre_file})와 맞지 않습니다. 없는 피처: {lack}\n"
                           "  -> python -m src.data.preprocessor 로 분할 파일을 다시 만드세요.")
    background = background[meta["feature_names"]]   # 컬럼 순서를 모델 학습 순서로 맞춤

    # 원본 입력에 없는 값 복원용
    scaler = pre.transformer_.named_transformers_["num"].named_steps["scaler"]
    scaler_names = list(scaler.feature_names_in_)
    raw_means  = {n: float(m) for n, m in zip(scaler_names, scaler.mean_)}
    raw_scales = {n: float(s) for n, s in zip(scaler_names, scaler.scale_)}

    # 설명기 - 서버 시작시 한 번 만 생성
    art["explainer"] = ModelExplainer(
        unwrap_model(art["model"]),
        background,
        predict_fn = art["model"].predict_proba,
        class_index = 1,
        raw_means = raw_means,    # 원본 입력에 없는 값을 원래 단위로 복원 (StandardScaler 평균)
        raw_scales = raw_scales,   # 위 복원용 (StandardScaler 표준편차)
        raw_medians = pre.medians_,        # 거절 사유 문장의 비교 기준 (Train 중앙값)
        # is_thin_filer: 서버가 계산한 값 / missingindicator_: 사용자 입력이 아닌 결측 지시변수
        # past_due_code_flag: 전처리기가 만드는 기록 이상 코드 여부 (API 입력으로는 항상 0) -> 거절사유 top에서 제외
        exclude_from_reasons = ("is_thin_filer", "missingindicator_", "past_due_code_flag")
    )

    # 반사실적 설명용 DiCE 데이터 - 서버 시작 시 한 번만 (실패해도 나머지 API는 동작)
    try:
        import dice_ml   # 여기서만 import: dice-ml이 없어도 서버 전체가 멈추지 않도록
        proc = PROJECT_ROOT / "data" / "processed"
        cf_cols = [c for c in art["raw_cols"] if c in schema_cols]   # 사용자가 입력하는 컬럼만 (is_thin_filer 제외)
        df = _cf_clean(pd.read_csv(proc / "raw_train.csv", index_col=0)[cf_cols], pre.medians_)
        # 제안 상한: 입력 상한이 없는 변수(소득·건수)는 학습 데이터 99% 분위수까지
        art["cf_upper"] = {c: FEATURE_SPEC[c]["max"] if FEATURE_SPEC[c].get("max") is not None
                              else float(df[c].quantile(0.99)) for c in cf_cols}
        df[CF_OUTCOME] = pd.read_csv(proc / "y_train.csv", index_col=0).iloc[:, 0].astype(int).values
        art["cf_cols"] = cf_cols
        # DiCE는 자릿수를 데이터에서 추측(최빈값 기준)해서 납부율이 0.1 단위로 나오고 1.1처럼 범위를 넘을 수 있음
        # -> 변수 종류별 자릿수를 직접 지정
        precision = {c: CF_PRECISION.get(FEATURE_SPEC[c].get("kind"), 2) for c in cf_cols}
        art["cf_data"] = dice_ml.Data(dataframe=df, continuous_features=cf_cols, outcome_name=CF_OUTCOME,
                                      continuous_features_precision=precision)
    except Exception as e:
        print(f"[api] 경고: 반사실적 설명 준비 실패 — /counterfactual 사용 불가: {e}")
        art["cf_data"] = None


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



# 원본 입력 DataFrame(여러 행 가능)을 모델이 학습 때 본 것과 똑같은 형태로 바꿔 주는 함수
# /counterfactual(DiCE)이 후보 수백 개를 한꺼번에 예측할 때도 사용
def raw_to_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reindex(columns=art["raw_cols"])
    raw[art["num_cols"]] = raw[art["num_cols"]].apply(pd.to_numeric, errors="coerce")
    raw["is_thin_filer"] = is_thin_filer(raw).astype(int)   # 행마다 다시 계산
    return art["pre"].transform(raw)

# 사용자 입력(schema 객체) 1명 -> 모델에 바로 넣을 수 있는 전처리된 DataFrame
def to_features(req: schema.predict_request) -> pd.DataFrame:
    return raw_to_features(pd.DataFrame([req.model_dump(by_alias=True)]))

# 소득이 미입력이면 deptratio를 중앙값으로 대체되었다고 안내
def display_values(req: schema.predict_request) -> dict:
    raw_input = req.model_dump(by_alias=True)
    income = raw_input.get("MonthlyIncome")
    if income is None or income <= VALID_INCOME_MIN:
        raw_input["DebtRatio"] = None
    return raw_input

def note_unused_debt_ratio(data: schema.predict_request, raw_input: dict, reasons: list) -> None:
    """월 소득이 없어 부채 비율이 반영되지 않은 경우, '미입력' 대신 이유를 알려 줌"""
    if raw_input.get("DebtRatio") is None and data.DebtRatio is not None:
        for r in reasons:
            if r["feature"] == "DebtRatio":
                ref = format_value("DebtRatio", r["reference_value"])
                r["explanation"] = (f"월 소득이 없어 입력한 부채 비율을 반영하지 않고 중앙값({ref})으로 계산했습니다 "
                                    f"(기여도: {r['shap_contribution']:+.3f})")

# 부도 확률 p -> 신용 점수, 승인 여부, 등급 (/predict, /explain, /counterfactual 공통)
def result_from_p(p: float) -> dict:
    bm = art["meta"]["best_model"]                    # 사용된 모델 정보
    return {
        "credit_score": int(round((1 - p) * 1000)),
        "approval_status": "APPROVED" if p <= bm["approval_cutoff"] else "REJECTED",
        "default_probability": round(p, 4),
        "risk_grade": bm["grade_labels"][int(np.searchsorted(bm["grade_cutoffs"], p, side="left"))]
    }

# 모델 예측 함수 - /predict, /explain 공통 사용
def predict_result(X) -> dict:
    return result_from_p(float(art["model"].predict_proba(X)[:, 1][0]))

#================================================================
# 반사실적 설명(DiCE)용 설정과 도우미
#================================================================
CF_OUTCOME = "target"   # DiCE 내부용 결과 컬럼 이름

# 바꿀 수 있는 변수와 방향 ("down": 줄이기만, "up": 늘리기만, "both": 양쪽)
# 목록에 없는 변수는 고정: 카드 이용 기간(시간이 지나야 늘어남), 연체 기록(지울 수 없음), 부양 가족 수, 부동산 대출 수
CF_ACTIONABLE = {
    "RevolvingUtilizationOfUnsecuredLines": "down",   # 카드·신용대출 사용액 줄이기
    "DebtRatio": "down",                              # 부채 상환
    "MonthlyIncome": "up",                            # 소득 증가
    "NumberOfOpenCreditLinesAndLoans": "down",        # 쓰지 않는 카드·대출 정리
    "telecom_payment_rate": "up",                     # 통신비 제때 내기
    "utility_payment_rate": "up",                     # 공과금 제때 내기
    "spending_consistency": "up",                     # 소비 패턴 일정하게
    # regular_payment_count, app_login_frequency: 고정
    #   "앱에 76번 로그인하세요" 같은 제안은 신용 개선 행동이 아니라 모델의 허점을 이용하는 것이라 제외
}

# 한 번에 늘릴 수 있는 최대 배수 (현실성: 월 소득 8,000 -> 22,000달러 같은 제안 방지)
CF_MAX_UP_RATIO = {"MonthlyIncome": 1.3}
# 한 번에 늘릴 수 있는 최대 폭 (현재 값 + 이 값까지)
CF_MAX_UP_STEP = {"spending_consistency": 30}   # 소비 일관성은 최대 30점까지만 올리도록

# 제안 값의 소수 자릿수 (feature_spec 의 kind 기준). ratio 2자리 = 1%p 단위
CF_PRECISION = {"ratio": 2, "score": 0, "amount": 0, "count": 0}

_cf_lock = threading.Lock()   # DiCE는 난수를 쓰므로 한 번에 하나씩     


class _RawModel:
    """원본 입력 DataFrame -> 전처리 -> 모델 확률. DiCE가 sklearn 모델처럼 호출
    keep_missing: 사용자가 입력하지 않은 변수. DiCE에는 결측을 넣을 수 없어 중앙값으로 채워 넘기지만,
                  모델에 넣기 직전 다시 NaN으로 돌려서 원래 예측(결측 처리)과 같게 만듦"""
    def __init__(self, keep_missing=()):
        self.keep_missing = list(keep_missing)

    def predict_proba(self, raw: pd.DataFrame):
        raw = raw.drop(columns=[CF_OUTCOME], errors="ignore").copy()
        if self.keep_missing:
            raw[self.keep_missing] = np.nan
        return art["model"].predict_proba(raw_to_features(raw))

    def predict(self, raw: pd.DataFrame):
        cutoff = art["meta"]["best_model"]["approval_cutoff"]
        return (self.predict_proba(raw)[:, 1] > cutoff).astype(int)


def _cf_clean(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """DiCE용 학습 데이터 정리: 입력 범위 밖 값(96/98 코드, 금액이 들어간 DebtRatio 등)은 중앙값,
    건수는 정수로 (DiCE가 소수점 없는 값을 제안하도록)"""
    df = df.apply(pd.to_numeric, errors="coerce").copy()
    for c in df.columns:
        spec = FEATURE_SPEC.get(c, {})
        if spec.get("min") is not None:
            df.loc[df[c] < spec["min"], c] = np.nan
        if spec.get("max") is not None:
            df.loc[df[c] > spec["max"], c] = np.nan
        df[c] = df[c].fillna(medians.get(c, df[c].median()))
        if spec.get("kind") == "count":
            df[c] = df[c].round().astype(int)
    return df

# matplotlib(pyplot)은 여러 요청이 동시에 그림을 그리면 그림이 섞일 수 있음 -> 한 번에 하나씩 그리도록 잠금
# 한 요청이 처리되는 동안 다른 요청은 대기
_plot_lock = threading.Lock()


# 모델 전체 그림(summary, dependence): 고객 입력과 상관없이 항상 같음
# -> 처음 요청 때 한 번만 그려서 art에 저장해 두고, 이후 요청은 저장된 값을 그대로 사용
def global_plots() -> dict:
    if "global_plots" in art:            # 이미 그려 뒀으면 바로 반환
        return art["global_plots"]

    with _plot_lock: # 여기부터 요청들을 줄 세우고 아래의 코드를 한 번에 하나씩 순서대로 처리 
        if "global_plots" in art:        # 기다리는 동안 다른 요청이 먼저 그렸으면 그대로 사용
            return art["global_plots"]

        plots = {"summary_png": None, "dependence_png": None}   # 실패한 그림은 None
        try:
            plots["summary_png"] = art["explainer"].plot_summary(max_display=15)
        except Exception as e:
            print(f"[api] 경고: summary plot 생성 실패: {e}")
        try:
            plots["dependence_png"] = art["explainer"].plot_dependence()   # 중요도 상위 연속형 변수 2개
        except Exception as e:
            print(f"[api] 경고: dependence plot 생성 실패: {e}")

        art["global_plots"] = plots
        return plots


#================================================================
# 엔드포인트 8개 정의
#================================================================

# 1. (GET) 상태, 사용된 머신러닝 모델, 접속 시간 응답
@router.get("/health", response_model = schema.health_response)
def health():
    return {
        "status" : "healthy",
        "model_version" : art["model_version"],   # 예: "lightgbm v1.3"
        "timestamp" : datetime.now(KST).isoformat(timespec="seconds")
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
    raw_input = display_values(data)

    # 거절 사유 기준
    # 승인된 고객: 의미 있는 요인(xai.py의 min_contribution 초과)만 표시
    # 거절된 고객: 거절 사유를 항상 보여주기 위해 기준 없이 부도 확률을 높인 상위 5개 표시
    use_min = pred["approval_status"] == "APPROVED"

    try:
        result = art["explainer"].explain(X.iloc[0].to_dict(), top_k = 5, raw_values = raw_input,
                                          use_min_contribution = use_min)
        note_unused_debt_ratio(data, raw_input, result["rejection_reasons"])
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"SHAP 설명 실패: {e}")

    return {
        "approval_status": pred["approval_status"],
        "rejection_reasons": result["rejection_reasons"],          # feature_spec 형식 그대로
        "total_rejection_factors": result["total_rejection_factors"],
        "shap_values": result["shap_values"],
    }

#================================================================================================================================
# 3-1. (POST) XAI 이미지: 거절 사유 top 5 + SHAP Waterfall(이 고객) / Summary·Dependence(모델 전체) 이미지
# 이미지는 PNG를 base64 문자열로 인코딩해서 반환, 생성에 실패한 이미지는 None
@router.post("/explain/plots", response_model = schema.explain_plots_response)
def explain_plots(data : schema.explain_request):
    # (1) 전처리 + 예측 (/explain과 동일)
    try:
        X = pd.DataFrame(to_features(data), columns=art["meta"]["feature_names"])
    except Exception as e:
        raise HTTPException(status_code = 422, detail=f"전처리 실패: {e}")
    pred = predict_result(X)
    raw_input = display_values(data)

    # (2) 거절 사유 top 5 (/explain과 동일)
    #     승인된 고객: 의미 있는 요인(xai.py의 min_contribution 초과)만 표시
    #     거절된 고객: 거절 사유를 항상 보여주기 위해 기준 없이 부도 확률을 높인 상위 5개 표시
    use_min = pred["approval_status"] == "APPROVED"

    try:
        result = art["explainer"].explain(X.iloc[0].to_dict(), top_k = 5, raw_values = raw_input,
                                          use_min_contribution = use_min)
        # 월 소득이 없어 부채 비율이 반영되지 않은 경우, '미입력' 대신 이유를 알려 줌
        note_unused_debt_ratio(data, raw_input, result["rejection_reasons"])

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"SHAP 설명 실패: {e}")

    # (3) 이 고객의 waterfall plot: 요청마다 새로 그림 (잠금 안에서 한 번에 하나씩)
    #     (2)에서 거절 사유를 계산할 때 구한 SHAP 값을 그대로 넘김
    #     -> SHAP을 다시 계산하지 않고, 거절 사유와 그림이 같은 값을 사용
    waterfall_png = None
    try:
        with _plot_lock:
            waterfall_png = art["explainer"].plot_waterfall(
                X,
                max_display = 10,
                display_values = raw_input,
                shap_values = result["shap_values"],   # {변수명: SHAP 값}
                base_value = result["base_value"],     # 기준값 (모든 변수의 기여가 0일 때의 예측값)
            )
    except Exception as e:
        print(f"[api] 경고: waterfall plot 생성 실패: {e}")

    # (4) 모델 전체 summary / dependence plot: 처음 한 번만 그리고 재사용
    #     ※ global_plots()는 안에서 직접 잠금을 걸기 때문에 반드시 with _plot_lock 블록 바깥에서 호출
    plots = global_plots()

    return {
        "approval_status": pred["approval_status"],
        "rejection_reasons": result["rejection_reasons"],
        "total_rejection_factors": result["total_rejection_factors"],
        "waterfall_png": waterfall_png,
        "summary_png": plots["summary_png"],
        "dependence_png": plots["dependence_png"],
    }
    

#================================================================================================================================
# 4. (GET) 공정성 검증(non-mitigation) -> DI ratio, tpr, fpr
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
# 5-1. (GET) 공정성 상세 (대시보드용): 연령 DI, 속성별 판정, 집단표, 판정 기준값
#      대시보드는 .pkl/파일을 직접 읽지 않고 HTTP로만 데이터를 받으므로 필요한 엔드포인트
@router.get("/fairness/detail", response_model = schema.fairness_detail_response)
def fairness_detail():
    if art["fairness"] is None:
        raise HTTPException(status_code=503, detail="공정성 지표를 계산하지 못했습니다.")
    return model_fairness_detail()

#================================================================================================================================
# 6. (GET) 모델 정확도
# 최신 버전 meta_json에서 값 로드
@router.get("/metrics", response_model = schema.metrics_response)
def metrics():
    result = art["meta"]["final_metrics"]
    return {
        "auc" : result["test_auc"],
        "ks" : result["test_ks"],
        "precision": result["test_precision"],
        "recall" : result["test_recall"],
        "f1" : result["test_f1"],
        # 이전 버전 메타에는 없을 수 있음 -> 없으면 None(null)
        "thin_filer_auc_uplift": result.get("thin_filer_auc_uplift"),
        "psi": result.get("psi"),
    }

#================================================================================================================================
# 7. (GET) ANOVA-통계 검증 , response_model_exclude_none: post_hoc_tukey가 None이면 응답에서 키 자체를 뺌
@router.get("/anova", response_model = schema.anova_response, response_model_exclude_none = True)
def anova():
    return art["anova"]


#================================================================================================================================
# 8. (POST) 반사실적 설명: 거절된 고객이 무엇을 얼마나 바꾸면 승인될지 (DiCE)
@router.post("/counterfactual", response_model = schema.counterfactual_response)
def counterfactual(data : schema.counterfactual_request):
    if art.get("cf_data") is None:
        raise HTTPException(status_code = 503, detail = "반사실적 설명을 사용할 수 없습니다.")
    import dice_ml

    # (1) 지금 결과 - 이미 승인이면 찾을 필요 없음
    try:
        pred = predict_result(to_features(data))
    except Exception as e:
        raise HTTPException(status_code = 422, detail = f"전처리 실패: {e}")
    if pred["approval_status"] == "APPROVED":
        return {"original": pred, "counterfactuals": [], "varied_features": [],
                "fixed_features": [], "message": "이미 승인된 고객입니다."}

    # (2) DiCE에 넣을 고객 1명: 미입력 값은 중앙값으로 임시 채움 (모델에는 다시 NaN으로 들어감)
    cols = art["cf_cols"]
    raw = data.model_dump(by_alias=True, exclude={"total_cfs"})
    query = pd.DataFrame([{c: raw.get(c) for c in cols}]).apply(pd.to_numeric, errors="coerce")
    missing = [c for c in cols if pd.isna(query.at[0, c])]
    for c in missing:
        query.at[0, c] = art["pre"].medians_.get(c, 0.0)
    for c in cols:
        if FEATURE_SPEC.get(c, {}).get("kind") == "count":
            query[c] = query[c].round().astype(int)

    # (3) 바꿀 변수와 범위 (방향 제한, 미입력 변수 제외, 소득이 없으면 부채 비율 제외)
    vary, ranges = [], {}
    for c, direction in CF_ACTIONABLE.items():
        if c not in cols or c in missing or (c == "DebtRatio" and "MonthlyIncome" in missing):
            continue
        cur = float(query.at[0, c])
        lo = float(FEATURE_SPEC[c].get("min") or 0.0)
        hi = max(art["cf_upper"][c], cur)
        if direction == "down":
            hi = cur
        elif direction == "up":
            lo = cur
            if c in CF_MAX_UP_RATIO and cur > 0:
                hi = min(hi, cur * CF_MAX_UP_RATIO[c])
            if c in CF_MAX_UP_STEP:
                hi = min(hi, cur + CF_MAX_UP_STEP[c])
        if hi - lo > 1e-9:                   # 이미 끝값(예: 납부율 100%)이면 더 바꿀 수 없음
            vary.append(c)
            ranges[c] = [lo, hi]
    fixed = [c for c in cols if c not in vary]

    # (4) DiCE로 방안 찾기 (승인 기준 = approval_cutoff, 난수 고정 -> 같은 입력이면 같은 결과)
    cutoff = art["meta"]["best_model"]["approval_cutoff"]
    wrapper = _RawModel(keep_missing = missing)
    cfs = None
    if vary:
        dice = dice_ml.Dice(art["cf_data"],
                            dice_ml.Model(model = wrapper, backend = "sklearn", model_type = "classifier"),
                            method = "random")
        with _cf_lock:
            try:
                res = dice.generate_counterfactuals(
                    query, total_CFs = data.total_cfs, desired_class = 0,   # 0 = 부도 아님(승인 쪽)
                    features_to_vary = vary, permitted_range = ranges,
                    stopping_threshold = cutoff, random_seed = 42)
                cfs = res.cf_examples_list[0].final_cfs_df
            except Exception as e:             # 방안을 못 찾으면 예외를 내기도 함
                print(f"[api] 반사실적 설명 탐색 실패: {e}")

    # (5) API 기준으로 다시 채점 -> 실제로 승인되는 방안만, 바뀐 항목만
    options = []
    if cfs is not None and len(cfs):
        cfs = cfs.drop(columns=[CF_OUTCOME], errors="ignore")[cols].astype(float)
        # DiCE가 상한을 조금 넘거나(예: 납부율 1.1) 자릿수가 어색한 값을 낼 수 있어 허용 범위로 자르고 반올림
        for c in vary:
            lo, hi = ranges[c]
            digits = CF_PRECISION.get(FEATURE_SPEC[c].get("kind"), 2)
            cfs[c] = cfs[c].clip(lo, hi).round(digits)
        cfs = cfs.drop_duplicates()
        for (_, row), p in zip(cfs.iterrows(), wrapper.predict_proba(cfs)[:, 1]):
            result = result_from_p(float(p))
            if result["approval_status"] != "APPROVED":
                continue
            changes = [{"feature": c, "feature_name_kr": label(c),
                        "current_value": float(query.at[0, c]), "suggested_value": float(row[c]),
                        "current_display": format_value(c, float(query.at[0, c])),
                        "suggested_display": format_value(c, float(row[c]))}
                       # 반올림 차이만 있는 항목(예: 0.9902 -> 0.99)은 바뀐 것으로 보지 않음
                       for c in vary
                       if abs(float(row[c]) - round(float(query.at[0, c]),
                                                    CF_PRECISION.get(FEATURE_SPEC[c].get("kind"), 2))) > 1e-6]
            if changes:
                # 변화 폭: 항목마다 (바꾼 양 / 허용 범위 폭)의 합. 작을수록 실천하기 쉬운 방안
                effort = sum(abs(ch["suggested_value"] - ch["current_value"]) / (ranges[ch["feature"]][1] - ranges[ch["feature"]][0])
                             for ch in changes)
                options.append({"changes": changes, "_effort": effort, **result})
    options.sort(key=lambda o: (len(o["changes"]), o.pop("_effort")))   # 적게, 덜 바꾸는 순

    for i, o in enumerate(options, start=1):
        o["rank"] = i

    msg = (f"아래 {len(options)}가지 중 하나를 실행하면 승인 기준을 넘을 것으로 예상됩니다. "
           "(모델 기준 예상이며 실제 승인을 보장하지 않습니다)" if options
           else "바꿀 수 있는 항목만으로는 승인 기준에 닿는 방안을 찾지 못했습니다.")
    return {"original": pred, "counterfactuals": options, "varied_features": vary,
            "fixed_features": fixed, "message": msg}

# 호출시 등록된 엔드포인트만 복사 -> 파일 맨 아래로, 맨 마지막에 호출
app.include_router(router)