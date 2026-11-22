# src/explainer/xai.py

# TreeExplainer 또는 KernelExplainer 사용
# (TreeExplainer는 트리 기반 모델 (XGB, LGBM 등) 대상, KernelExplainer는 모든 모델 대상)


# 거절 사유 top5 변수명, 현재 값, shap 기여도, 자연어 설명 반환
# summary plot 이미지, dependence plot 이미지, waterfall plot 이미지

# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

from typing import Optional
import io
import base64
import threading

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager
from src.data.feature_spec import build_rejection_reasons, label, format_value, FEATURE_SPEC

def _set_korean_font():
    """그래프의 한글이 깨지지 않도록 설치된 한글 폰트 사용
    (Docker: NanumGothic / 로컬 Windows: Malgun Gothic / Mac: AppleGothic)"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Malgun Gothic", "AppleGothic"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에서 마이너스 기호가 깨지는 문제 방지

_set_korean_font()

class ModelExplainer:
    # 승인 고객의 거절 사유 최소 기여도 (Explainer 종류마다 SHAP 단위가 달라 따로 둠)
    MIN_CONTRIB_KERNEL = 0.01   # KernelExplainer: 부도 확률 단위, 0.01 = 1%p
    MIN_CONTRIB_TREE = 0.05     # TreeExplainer: log-odds 단위, 승인 기준선 근처에서 약 1%p

    # 서버 시작시 클래스 호출하여 설명기(explainer) 객체 한 번만 만들어놓기
    def __init__(self, 
                 model, 
                 background: pd.DataFrame, 
                 predict_fn=None, # 모델의 예측 메소드(ex) model.predict)
                 class_index=1, # 어느 클래스(부도x: 0/ 부도o: 1)에 대한 설명을 볼지 정하는 값
                 global_sample_size: Optional[int] = None, # Summary/Dependence용 표본 크기 (None이면 Tree 2000, Kernel 100)
                 random_state: int = 42,
                 exclude_from_reasons=("is_thin_filer",), # 기여 변수 top에서 "is_thin_filer" 변수는 제외
                 raw_means: Optional[dict] = None, # 원본 입력에 없는 값을 원래 단위로 복원 (StandardScaler 평균)
                 raw_scales: Optional[dict] = None,      # 위 복원용 (StandardScaler 표준편차)
                 raw_medians: Optional[dict] = None,     # 거절 사유 문장의 비교 기준 (Train 중앙값)
                 min_contribution: Optional[float] = None,   # None이면 Explainer 종류에 맞춰 자동 선택
                 feature_labels: Optional[dict] = None   # {변수명: 화면에 보여줄 이름}. 설명 문장에 사용, 없는 변수는 원래 이름
    ):
        self.columns = list(background.columns) # 학습에 사용된 데이터 컬럼
        self.class_index = class_index # 어떤 결과(부도 o/x)에 대해 설명할지
        self.random_state = random_state 
        # KernelExplainer 계산용 잠금: 난수 고정 + 계산을 한 번에 하나씩 (아래 _shap_for 참고)
        self._kernel_lock = threading.Lock()
        # 원본 입력에 없는 값 복원용
        self.feature_means = pd.Series(raw_means) if raw_means is not None else background.mean()
        self.raw_scales = raw_scales or {}
        self.raw_medians = raw_medians or {}
        self.exclude_from_reasons = tuple(exclude_from_reasons)  # startswith()는 tuple을 받음
        self.feature_labels = feature_labels or {}   # None이 들어와도 .get()이 되도록 빈 dict


        # KernelExplainer는 확률을 설명해야 의미가 있음
        # model.predict는 분류기에서 0/1라벨 반환. predict_proba가 있으면 그것을 기본으로 사용
        if predict_fn is None: # 모델 객체의 predict메서드에 접근
            predict_fn = getattr(model, "predict_proba", model.predict)
        self.predict_fn = predict_fn


        # 설명기(explainer) 생성, self.kind(사용할 모델 종류) 결정
        # 모델을 TreeExplainer에 넣어서 설명기 생성해보고 오류 없으면 모델 종류 "tree"/오류 발생하면 종류 "kernel"
        try:
            self.explainer = shap.TreeExplainer(model)
            self.kind = "tree"
        except Exception as e:
            print(f"TreeExplainer 사용 불가, KernelExplainer 사용:{e}")
            background_data = shap.kmeans(background, 10)
            self.explainer = shap.KernelExplainer(self._predict_np, background_data) # KernelExplainer(모델의 예측 메소드, 배경 데이터)
            self.kind = "kernel"

        # 승인 고객 거절 사유의 최소 기여도: 직접 넘긴 값 > Explainer 종류별 기본값
        if min_contribution is not None:
            self.min_contribution = min_contribution
        elif self.kind == "kernel":
            self.min_contribution = self.MIN_CONTRIB_KERNEL
        else:
            self.min_contribution = self.MIN_CONTRIB_TREE


        # Summary/Dependence용 데이터. 처음 한 번만 계산
        if global_sample_size is None:
            global_sample_size = 2000 if self.kind == "tree" else 100 # 트리 기반 모델이면 2000, 아니면 100
        self.global_sample_size = global_sample_size
        self._global_X = self._sample(background)
        self._global_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 내부 도구
    # ------------------------------------------------------------------

    # KernelExplainer- 데이터를 컬럼명 없는 numpy로 넘김. 
    # 데이터에 컬럼명 붙여 DataFrame으로 바꾼 뒤 모델에 넣어 예측하도록
    def _predict_np(self, arr):
        return self.predict_fn(pd.DataFrame(arr, columns = self.columns))

    # Summary/Dependence 플롯을 그릴 데이터 추출
    def _sample(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns]
        if len(X) > self.global_sample_size:
            X = X.sample(self.global_sample_size, random_state = self.random_state)
        return X
    
    # 사용자 입력(dict 또는 1행 DataFrame)을 모델 컬럼 순서의 DataFrame으로 변환
    def _to_frame(self, user_input) -> pd.DataFrame:
        if isinstance(user_input, pd.DataFrame): # DataFrame으로 변환
            x = user_input
        else:
            x = pd.DataFrame([user_input])
        missing = set(self.columns) - set(x.columns)
        if missing: # 모델 입력에 있어야 하는데 사용자 입력에 없는 변수 있으면
            raise ValueError(f"입력에 없는 변수: {missing}")
        return x[self.columns]

    # 변수명 -> 화면에 보여줄 이름 (대응표에 없으면 원래 이름 그대로)
    def _label(self, name: str) -> str:
        return self.feature_labels.get(name, name)

    def _shap_for(self, X: pd.DataFrame):
        # 모델 종류에 따라 설명기의 .shap_values() 호출
        if self.kind == "tree":
            sv = self.explainer.shap_values(X)
        else:
            # 같은 입력에 대해 실행마다 결과가 달라지는 문제
            # KernelExplainer는 무작위 표본으로 SHAP 값을 근사 -> 매번 결과가 달라짐
            # 계산 직전에 난수 시작점(seed)을 고정 -> 같은 입력이면 항상 같은 결과
            with self._kernel_lock:
                np.random.seed(self.random_state)
                # nsamples="auto": 변수 개수에 맞춘 권장 조합 수 (2 * 변수 수 + 2048)
                # 200처럼 너무 적으면 대부분의 변수가 SHAP 0으로 처리되어 거절 사유가 빠짐
                sv = self.explainer.shap_values(X, nsamples = "auto",  l1_reg = False)
        # 모델·shap 버전에 따라 sv 형태 다름(list / 2차원 / 3차원)
        # sv가 [클래스0 배열, 클래스1 배열]로 나오면 관심 클래스(부도) 배열(1)만 선택
        if isinstance(sv, list):
            sv = sv[self.class_index]
        sv = np.asarray(sv)
        # sv가 3차원으로 나오는 경우 결과 통일
        if sv.ndim == 3:
            sv = sv[..., self.class_index]

        # expected value: 숫자 하나(Tree 모델) 또는 클래스별 배열(다른 모델)
        # TreeExplainer(XGB/LGBM): log-odds 단위 / KernelExplainer(predict_proba): 확률 단위
        base = np.ravel(self.explainer.expected_value)
        base = base[self.class_index] if len(base) > 1 else base[0]

        return sv, float(base)
    
    # API 응답(JSON)에 이미지를 담기 위해 matplotlib 그림을 문자열로 변환
    @staticmethod
    def _to_base64(fig, dpi: int = 120) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format = "png", dpi = dpi, bbox_inches = "tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")  

    # ------------------------------------------------------------------
    # 1. 개별 고객 설명 (API /explain용)
    # ------------------------------------------------------------------
    def explain(self, user_input, top_k: int = 5, raw_values: Optional[dict] = None,
                use_min_contribution: bool = False):
        """개별 고객 SHAP + 거절 사유 Top K
        - 후보 선정(제외 변수, 승인 고객 최소 기여도)은 여기서
        - 문장·단위·비교 기준(중앙값) 표시는 feature_spec.build_rejection_reasons()
        raw_values: 사용자 원본 입력 {컬럼명: 값}, 미입력은 None"""
        x = self._to_frame(user_input)
        sv, base_value = self._shap_for(x)
        shap_dict = {name: float(v) for name, v in zip(self.columns, sv[0])}

        # 거절 사유 후보: 제외 변수(is_thin_filer, missingindicator_, past_due_code_flag) 빼고
        # use_min_contribution=True(승인 고객): 기여도가 self.min_contribution보다 큰 변수만
        # use_min_contribution=False(거절 고객): 부도 위험을 조금이라도 높인(SHAP > 0) 변수 모두
        threshold = self.min_contribution if use_min_contribution else 0.0
        candidates = {name: v for name, v in shap_dict.items()
                      if not name.startswith(self.exclude_from_reasons) and v > threshold}

        # 문장에 보여줄 값: 사용자 원본 입력 우선 (미입력은 None → '미입력이어서 중앙값으로 계산' 문장)
        # 원본 입력에 아예 없는 변수는 모델이 실제로 쓴 값을 원본 단위로 복원 (used)
        #   used = 표준화 값 × 표준편차(raw_scales) + 평균(raw_means)  ← StandardScaler 역변환
        values = dict(raw_values or {})
        for name in candidates:
            if name not in values and name in self.raw_scales:
                scaled = float(x.iloc[0][name])
                values[name] = scaled * self.raw_scales[name] + float(self.feature_means[name])

        result = build_rejection_reasons(values, candidates, self.raw_medians, top_k=top_k)
        for r in result["rejection_reasons"]:
            r["feature_label"] = r["feature_name_kr"]   # 이전 필드 이름 호환
        return {
            "base_value": base_value,
            "rejection_reasons": result["rejection_reasons"],
            "total_rejection_factors": result["total_rejection_factors"],
            "shap_values": shap_dict,   # 전체 변수 (대시보드 막대그래프, waterfall 재사용)
        }

    # ------------------------------------------------------------------
    # 2. 모델 전체 설명용 데이터 (Summary / Dependence 공통)
    # ------------------------------------------------------------------
    def global_shap(self) -> np.ndarray:
        """전역 데이터의 SHAP 값 (처음 한 번만 계산 후 캐시)"""
        if self._global_values is None:
            self._global_values, _ = self._shap_for(self._global_X)
        return self._global_values

    def feature_importance(self) -> pd.Series:
        """평균 |SHAP| 기준 변수 중요도 (내림차순)"""
        imp = np.abs(self.global_shap()).mean(axis=0)
        return pd.Series(imp, index=self.columns).sort_values(ascending=False)

    def top_features(self, k: int = 2, continuous_only: bool = True) -> list:
        """중요도 상위 k개 변수. continuous_only=True면 0/1 변수(결측 지시변수, is_thin_filer_1)는 제외
        (이진 변수의 dependence plot은 점이 두 줄로만 찍혀 정보가 적음)"""
        names = list(self.feature_importance().index)
        if continuous_only:
            names = [c for c in names if self._global_X[c].nunique() > 2]
        return names[:k]


    # ------------------------------------------------------------------
    # 3. SHAP Summary Plot (상위 max_display개 변수)
    # ------------------------------------------------------------------
    def plot_summary(self, max_display: int = 15) -> str:
        values = self.global_shap()
        plt.figure()
        shap.summary_plot(values, self._global_X, max_display = max_display, show = False)
        return self._to_base64(plt.gcf())

    # ------------------------------------------------------------------
    # 4. SHAP Dependence Plot (최소 2개의 변수에 대해 생성)
    # ------------------------------------------------------------------
    def plot_dependence(self, features=None, interaction_index="auto") -> dict:
        """변수별 dependence plot을 {변수명: base64 PNG 문자열}로 반환
        features: 변수명 리스트 (None이면 top_features(2))
        interaction_index: 색으로 표시할 상호작용 변수 ("auto"면 shap이 자동 선택, None이면 색 없음)"""
        if features is None:
            features = self.top_features(2)
        elif isinstance(features, str):
            features = [features]

        unknown = set(features) - set(self.columns)
        if unknown:
            raise ValueError(f"모델에 없는 변수: {unknown}")

        values = self.global_shap()
        images = {}
        for name in features:
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(name, values, self._global_X,
                                 interaction_index=interaction_index, ax=ax, show=False)
            images[name] = self._to_base64(fig)
        return images

    # ------------------------------------------------------------------
    # 5. 개별 고객 SHAP Waterfall Plot
    # ------------------------------------------------------------------
    
    def plot_waterfall(self, user_input, max_display: int = 10,
                       display_values: Optional[dict] = None,
                       shap_values: Optional[dict] = None,
                       base_value: Optional[float] = None) -> str:
        """개별 고객 SHAP Waterfall plot -> base64 PNG 문자열
        user_input: 전처리 후 입력 (dict 또는 1행 DataFrame)
        display_values: 그래프 왼쪽에 '값 = 변수명'으로 보여줄 원본 값 {변수명: 원본값}
            (전처리 후 값은 표준화돼 있어 사용자가 읽기 어려움 → 원본 입력을 넘기는 것을 권장)
            없는 변수는 전처리 후 값으로 표시, None/NaN은 'N/A'로 표시
        shap_values, base_value: explain()이 이미 계산한 값 ({변수명: SHAP 값}, 기준값)
            둘 다 주면 다시 계산하지 않고 재사용 -> 거절 사유와 그림이 같은 SHAP 값을 사용
            주지 않으면 여기서 새로 계산
        x축 단위: TreeExplainer(XGB/LGBM)는 log-odds, KernelExplainer(predict_proba)는 확률"""
        x = self._to_frame(user_input)

        if shap_values is not None and base_value is not None:
            # 이미 계산된 값 재사용 (dict -> 모델 컬럼 순서의 배열)
            values = np.array([shap_values[name] for name in self.columns], dtype=float)
        else:
            # 새로 계산
            sv, base_value = self._shap_for(x)
            values = sv[0]

        data = x.iloc[0].to_numpy(dtype=float)
        if display_values:
            shown = []
            for name, v in zip(self.columns, data):
                raw = display_values.get(name, v)
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    shown.append("N/A")
                elif name in FEATURE_SPEC:
                    shown.append(format_value(name, raw))   # 예: 0.99 -> '99%', 4800 -> '4,800달러'
                else:
                    shown.append(raw)
            data = np.array(shown, dtype=object)

        exp = shap.Explanation(values=values, base_values=base_value, data=data,
                               feature_names=[label(c) for c in self.columns])   # 한글 이름 (feature_spec)
        plt.figure()
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        return self._to_base64(plt.gcf())
