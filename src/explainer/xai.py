# src/explainer/xai.py

# TreeExplainer 또는 KernelExplainer 사용
# (TreeExplainer는 트리 기반 모델 (XGB, LGBM 등) 대상, KernelExplainer는 모든 모델 대상)


# 거절 사유 top5 변수명, 현재 값, shap 기여도, 자연어 설명 반환
# summary plot 이미지, dependence plot 이미지, waterfall plot 이미지

# ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

from typing import Optional
from pathlib import Path

import shap 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 이미지 저장 기준 폴더: 이 파일(xai.py)이 있는 폴더 아래 plots (.../src/explainer/plots)
# 실행 위치(VS Code 작업 폴더, 대시보드 실행 폴더 등)와 상관없이 항상 같은 곳에 저장됨
# 폴더가 없으면 처음 저장할 때 자동으로 만들어짐
SCRIPT_DIR = Path(__file__).resolve().parent   # xai.py가 있는 폴더 (src/explainer)
PLOT_DIR = SCRIPT_DIR / "plots"           

class ModelExplainer:
    # 서버 시작시 클래스 호출하여 설명기(explainer) 객체 한 번만 만들어놓기
    def __init__(self, 
                 model, 
                 background: pd.DataFrame, 
                 predict_fn=None, # 모델의 예측 메소드(ex) model.predict)
                 class_index=1, # 어느 클래스(부도x: 0/ 부도o: 1)에 대한 설명을 볼지 정하는 값
                 global_sample_size: Optional[int] = None, # Summary/Dependence용 표본 크기 (None이면 Tree 2000, Kernel 200)
                 random_state: int = 42,
                 exclude_from_reasons=("is_thin_filer",), # 기여 변수 top에서 "is_thin_filer" 변수는 제외
                 raw_means: Optional[dict] = None, # 전처리전 원본 평균을 따로 받기
                 raw_scales: Optional[dict] = None        # 변수별 원본 단위 표준편차 (scaler.scale_)
    ):
        self.columns = list(background.columns) # 학습에 사용된 데이터 컬럼
        self.class_index = class_index # 어떤 결과(부도 o/x)에 대해 설명할지
        self.random_state = random_state 
        # 자연어 설명의 기준값: 원본 단위 평균이 주어지면 그것을, 없으면 background 평균(표준화 값)
        self.feature_means = pd.Series(raw_means) if raw_means is not None else background.mean()
        self.raw_scales = raw_scales or {}                 # 표준화 되돌리기용 표준편차
        self.exclude_from_reasons = tuple(exclude_from_reasons)  # startswith()는 tuple을 받음


        # KernelExplainer는 확률을 설명해야 의미가 있음
        # model.predict는 분류기에서 0/1라벨 반환. predict_proba가 있으면 그것을 기본으로 사용
        if predict_fn is None: # 모델 객체의 predict메서드에 접근
            predict_fn = getattr(model, "predict_proba", model.predict)
        self.predict_fn = predict_fn


        # 설명기(explainer) 생성, self.kind(사용할 모델 종류) 결정
        # 모델을 TreeExplainer에 넣어서 설명기 생성해보고 오류 발생하면 KernelExplainer 사용
        # 모델을 TreeExplainer에 넣어서 설명기 생성해보고 오류 없으면 모델 종류 "tree"/오류 발생하면 종류 "kernel"
        try:
            self.explainer = shap.TreeExplainer(model)
            self.kind = "tree"
        except Exception as e:
            print(f"TreeExplainer 사용 불가, KernelExplainer 사용:{e}")
            background_data = shap.kmeans(background, 10)
            self.explainer = shap.KernelExplainer(self._predict_np, background_data) # KernelExplainer(모델의 예측 메소드, 배경 데이터)
            self.kind = "kernel"


        # Summary/Dependence용 데이터. 처음 한 번만 계산
        if global_sample_size is None:
            global_sample_size = 2000 if self.kind == "tree" else 200 # 트리 기반 모델이면 2000, 아니면 200
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

    def _shap_for(self, X: pd.DataFrame):
        # 모델 종류에 따라 설명기의 .shap_values() 호출
        if self.kind == "tree":
            sv = self.explainer.shap_values(X)
        else:
            sv = self.explainer.shap_values(X, nsamples = 200)

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

    @staticmethod
    def _finish(fig, save_path = None):
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.is_absolute():             # ← 추가: 상대 경로면
                save_path = PLOT_DIR / save_path        # ← 추가: plots 폴더 아래로
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)    
        return fig      

    # ------------------------------------------------------------------
    # 1. 개별 고객 설명 (API /explain용)
    # ------------------------------------------------------------------
    def explain(self, user_input, top_k: int = 5, raw_values: Optional[dict] = None,
            only_positive: bool = True):  
        x = self._to_frame(user_input)
        sv, base_value = self._shap_for(x)
        values = sv[0] # 입력: 한 행 -> 각 변수의 shap값: 첫 번째 행


        # 거절 사유 후보: 제외 변수(is_thin_filer 등) 빼고, only_positive면 SHAP > 0만
        candidates = [
            (name, v) for name, v in zip(self.columns, values)
            if not name.startswith(self.exclude_from_reasons)
            and (v > 0 or not only_positive)
        ]
        # 부도 확률을 많이 높인 순서로 top_k개
        # (only_positive=False일 때는 영향력 크기 기준이 되도록 절댓값으로 정렬)
        sort_key = (lambda t: t[1]) if only_positive else (lambda t: abs(t[1]))
        top = sorted(candidates, key=sort_key, reverse=True)[:top_k]

        raw_values = raw_values or {}   # None이 들어와도 'in' 검사가 되도록
        reasons = []

        for name, contrib in top:
            scaled = float(x.iloc[0][name])   # 전처리 후 값 (표준화됨)

            # 모델이 실제로 사용한 값을 원본 단위로 복원 (미입력이면 대체값)
            if name in self.raw_scales:
                used = scaled * self.raw_scales[name] + float(self.feature_means[name])
            else:
                used = scaled

            # 평균: 원본 평균이 있으면 그것, 없으면 전역 데이터(표준화) 평균
            if name in self.feature_means.index:
                mean = float(self.feature_means[name])
            else:
                mean = float(self._global_X[name].mean())

            # 키는 있는데 값이 None → 사용자가 입력하지 않은 항목
            if name in raw_values and raw_values[name] is None:
                current = None
                desc = f"{name}: 미입력 (대체값 {used:,.2f}(으)로 계산됨) (기여도: {contrib:+.3f})"
            else:
                raw = raw_values.get(name)
                current = float(raw) if raw is not None else used   # 원본 입력 우선, 없으면 복원값

                if current > mean:
                    direction = "높습니다"
                elif current < mean:
                    direction = "낮습니다"
                else:
                    direction = "같습니다"
                desc = f"{name}이(가) {current:,.2f}(으)로, 평균({mean:,.2f}) 대비 {direction} (기여도: {contrib:+.3f})"

            reasons.append({
                "feature": name,
                "value": current,
                "shap_val": float(contrib),
                "description": desc          # 위에서 만든 문장 사용
            })
        return {
            "base_value": base_value,
            "rejection_reasons": reasons,
            "shap_values": {name: float(v) for name, v in zip(self.columns, values)}
        }

    # ------------------------------------------------------------------
    # 2. 모델 전체 설명용 데이터 (Summary / Dependence 공통)
    # ------------------------------------------------------------------
    def set_global_data(self, X: pd.DataFrame) -> None:
        """Summary/Dependence에 쓸 데이터 지정 (권장: 전처리 후 X_test). 캐시는 초기화됨"""
        self._global_X = self._sample(X)
        self._global_values = None

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
    def plot_summary(self, max_display: int = 15, save_path = None):
        values = self.global_shap()
        plt.figure()
        shap.summary_plot(values, self._global_X, max_display = max_display, show = False)
        return self._finish(plt.gcf(), save_path)

    # ------------------------------------------------------------------
    # 4. SHAP Dependence Plot (기본: 중요도 상위 연속형 변수 2개)
    # ------------------------------------------------------------------
    def plot_dependence(self, features=None, interaction_index="auto", save_dir=None) -> dict:
        """변수별 dependence plot을 {변수명: Figure}로 반환
        features: 변수명 리스트 (None이면 top_features(2))
        interaction_index: 색으로 표시할 상호작용 변수 ("auto"면 shap이 자동 선택, None이면 색 없음)
        save_dir: 지정하면 dependence_<변수명>.png로 저장"""
        if features is None:
            features = self.top_features(2)
        elif isinstance(features, str):
            features = [features]

        unknown = set(features) - set(self.columns)
        if unknown:
            raise ValueError(f"모델에 없는 변수: {unknown}")

        values = self.global_shap()
        figs = {}
        for name in features:
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(name, values, self._global_X,
                                 interaction_index=interaction_index, ax=ax, show=False)
            path = Path(save_dir) / f"dependence_{name}.png" if save_dir else None
            figs[name] = self._finish(fig, path)
        return figs

    # ------------------------------------------------------------------
    # 5. 개별 고객 SHAP Waterfall Plot
    # ------------------------------------------------------------------
    def plot_waterfall(self, user_input, max_display: int = 10,
                       display_values: Optional[dict] = None, save_path=None):
        """user_input: 전처리 후 입력 (dict 또는 1행 DataFrame)
        display_values: 그래프 왼쪽에 '값 = 변수명'으로 보여줄 원본 값 {변수명: 원본값}
            (전처리 후 값은 표준화돼 있어 사용자가 읽기 어려움 → 원본 입력을 넘기는 것을 권장)
            없는 변수는 전처리 후 값으로 표시, None/NaN은 'N/A'로 표시
        x축 단위: TreeExplainer(XGB/LGBM)는 log-odds, KernelExplainer(predict_proba)는 확률"""
        x = self._to_frame(user_input)
        sv, base_value = self._shap_for(x)

        data = x.iloc[0].to_numpy(dtype=float)
        if display_values:
            shown = []
            for name, v in zip(self.columns, data):
                raw = display_values.get(name, v)
                shown.append("N/A" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else raw)
            data = np.array(shown, dtype=object)

        exp = shap.Explanation(values=sv[0], base_values=base_value,
                               data=data, feature_names=self.columns)
        plt.figure()
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        return self._finish(plt.gcf(), save_path)




        
    





    
                


