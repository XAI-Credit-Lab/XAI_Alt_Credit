"""
src/data/preprocessor.py
XAI 기반 대안신용평가 시스템 - 전처리 및 데이터 분할 모듈

체크리스트 대응
  1) 결측치 처리      : 수치형 -> 중앙값(median) 대체 (+ 결측 여부 지시변수 추가)
                        범주형 -> 최빈값 대체, 타깃이 결측인 행은 삭제
  2) 수치형 스케일링  : StandardScaler (평균 0, 표준편차 1)
  3) 범주형 인코딩    : One-Hot Encoding
  4) 분할 비율        : Train/Validation/Test = 70:15:15
  5) 층화 분할        : stratify=y 로 클래스(부도) 비율 유지

설계 원칙
  - 데이터 누수 방지
      * 먼저 분할한 뒤, 이상치 상한/결측 대체값/스케일러/인코더는 Train에만 fit 하고
        Validation/Test에는 transform만 적용한다.
      * 시뮬레이터가 만든 승인 라벨(approv/승인 포함 컬럼)은 자동으로 제외한다.
  - 보호 속성(민감 속성) 처리
      * gender, age_group 은 모델 피처에서 빼고, 공정성 평가(AIF360/Fairlearn)용으로
        분할 인덱스에 맞춰 따로 돌려준다.
      * age(원본 나이), persona(주부=여성, 사회초년생=20~25세 기준)는 보호 속성의
        대리변수(proxy)이므로 피처에서 제외한다.
  - 씬파일러 신호 보존
      * 중앙값으로 채우기 전에 '원래 결측이었는지'를 0/1 컬럼으로 남긴다
        (SimpleImputer add_indicator=True).
  - SHAP에서 쓸 수 있도록 변환 후 피처 이름을 보존한다.

요구 사항: scikit-learn >= 1.2 (sparse_output, set_output 사용)
"""
# 코드 수정은 낮에 할 예정
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# 고정 설정
# ----------------------------------------------------------------------
TARGET = "SeriousDlqin2yrs"                    # 타깃: 2년 내 90일 이상 연체 여부

SENSITIVE_COLS = ["gender", "age_group"]       # 보호 속성: 피처 X, 공정성 평가용으로 반환
PROXY_COLS = ["age", "persona"]                # 보호 속성의 대리변수: 피처 X
# 피처로 쓸 범주형 -> One-Hot Encoding 적용 (문자열/bool 컬럼은 자동 추가)
# is_thin_filer: 시뮬레이터의 씬파일러 판정 결과 (True/False 또는 0/1)
#   ※ 시뮬레이터에서 저장한 컬럼 이름과 같아야 함
CATEGORICAL_COLS = ["is_thin_filer"]
DROP_COLS = ["Unnamed: 0"]                     # ID 등
LEAK_KEYWORDS = ["approv", "승인"]             # 이 단어가 들어간 컬럼은 정답 누수로 보고 제외

# GMSC 연체 횟수 컬럼의 96, 98은 실제 횟수가 아니라 '기록 이상' 코드 -> 결측 처리
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
PAST_DUE_INVALID = [96, 98]
CLIP_QUANTILE = 0.99                           # 수치형 상위 1% 초과 값은 상한으로 자름

# 경로는 실행 위치(VS Code 작업 폴더 등)와 상관없이 이 파일 위치 기준으로 계산
SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/data
PROJECT_ROOT = SCRIPT_DIR.parents[1]                  # .../XAI_Simulator-1

DEFAULT_INPUT = SCRIPT_DIR / "결과.csv"                  # 시뮬레이터 결과 CSV
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"   # 분할 CSV 저장 폴더
DEFAULT_SAVE = PROJECT_ROOT / "artifacts" / "preprocessor.joblib"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 이상치 상한 자르기 (Train 분위수로 fit -> 누수 없음)
# ----------------------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """각 수치형 컬럼을 Train 기준 q 분위수로 상한 처리. 결측은 그대로 둔다."""

    def __init__(self, q: float = CLIP_QUANTILE):
        self.q = q

    def fit(self, X, y=None):
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        self.n_features_in_ = self.upper_.shape[0]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        out = X.copy() if hasattr(X, "clip") else np.asarray(X, dtype=float).copy()
        if hasattr(out, "clip"):
            return out.clip(upper=pd.Series(self.upper_, index=out.columns), axis=1)
        return np.minimum(out, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", None)


@dataclass
class SplitData:
    """분할 + 전처리가 끝난 결과 묶음"""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    # 공정성 평가용 보호 속성 (인코딩 전 원본 값)
    sensitive_train: pd.DataFrame
    sensitive_val: pd.DataFrame
    sensitive_test: pd.DataFrame


class CreditPreprocessor:
    """
    사용 예
    -------
    >>> pre = CreditPreprocessor()
    >>> data = pre.load_and_process(DEFAULT_INPUT)
    >>> model.fit(data.X_train, data.y_train)
    >>> pre.save_splits(data, DEFAULT_OUT_DIR)
    """

    def __init__(self, drop_cols: list[str] | None = None):
        self.drop_cols = DROP_COLS + (drop_cols or [])
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.dropped_cols_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    # ------------------------------------------------------------------
    # 0. 원본 정리: 누수/ID 컬럼 제거, GMSC 코드값 결측화
    # ------------------------------------------------------------------
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        leak = [c for c in df.columns
                if c != TARGET and any(k in c.lower() for k in LEAK_KEYWORDS)]
        if leak:
            print(f"[preprocessor] 정답 누수 의심 컬럼 자동 제외: {leak}")
        drop = [c for c in self.drop_cols if c in df.columns] + leak
        self.dropped_cols_ = drop
        df = df.drop(columns=drop)

        for c in PAST_DUE_COLS:
            if c in df.columns:
                df[c] = df[c].where(~df[c].isin(PAST_DUE_INVALID), np.nan)
        return df

    # ------------------------------------------------------------------
    # 1. 컬럼 구분 (수치형 / 범주형)
    # ------------------------------------------------------------------
    def _resolve_columns(self, X: pd.DataFrame) -> None:
        cat = [c for c in CATEGORICAL_COLS if c in X.columns]
        cat += [c for c in X.select_dtypes(include=["object", "category", "bool"]).columns
                if c not in cat]
        self.categorical_cols_ = cat
        self.numeric_cols_ = [c for c in X.columns
                              if c not in cat and pd.api.types.is_numeric_dtype(X[c])]

    @staticmethod
    def _to_category_str(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """범주형 값을 문자열로 통일 (bool은 SimpleImputer가 지원하지 않음).
        True/1 -> "1", False/0 -> "0" 으로 맞춰 저장 형식이 달라도 같은 컬럼이 나오게 한다."""
        X = X.copy()
        for c in cols:
            if c not in X.columns:
                continue
            s = X[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                s = s.astype("Int64")             # True->1, 2.0->2, NaN 유지
            X[c] = s.astype("string").astype(object).where(s.notna(), np.nan)
        return X

    # ------------------------------------------------------------------
    # 2. 전처리 파이프라인
    #    수치형: 상위 1% 상한 -> 중앙값 대체(+결측 지시변수) -> StandardScaler
    #    범주형: 최빈값 대체 -> One-Hot Encoding
    # ------------------------------------------------------------------
    def _build_transformer(self) -> ColumnTransformer:
        numeric_pipe = Pipeline([
            ("clip", QuantileClipper()),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": 학습 때 없던 범주가 들어와도 에러 없이 모두 0 처리
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = [("num", numeric_pipe, self.numeric_cols_)]
        if self.categorical_cols_:
            transformers.append(("cat", categorical_pipe, self.categorical_cols_))

        ct = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,  # SHAP 그래프에서 원래 이름이 보이도록
        )
        ct.set_output(transform="pandas")
        return ct

    # ------------------------------------------------------------------
    # 3. 70:15:15 층화 분할 (stratify=y)
    # ------------------------------------------------------------------
    @staticmethod
    def split(X: pd.DataFrame, y: pd.Series):
        # 1단계: Train 70% vs 나머지 30%
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=VAL_RATIO + TEST_RATIO,
            stratify=y, random_state=RANDOM_STATE,
        )
        # 2단계: 나머지 30%를 Validation 15% / Test 15%로 반반
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=y_tmp, random_state=RANDOM_STATE,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # 4. 전체 흐름: 정리 -> 분할 -> Train에 fit -> 전체 transform
    # ------------------------------------------------------------------
    def process(self, df: pd.DataFrame) -> SplitData:
        if TARGET not in df.columns:
            raise KeyError(f"타깃 컬럼 '{TARGET}'이 데이터에 없습니다.")

        df = self._clean(df)
        df = df.dropna(subset=[TARGET])        # 타깃 결측 행은 삭제
        y = df[TARGET].astype(int)
        X = df.drop(columns=[TARGET])

        sensitive = X[[c for c in SENSITIVE_COLS if c in X.columns]].copy()
        X = X.drop(columns=[c for c in SENSITIVE_COLS + PROXY_COLS if c in X.columns])

        self._resolve_columns(X)
        X = self._to_category_str(X, self.categorical_cols_)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)

        # Train에만 fit -> Validation/Test 정보가 새어 들어가지 않음
        self.transformer_ = self._build_transformer()
        X_train_t = self.transformer_.fit_transform(X_train)
        X_val_t = self.transformer_.transform(X_val)
        X_test_t = self.transformer_.transform(X_test)

        data = SplitData(
            X_train_t, X_val_t, X_test_t, y_train, y_val, y_test,
            sensitive.loc[X_train.index],
            sensitive.loc[X_val.index],
            sensitive.loc[X_test.index],
        )
        self.report(data)
        return data

    def load_and_process(self, path: str | Path) -> SplitData:
        return self.process(pd.read_csv(path))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습된 전처리를 새 데이터(API /predict 입력 등)에 그대로 적용"""
        if self.transformer_ is None:
            raise RuntimeError("먼저 process()로 학습해야 합니다.")
        df = df.copy()
        for c in PAST_DUE_COLS:
            if c in df.columns:
                df[c] = df[c].where(~df[c].isin(PAST_DUE_INVALID), np.nan)
        df = self._to_category_str(df, self.categorical_cols_)
        return self.transformer_.transform(df)

    @property
    def feature_names(self) -> list[str]:
        """변환 후 피처 이름 (SHAP 시각화용)"""
        if self.transformer_ is None:
            raise RuntimeError("먼저 process()로 학습해야 합니다.")
        return list(self.transformer_.get_feature_names_out())

    # ------------------------------------------------------------------
    # 5. 검증 리포트 (비율, 클래스 비율, 남은 결측 확인)
    # ------------------------------------------------------------------
    @staticmethod
    def report(data: SplitData) -> None:
        sets = {"train": (data.X_train, data.y_train),
                "val": (data.X_val, data.y_val),
                "test": (data.X_test, data.y_test)}
        total = sum(len(y) for _, y in sets.values())
        print("[preprocessor] 분할 결과 (70:15:15, stratify=y)")
        for name, (X, y) in sets.items():
            print(f"  {name:5s}: {len(y):>7,}행 ({len(y)/total:6.2%})  "
                  f"부도비율 {y.mean():.4f}  남은 결측 {int(X.isna().sum().sum())}")
        print(f"  피처 수: {data.X_train.shape[1]}")

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------
    @staticmethod
    def save_splits(data: SplitData, out_dir: str | Path) -> None:
        """팀원 모두 같은 분할로 작업할 수 있도록 CSV로 저장"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            getattr(data, f"X_{name}").to_csv(out / f"X_{name}.csv")
            getattr(data, f"y_{name}").to_csv(out / f"y_{name}.csv")
            getattr(data, f"sensitive_{name}").to_csv(out / f"sensitive_{name}.csv")

    def save(self, path: str | Path) -> None:
        """FastAPI /predict, MLflow에서 같은 전처리를 재사용하기 위해 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CreditPreprocessor":
        return joblib.load(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="신용평가 데이터 전처리 및 70:15:15 분할")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_INPUT),
                        help=f"시뮬레이터 결과 CSV 경로 (기본값: {DEFAULT_INPUT})")
    parser.add_argument("--drop", nargs="*", default=[],
                        help="추가로 뺄 컬럼 (승인 라벨 이름에 approv/승인이 없을 때 지정)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="분할 CSV 저장 폴더")
    parser.add_argument("--save", default=str(DEFAULT_SAVE), help="전처리기 저장 경로")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(f"[preprocessor] 입력 파일을 찾을 수 없습니다: {Path(args.csv).resolve()}\n"
                         f"  -> 파일 위쪽 DEFAULT_INPUT 경로를 확인하세요.")

    pre = CreditPreprocessor(drop_cols=args.drop)
    data = pre.load_and_process(args.csv)
    pre.save_splits(data, args.out)
    pre.save(args.save)
    print(f"[preprocessor] 저장 완료: {args.out}/, {args.save}")