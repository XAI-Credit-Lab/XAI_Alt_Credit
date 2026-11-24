# XAI 기반 대안신용평가 시스템

> ※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

## 1. 프로젝트 개요

2024년 기준 국내 금융이력부족자(씬파일러)는 약 1,100만 명으로, 금융 이력이 부족해 기존 신용평가(CB) 모델로는 실제 상환 능력을 인정받기 어렵습니다. 이 프로젝트는 Kaggle **Give Me Some Credit** 데이터에 통신비·공과금 납부율, 소비 일관성 등 **대안 데이터 시뮬레이터**로 만든 변수를 결합해 신용위험을 예측하고, 다음 세 가지를 함께 제공하는 대안신용평가 시스템입니다.

- **설명가능성(XAI)**: SHAP으로 거절 사유 Top 5를 변수명·현재값·기여도와 함께 문장으로 제시하고, DiCE 반사실적 설명으로 "무엇을 바꾸면 승인되는지"를 보여 줍니다.
- **공정성 검증**: 성별·연령대별 Disparate Impact와 Equalized Odds를 측정하고, Threshold Adjustment로 편향을 완화합니다.
- **통계 검증**: One-way ANOVA와 Tukey HSD로 피처 그룹·알고리즘 간 성능 차이의 유의성을 검정합니다.

결과물은 **FastAPI 심사 API**, **Streamlit 심사역 대시보드**, **MLflow 실험 추적**으로 구성되며 Docker Compose로 함께 실행됩니다. 자세한 실험 내용은 [`docs/experiment_report.pdf`](docs/experiment_report.pdf)에 있습니다.

## 2. 팀 구성원 역할

| 이름 | 역할 | 담당 업무 |
|---|---|---|
| jangyc3972-bot (팀장) | 데이터·모델·통계 | 데이터 파이프라인 구축(로더·대안 데이터 시뮬레이터·전처리), 모델 학습·MLflow, 통계 검증(ANOVA), 조건 변경에 따른 예측 결과 비교 로직 |
| SeonuChoi (팀원) | API·XAI | FastAPI 서버 구축 및 모델 연동, 설명가능성(XAI) 모듈, 반사실적 설명 API 엔드포인트, Docker Compose |
| ahncs0329-star (팀원) | 공정성·대시보드 | 공정성 검증(DI, Equalized Odds, Bias Mitigation), Streamlit 대시보드 개발, 반사실적 설명 시뮬레이터 화면 구성 |

## 3. 실행 방법

### 3.1 데이터 준비

원본 데이터(`src/data/cs-training.csv`)는 재현과 테스트를 위해 저장소에 포함되어 있습니다. 출처는 Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)이며, 직접 받으려면 대회 페이지에서 규칙에 동의한 뒤 둘 중 하나를 사용합니다.

- **웹**: Data 탭에서 `cs-training.csv`를 내려받아 `src/data/`에 넣습니다.
- **Kaggle API** (`pip install kaggle`, 인증 키 `kaggle.json` 필요):

```bash
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip cs-training.csv -d src/data/
```

### 3.2 Docker Compose로 전체 실행

학습된 모델(`models/`), 전처리기(`artifacts/`), 분할 데이터(`data/processed/`), 리포트(`reports/`)가 저장소에 포함되어 있어 바로 실행할 수 있습니다.

```bash
docker-compose up --build
```

| 서비스 | 주소 |
|---|---|
| API 서버 | http://localhost:8000 (API 문서: http://localhost:8000/docs) |
| 대시보드 | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

MLflow UI에는 저장소에 포함된 실험 기록(`mlflow_data/`)이 그대로 표시됩니다. 실험 `credit-scoring`의 run 9개와 Model Registry `credit_scoring_model`(Production: LightGBM v1.3)을 확인할 수 있습니다.

### 3.3 로컬 실행 (데이터 생성부터 재현)

Python 3.14 기준으로 개발했습니다(Docker 이미지와 동일).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  /  macOS·Linux: source .venv/bin/activate
pip install -r src/api/requirements.txt -r src/dashboard/requirements.txt mlflow==3.17.0 pytest
```

```bash
# 0) MLflow 서버 먼저 실행 (학습 기록이 mlflow_data/에 저장되도록)
#    서버가 꺼져 있으면 프로젝트 루트의 mlflow.db에 기록되어 MLflow UI에 나타나지 않습니다.
docker-compose up -d mlflow
# 1) 대안 데이터 생성 → data/processed/결과_시나리오.csv, 결과_통제.csv
python -m src.data.simulator
# 2) 전처리 + 모델 학습 + MLflow 기록 (SMOTE 비교 실험은 --with-smote)
python -m src.models.model
# 3) 공정성 검증 → reports/fairness_report.json, fairness_comparison.csv
python -m src.fairness.fairness
# 4) ANOVA 통계 검증 → reports/anova_results.json
python -m src.statistics.anova
# 5) API와 대시보드 (각각 다른 터미널에서)
uvicorn src.api.main:app --port 8000
streamlit run src/dashboard/app.py
# 6) 테스트 (164개)
pytest tests/
```

> `--with-smote`로 실행한 결과는 데이터와 관계없이 버전 `v1.1`로 저장되어 기존 `models/*_v1.1`, `model_meta_v1.1.json` 파일을 덮어씁니다.

## 4. 디렉토리 구조

```
XAI_Simulator/
├── docker-compose.yml          # MLflow(:5000) + API(:8000) + 대시보드(:8501) 통합 실행
├── README.md
├── compare_meta.py             # v1.2·v1.3 모델 메타 지표 비교 (보조 스크립트)
├── .streamlit/config.toml      # 대시보드 테마 (로컬 실행 시 적용)
├── docs/
│   └── experiment_report.pdf   # 실험 리포트
├── src/
│   ├── data/
│   │   ├── loader.py           # GMSC 로드 + 컬럼명·dtype·결측 비율 로그 출력
│   │   ├── simulator.py        # 대안 변수 5개·보호 속성·씬파일러 생성 (thin_filer_ratio, bias_ratio)
│   │   ├── preprocessor.py     # 결측 대체·스케일링·인코딩, 70:15:15 층화 분할
│   │   ├── feature_spec.py     # 변수 한글명·단위·입력 범위, 거절 사유 문장 생성
│   │   ├── audit_data.py       # 데이터 점검(단위·결측·코드값) 리포트
│   │   ├── check_debtratio.py  # DebtRatio 이상값 원인 점검 (reports/check_debtratio.txt)
│   │   └── cs-training.csv     # GMSC 원본 데이터 (Kaggle)
│   ├── models/
│   │   ├── model.py            # 3개 알고리즘 × 불균형 처리, 5-Fold CV, MLflow 기록·Registry 등록
│   │   └── check_shap.py       # SHAP 안정성·가법성 점검
│   ├── explainer/xai.py        # SHAP: Summary·Dependence·Waterfall, 거절 사유 Top 5
│   ├── fairness/fairness.py    # DI, Equalized Odds, WARNING 로그, Threshold Adjustment
│   ├── statistics/anova.py     # One-way ANOVA, Tukey HSD
│   ├── api/                    # FastAPI (main.py, schema.py, Dockerfile, requirements.txt)
│   └── dashboard/              # Streamlit 5개 탭 (app.py, fairness_tab.py, theme.py, units.py, Dockerfile, requirements.txt)
├── models/                     # 학습된 모델(*_v1.x.pkl)과 메타 정보(model_meta_v1.x.json)
├── artifacts/                  # 학습된 전처리기 (preprocessor_v1.x.joblib)
├── data/processed/             # 시뮬레이터 결과, 분할 데이터, 상관행렬
│                               #   X_/y_/raw_*.csv  : 모델 입력·정답·원본 값
│                               #   sensitive_*.csv  : 성별·연령대·씬파일러 여부 (모델 피처 아님, 공정성·씬파일러 평가 전용)
├── reports/                    # 공정성 리포트·완화 전후 비교표, ANOVA 결과, 데이터 점검 결과
├── mlflow_data/                # MLflow 실험 기록 공유용 (mlflow.db + 모델·전처리기 아티팩트, v1.1·v1.3)
└── tests/                      # pytest 단위 테스트 164개
```

## 5. 환경 변수

코드는 `.env` 파일을 자동으로 읽지 않습니다. 터미널에서 직접 설정하거나 `docker-compose.yml`의 `environment:`로 전달합니다. 모두 기본값이 있어 설정하지 않아도 실행됩니다.

| 변수 | 사용 위치 | 기본값 | 설명 |
|---|---|---|---|
| `API_URL` | 대시보드 | `http://localhost:8000` | 대시보드가 호출할 API 주소 (Docker: `http://api:8000`) |
| `MLFLOW_TRACKING_URI` | `src/models/model.py` | `http://localhost:5000` | MLflow 서버 주소 |
| `DATA_DIR` | `src/data/loader.py` | `src/data` | GMSC 원본 데이터 폴더 |
| `GMSC_FILENAME` | `src/data/loader.py` | `cs-training.csv` | GMSC 파일 이름 |
| `MODEL_VERSION` | `src/fairness/fairness.py` | 가장 높은 버전 | 공정성 계산에 쓸 모델 메타 버전 (예: `1.3`) |
| `ANOVA_ALPHA` | `src/statistics/anova.py` | `0.05` | 유의수준 |
| `ANOVA_N_SPLITS` | 〃 | `5` | K-Fold의 K |
| `ANOVA_N_REPEATS` | 〃 | `3` | K-Fold 반복 횟수 |
| `RANDOM_STATE` | 〃 | `42` | 난수 시드 |
| `ANOVA_RESULT_PATH` | 〃 | `reports/anova_results.json` | ANOVA 결과 저장 위치 |
| `ANOVA_INPUT_CSV` | 〃 | `data/processed/결과_시나리오.csv` | ANOVA 입력 데이터 |

포트는 `docker-compose.yml`의 `ports:`에서 바꿀 수 있습니다. 공정성 판정 기준(DI 0.8~1.25, Equalized Odds 차이 0.10)은 과제 요구사항으로 정해진 값이라 `src/fairness/fairness.py`의 상수(`DI_LOWER`, `DI_UPPER`, `EO_THRESHOLD`)로 관리합니다. 승인 기준 점수(threshold)는 학습 시 정해져 `models/model_meta_v1.x.json`에 저장됩니다.
