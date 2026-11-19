FROM python:3.14-slim

# .pyc 파일을 만들지 않도록 \ print()나 로그 출력이 바로바로 화면에 나오도록 \ 모듈을 찾을 때 /app 폴더(루트)도 찾아보도록
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg

WORKDIR /app

# OpenMP(병렬 처리) 기능을 제공하는 라이브러리 libgomp1 설치 (LightGBM, XGBoost가 사용)
# 윈도우 - 해당 기능 제공하는 라이브러리가 LightGBM, XGBoost패키지에 포함되어 있거나 OS에 이미 있어서 따로 설치할 필요 없었음
# 리눅스용 LightGBM, python:slim 이미지에는 이 라이브러리가 빠져 있음 -> 직접 설치
# build-essential: shap에 필요한 g++ 등 컴파일 도구 모음
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# main.py가 import하는 코드 (src.api / src.data / src.explainer / src.fairness)
COPY src/api ./src/api
COPY src/data ./src/data
COPY src/explainer ./src/explainer
COPY src/fairness ./src/fairness

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]