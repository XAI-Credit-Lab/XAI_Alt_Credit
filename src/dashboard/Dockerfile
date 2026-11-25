FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 대시보드 테마 설정 (streamlit은 실행 위치 /app 의 .streamlit/config.toml을 읽음)
COPY .streamlit ./.streamlit

# 대시보드가 쓰는 공통 모듈 (한글 변수 이름·단위 표시)
COPY src/data/feature_spec.py ./src/data/feature_spec.py

COPY src/dashboard ./src/dashboard

EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]