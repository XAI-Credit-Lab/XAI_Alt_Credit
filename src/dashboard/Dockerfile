FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/dashboard ./src/dashboard

EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]