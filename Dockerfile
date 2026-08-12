FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib numpy

COPY model.joblib /app/model.joblib
COPY app.py /app/app.py

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]
