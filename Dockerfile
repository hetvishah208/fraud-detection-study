# Fraud Cost Explorer - HF Spaces (Docker) deployment.
# HF Spaces expects the app to listen on port 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERSTATS=false

WORKDIR /app

# system deps for xgboost / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Pre-compute artifacts at build time so the Space boots straight into the app.
# Uses the synthetic sample if the real CSV isn't committed (it's large/gitignored).
RUN python src/make_synthetic.py && python src/run_all.py || true

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py"]
