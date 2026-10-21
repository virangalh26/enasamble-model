FROM python:3.12-slim

# libgomp is needed by LightGBM/XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/

ENV PYTHONUNBUFFERED=1
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/health')"
# app/server.py: Python standard-library web server + app/index.html (no Streamlit)
CMD ["python", "app/server.py", "--host", "0.0.0.0", "--port", "8501", "--no-browser"]
