FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY aurora_check.py web.py benchmark.py ./
COPY static ./static
COPY examples ./examples

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
  "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "8000"]
