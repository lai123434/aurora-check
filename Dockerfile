# Aurora Check · 一键部署镜像
# 同学无需装 Python、无需下载 2GB 模型：docker compose up 后
# 浏览器打开 http://localhost:8000 即可使用。
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# 系统依赖（curl 供健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖：torch 用 CPU 版（避免拉 CUDA 巨型包），
# transformers 供 ML 检测器 + 伪困惑度，web 依赖供 FastAPI/uvicorn。
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers>=4.30.0" \
    && pip install --no-cache-dir -r requirements-web.txt

# 源码 + 静态资源 + 校准语料（不含 models/ 大权重，模型从 HF 缓存挂载加载）
COPY aurora_check.py dictionaries.py forensic.py persona.py check.py teacher.py normalize_meta.py server.py ./
COPY static ./static
COPY corpus ./corpus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
