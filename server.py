# -*- coding: utf-8 -*-
"""
Aurora Check API 服务（v3）

把纯本地脚本 aurora_check.py 封装成 Web API，别人传一段文本即可拿回 JSON 报告，
也能被微信小程序 / 网页后端 / 其他 AI 通过 MCP 直接调用。

启动：
    python server.py                  # 默认 http://127.0.0.1:8000
    python server.py --port 8080
    uvicorn server:app --reload       # 开发模式（热重载）

接口：
    GET  /               可视化首页（网页输入框粘贴文字即可测）
    POST /api/check      提交文本 JSON: {"text": "...", "mode": "strict"}
    POST /api/check-file 上传文件（.txt/.md/.docx），mode 走 query 参数
    GET  /api/health     健康检查
"""
from __future__ import annotations

import argparse
import asyncio
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import aurora_check as core

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")

app = FastAPI(
    title="Aurora Check API",
    version="3.0.0",
    description="面向学生的 AI 文本自检排雷工具：AI 检测 + 查重 + 引用验证",
)

_VALID_MODES = ("normal", "strict")


class CheckPayload(BaseModel):
    text: str = Field(..., min_length=1, description="待检测的正文内容")
    mode: str = Field("normal", description="normal=稳健 / strict=严苛找茬")


def _run_check_blocking(text: str, mode: str) -> dict:
    """同步跑完整检测流程，返回可直接 JSON 序列化的 dict。

    单独拆出来是为了方便丢进 asyncio.to_thread 的线程池：
    大模型推理跑在 CPU 上要十几秒，绝不能占用事件循环。
    """
    report = core.run_check(text=text, mode=mode)
    return report.to_dict()


@app.get("/", response_class=JSONResponse)
async def index():
    """可视化首页：网页输入框粘贴文字即可测试。"""
    path = os.path.join(STATIC, "index.html")
    if not os.path.exists(path):
        raise HTTPException(500, "界面文件缺失：static/index.html")
    return FileResponse(path, media_type="text/html")


@app.post("/api/check")
async def check(payload: CheckPayload):
    """核心接口：提交一段文本，返回完整 JSON 报告。

    请求体示例：
        {"text": "文章内容...", "mode": "strict"}
    """
    if not payload.text.strip():
        raise HTTPException(400, "文本为空")
    if payload.mode not in _VALID_MODES:
        raise HTTPException(400, f"mode 必须是 {' / '.join(_VALID_MODES)}")

    # 大模型跑在 CPU 上很慢，丢进线程池，避免阻塞事件循环。
    # 注意：run_check 内部用 ContextVar + lru_cache，跨线程调用各自隔离、线程安全。
    result = await asyncio.to_thread(_run_check_blocking, payload.text, payload.mode)
    return JSONResponse(result)


@app.post("/api/check-file")
async def check_file(file: UploadFile = File(...), mode: str = "normal"):
    """上传文件检测（.txt / .md / .docx）。mode 走 query 参数：?mode=strict"""
    if mode not in _VALID_MODES:
        raise HTTPException(400, f"mode 必须是 {' / '.join(_VALID_MODES)}")

    raw = await file.read()
    name = (file.filename or "upload").lower()

    tmp_path = ""
    try:
        if name.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            text = core.extract_text_from_file(tmp_path)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"无法解析文件：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(400, "文件内容为空")

    result = await asyncio.to_thread(_run_check_blocking, text, mode)
    return JSONResponse(result)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "modes": list(_VALID_MODES)}


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="Aurora Check Web API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    print(f"Aurora Check 已启动 → http://{a.host}:{a.port}")
    print("接口：POST /api/check  首页：GET /")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
