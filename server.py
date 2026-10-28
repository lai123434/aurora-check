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
    POST /api/rewrite       按需 AI 降维改写 JSON: {"sentence": "..."}
    POST /api/rewrite-batch 一键批量改写（最多 3 句）JSON: {"sentences": [...]}
    GET  /api/health        健康检查
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import aurora_check as core

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")

app = FastAPI(
    title="Aurora Check API",
    version="3.16.0",
    description="面向学生的 AI 文本自检排雷工具：AI 检测 + 查重 + 引用验证 + 按需降维改写",
)

_VALID_MODES = ("normal", "strict")

# —— LLM 改写并发控制 ——
# 本地 Qwen2.5-0.5b 在 CPU 上会占满多核，若同时来 3 个改写请求，3 句都拖到 40 秒+
# 互相拖慢、前端 30 秒超时变 504。这里用全局信号量(1) 强制「一次只跑一句」，
# 后来的请求排队等待，最多等 60 秒，超时返回 429 提示稍后再试。
_REWRITE_SEMAPHORE = asyncio.Semaphore(1)

# —— SQLite 持久化：记录每次查重结果，前端画「降重曲线」——
# 建在源码同目录，进程退出也不丢；存库失败绝不阻塞主检测流程。
DB_PATH = os.path.join(BASE, "aurora_records.db")


def _init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS check_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            ai_probability REAL,
            label TEXT,
            mode TEXT,
            timestamp TEXT,
            raw_text TEXT
        )
        """
    )
    con.commit()
    con.close()


_init_db()


def _save_record(filename: str, ai_probability: float, label: str,
                 mode: str, raw_text: str) -> None:
    """把一次查重结果写入 SQLite（失败静默，不影响主流程）。"""
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO check_records"
            " (filename, ai_probability, label, mode, timestamp, raw_text)"
            " VALUES (?,?,?,?,?,?)",
            (filename, ai_probability, label, mode,
             datetime.now().isoformat(timespec="seconds"), raw_text),
        )
        con.commit()
        con.close()
    except Exception:
        pass


class CheckPayload(BaseModel):
    text: str = Field(..., min_length=1, description="待检测的正文内容")
    mode: str = Field("normal", description="normal=稳健 / strict=严苛找茬")


class RewritePayload(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=400,
                          description="待改写的高危句（前端按钮按需触发）")


class RewriteBatchPayload(BaseModel):
    sentences: list[str] = Field(..., min_length=1, max_length=3,
                                 description="待改写的句子数组（串行处理，最多 3 句）")


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

    # 记录到 SQLite，供前端「降重曲线」展示（存库失败不影响返回）
    _ai = result.get("ai") or {}
    if _ai:
        _title = payload.text[:20].strip().replace("\n", " ") or "粘贴文本"
        _save_record(
            filename=_title,
            ai_probability=float(_ai.get("ai_probability", 0.0)),
            label=_ai.get("label", ""),
            mode=payload.mode,
            raw_text=payload.text,
        )

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

    # 记录到 SQLite（上传的文件名，供前端「降重曲线」）
    _ai = result.get("ai") or {}
    if _ai:
        _save_record(
            filename=(file.filename or "上传文件"),
            ai_probability=float(_ai.get("ai_probability", 0.0)),
            label=_ai.get("label", ""),
            mode=mode,
            raw_text=text,
        )

    return JSONResponse(result)


@app.post("/api/rewrite")
async def rewrite(payload: RewritePayload):
    """按需触发 AI 降维改写（独立接口，与查重解耦）。

    查重阶段只用 Mock 规则版改写建议（瞬时），真正需要「更自然」的 AI 改写时，
    前端每条高危句旁的「帮我改写」按钮才会调这里。本地 Qwen2.5-0.5b 在 CPU 上
    每句约 5~15 秒，所以丢线程池异步跑，不阻塞事件循环；无模型/异常自动回退 Mock。

    并发控制：全局信号量(1) 保证同一时刻只有一个改写请求在跑，后来的排队等，
    最多等 60 秒；拿不到就返回 429，避免并发把 CPU 打满、互相拖慢成 504。
    """
    sentence = payload.sentence.strip()
    if not sentence:
        raise HTTPException(400, "句子为空")
    if len(sentence.split()) < 3:
        raise HTTPException(400, "句子太短，无需改写")

    try:
        await asyncio.wait_for(_REWRITE_SEMAPHORE.acquire(), timeout=60)
    except TimeoutError:
        raise HTTPException(429, "当前改写请求较多，请稍候再试")

    try:
        rewritten = await asyncio.to_thread(core._llm_rewrite_sentence, sentence)
        return JSONResponse({"original": sentence, "rewritten": rewritten})
    finally:
        _REWRITE_SEMAPHORE.release()


@app.post("/api/rewrite-batch")
async def rewrite_batch(payload: RewriteBatchPayload):
    """一键批量改写（前端「一键改写前 3 句」按钮调用）。

    内部串行处理最多 3 句，整体受同一把信号量保护——既保证 batch 内部不并发，
    也保证 batch 与单句 /api/rewrite 请求互斥，CPU 上永远只有一个改写任务在跑。
    前端对 batch 的心理预期是「等约 40 秒」，一次拿到 3 条，比点 3 次体验好。
    """
    sentences = [s.strip() for s in payload.sentences
                 if s and len(s.strip().split()) >= 3]
    if not sentences:
        raise HTTPException(400, "没有可改写的句子")

    # batch 最多 3 句，给 120 秒等待 + 40 秒执行余量
    try:
        await asyncio.wait_for(_REWRITE_SEMAPHORE.acquire(), timeout=120)
    except TimeoutError:
        raise HTTPException(429, "当前改写请求较多，请稍候再试")

    try:
        results = []
        for s in sentences:
            r = await asyncio.to_thread(core._llm_rewrite_sentence, s)
            results.append({"original": s, "rewritten": r})
        return JSONResponse({"results": results})
    finally:
        _REWRITE_SEMAPHORE.release()


@app.get("/api/history")
async def history():
    """返回最近 10 条查重记录（前端画「降重曲线」）。"""
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT id, filename, ai_probability, label, mode, timestamp"
            " FROM check_records ORDER BY id DESC LIMIT 10"
        ).fetchall()
        con.close()
        # 按 id 升序返回（时间从左到右）
        rows = list(reversed(rows))
        return JSONResponse([
            {
                "id": r[0],
                "filename": r[1],
                "ai_probability": round(float(r[2]), 4),
                "label": r[3],
                "mode": r[4],
                "timestamp": r[5],
            }
            for r in rows
        ])
    except Exception:
        return JSONResponse([])


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.16.0", "modes": list(_VALID_MODES)}


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
