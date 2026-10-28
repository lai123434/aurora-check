# -*- coding: utf-8 -*-
"""
Aurora Check API service (v3).

Wraps the purely local ``aurora_check.py`` script as a Web API: hand it a piece
of text and get a JSON report back. It can also be called directly by WeChat mini
programs, web backends, or other AI agents via MCP.

Start:
    python server.py                  # default http://127.0.0.1:8000
    python server.py --port 8080
    uvicorn server:app --reload       # dev mode (hot reload)

Endpoints:
    GET  /                   Visual home page (paste text into the box)
    POST /api/check          Submit text JSON: {"text": "...", "mode": "strict"}
    POST /api/check-file     Upload a file (.txt/.md/.docx); mode as a query param
    POST /api/rewrite        On-demand AI humanizing rewrite: {"sentence": "..."}
    POST /api/rewrite-batch  Batch rewrite (max 3 sentences): {"sentences": [...]}
    GET  /api/history        Recent check records (for the de-AI curve)
    GET  /api/health         Health check
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

# —— LLM rewrite concurrency control ——
# The local Qwen2.5-0.5b saturates multiple cores on CPU: three simultaneous
# rewrite requests would drag each other past 40 s and trip the front-end's
# 30 s timeout into a 504. A global Semaphore(1) enforces "one sentence at a
# time"; later requests queue for at most 60 s, then get a 429 asking to retry.
_REWRITE_SEMAPHORE = asyncio.Semaphore(1)

# —— SQLite persistence: record every check for the front-end "de-AI curve" ——
# The DB lives next to the source, so it survives process exits; storage failures
# never block the main detection flow.
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
    """Persist one check result to SQLite (fails silently, never blocks detection)."""
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
    """Run the full detection pipeline synchronously, returning a JSON-ready dict.

    Split out as a standalone function so it can be handed to
    ``asyncio.to_thread``'s thread pool: model inference takes over ten seconds
    on CPU and must never occupy the event loop.

    Args:
        text: The text to analyse.
        mode: ``"normal"`` or ``"strict"``.

    Returns:
        dict: A JSON-serializable report produced by ``CheckReport.to_dict()``.
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
    """Core endpoint: submit a piece of text, get the full JSON report back.

    Request body example:
        {"text": "Article content...", "mode": "strict"}
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
    """Detect an uploaded file (.txt / .md / .docx). mode comes from the query string: ?mode=strict"""
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
    """On-demand AI humanizing rewrite (standalone endpoint, decoupled from checks).

    Detection itself only attaches instant rule-based rewrite suggestions. This
    endpoint is hit only when the user clicks the "Rewrite this sentence" button
    beside a flagged sentence and wants a more natural, LLM-generated version.
    The local Qwen2.5-0.5b takes ~5-15 s per sentence on CPU, so the call is
    dispatched into a thread pool to keep the event loop free; if the model is
    unavailable or raises, it falls back to the rule-based rewriter.

    Concurrency: a global Semaphore(1) guarantees only one rewrite runs at a
    time. Later requests queue for at most 60 s; a request that cannot acquire
    the semaphore gets a 429, which prevents concurrent work from saturating the
    CPU and slowing each other into 504s.
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
    """Batch rewrite (backed by the front-end "Rewrite top 3" button).

    Processes at most 3 sentences serially, guarded by the same semaphore: this
    keeps requests inside one batch from running concurrently and also makes the
    batch mutually exclusive with single /api/rewrite calls, so only one rewrite
    task is ever on the CPU. The front-end treats a batch as "wait ~60 s" and gets
    three results at once, which feels better than clicking three times.
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
    """Return the 10 most recent check records (powers the front-end de-AI curve)."""
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
