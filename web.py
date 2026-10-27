# -*- coding: utf-8 -*-
"""
Aurora Check Web 服务

启动：
    python web.py                 # 默认 http://127.0.0.1:8000
    python web.py --port 8080

接口：
    GET  /                 报告界面
    POST /api/analyze      上传文件（.txt/.md/.docx）
    POST /api/analyze-text 直接提交文本（JSON: {"text": "..."}）
"""
from __future__ import annotations

import argparse
import io
import os
import re
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import aurora_check as core

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")

app = FastAPI(title="Aurora Check", version="2.0")


class TextPayload(BaseModel):
    text: str
    filename: str = "直接输入的文本"


def _split_with_index(text: str):
    """把正文切成句子，同时保留原文偏移，供前端高亮定位。"""
    out = []
    pos = 0
    for s in core._split_sentences(text):
        idx = text.find(s, pos)
        if idx < 0:
            idx = pos
        out.append({"text": s, "start": idx, "end": idx + len(s)})
        pos = idx + len(s)
    return out


_AI_MARKERS = (
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "in conclusion", "overall", "it is important", "plays a crucial",
    "in today's", "it is worth noting", "on the other hand", "as a result",
)


def _sentence_breakdown(sentences, base):
    """给每个句子一个风险分，围绕总分上下浮动。

    调整依据（都是可解释的表层特征）：
      + 出现学术模板连接词        → 更像生成式模型
      + 句长贴近全文均值          → AI 句式均匀
      - 含具体数字/年份           → 人写更常给具体量
      - 含第一人称                → 人写更多自我指涉
      - 过短（<8 词）             → 口语化停顿
    """
    lenses = [len(s.split()) for s in sentences] or [1]
    mean_len = sum(lenses) / len(lenses)

    out = []
    for s, n in zip(sentences, lenses):
        low = s.lower()
        adj = 0.0
        adj += 0.30 * sum(1 for m in _AI_MARKERS if m in low)
        if mean_len > 0 and abs(n - mean_len) / mean_len < 0.25:
            adj += 0.10
        if re.search(r"\d", s):
            adj -= 0.10
        if re.search(r"\b(I|my|me|we|our)\b", s):
            adj -= 0.07
        if n < 8:
            adj -= 0.10
        score = max(0.02, min(0.97, base + adj))
        level = "high" if score >= 0.62 else ("mid" if score >= 0.35 else "low")
        out.append({"text": s, "score": round(score, 3), "level": level})
    return out


def _analyze(text: str, filename: str) -> dict:
    t0 = time.time()
    report = core.run_check(text=text)

    if not report.ai:
        return {
            "ok": False,
            "filename": filename,
            "reason": report.note,
            "chars": report.text_length,
            "elapsed": round(time.time() - t0, 2),
        }

    ai = report.ai
    prose, _refs = core._extract_prose_only(text)

    # 逐句风险。
    # 只用滑动窗口会给整段相同的分数（窗口平滑过度），16 句全红等于没有信息。
    # 这里以总分为基线，再按句子自身的特征上下调整，让风险形成梯度。
    # 按原始段落切分，段内再分句 —— 保持阅读形态，而不是每句独立成段
    paragraphs = []
    flat = []
    for para in [x.strip() for x in re.split(r"\n+", prose) if x.strip()]:
        sents = core._split_sentences(para)
        if not sents:
            continue
        scored = _sentence_breakdown(sents, ai.ai_probability)
        paragraphs.append(scored)
        flat.extend(scored)
    per_sentence = flat

    engines = [
        {"name": e["name"], "score": e["score"], "weight": e["weight"]}
        for e in ai.adopted_engines
    ]

    return {
        "ok": True,
        "filename": filename,
        "chars": report.text_length,
        "elapsed": round(time.time() - t0, 2),
        "score": ai.ai_probability,
        "label": ai.label,
        "confidence": ai.confidence,
        "disagreement": ai.engine_disagreement,
        "rewrite_hint": ai.rewrite_hint,
        "adopted": len(ai.adopted_engines),
        "total_engines": len(ai.engine_scores),
        "engines": engines,
        "rejected": [x["name"] for x in ai.rejected_engines],
        "sentences": per_sentence,
        "paragraphs": paragraphs,
        "stats": {
            "sentences": len(flat),
            "words": len(prose.split()),
            "high_risk": sum(1 for s in per_sentence if s["level"] == "high"),
        },
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(STATIC, "index.html")
    if not os.path.exists(path):
        raise HTTPException(500, "界面文件缺失")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "upload").lower()

    if name.endswith(".docx"):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            tmp.write(raw)
            tmp.close()
            text = core.extract_text_from_file(tmp.name)
            os.unlink(tmp.name)
        except Exception as e:
            raise HTTPException(400, f"无法解析 docx：{e}")
    else:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "文件内容为空")

    return JSONResponse(_analyze(text, file.filename or "upload"))


@app.post("/api/analyze-text")
async def analyze_text(payload: TextPayload):
    if not payload.text.strip():
        raise HTTPException(400, "文本为空")
    return JSONResponse(_analyze(payload.text, payload.filename))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0"}


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"Aurora Check 已启动 → http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
