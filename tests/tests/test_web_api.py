# -*- coding: utf-8 -*-
"""Web 可视化（to_html）+ FastAPI（server.py）契约测试。

全部为「快测」：不加载深度学习模型。
- to_html 用手工构造的 CheckReport 验证渲染契约；
- server.py 只测校验路径（400）与健康检查（200），不触发真实检测。
"""
import json
import os
import re
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import aurora_check as ac


def _make_report() -> ac.CheckReport:
    """构造一个不含真实检测的 CheckReport，专测 to_html 渲染。"""
    ai = ac.AIResult(
        ai_probability=0.45,
        label="mixed",
        method="ensemble v3.3[strict] (test)",
        mode="strict",
        student_verdict="危险（会被老师怀疑用了 AI）",
        high_pressure_warning="⚠️ 极度危险！请检查段落开头镜像。",
        engine_scores={
            "formulaic-patterns": 0.8, "paragraph-rhythm": 0.7, "burstiness": 0.6,
            "mlm-perplexity": 0.55, "lexical-diversity": 0.4,
        },
        adopted_engines=[
            {"name": "formulaic-patterns", "score": 0.8, "weight": 0.12, "note": "公式化表达"},
            {"name": "paragraph-rhythm", "score": 0.7, "weight": 0.07, "note": "段落长度均匀度"},
            {"name": "burstiness", "score": 0.6, "weight": 0.10, "note": "突发度"},
            {"name": "mlm-perplexity", "score": 0.55, "weight": 0.10,
             "note": "MLM 伪困惑度（专攻表演式不完美拟人 AI，需本地 ModernBERT-base）"},
            {"name": "lexical-diversity", "score": 0.4, "weight": 0.05, "note": "词汇多样性"},
        ],
        flagged_sentences=[
            {"text": "The authors delve into the multifaceted nature of employability.",
             "score": 0.75, "uncertainty": 0.05, "confidence": "high",
             "strategy": "含高级词汇「delve」，请换成大白话。"},
        ],
        confidence="medium",
        rewrite_hint="词汇层仍有 AI 特征",
    )
    plag = ac.PlagiarismResult(external_similarity=0.0, self_overlap=0.27,
                               network_note="已联网查询 Crossref")
    cite = ac.CitationResult(total_extracted=3, resolved=3, failed=[], bibtex="")
    return ac.CheckReport(text_length=500, ai=ai, plagiarism=plag, citations=cite)


# ---------------------------------------------------------------
# to_html 渲染契约
# ---------------------------------------------------------------

def test_to_html_contains_echarts_and_visual_anchors():
    """可视化报告必须包含 ECharts、仪表盘、雷达图、气泡、高危句下划线。"""
    html = _make_report().to_html()
    assert "cdn.jsdelivr.net" not in html, "to_html 应内联本地 ECharts，不再依赖 CDN"
    assert "Apache Software Foundation" in html, "应内联 echarts 源码（含其 License 头）"
    assert 'id="gauge"' in html, "应有仪表盘容器"
    assert 'id="radar"' in html, "应有雷达图容器"
    assert ".bubble" in html, "应有悬停气泡样式"
    assert "<mark>" in html, "高危句应用 <mark> 红色下划线标记"
    assert "严苛找茬模式" in html, "strict 模式应标注"


def test_to_html_radar_top5_and_truncates_long_name():
    """雷达图数据：只取前 5 引擎，超长名称截断到 16 字符内。"""
    html = _make_report().to_html()
    m = re.search(r"var radarData = (\[.*?\]);", html)
    assert m, "应输出雷达图 JSON 数据"
    data = json.loads(m.group(1))
    assert len(data) == 5, f"雷达图应只取前 5 引擎，实际 {len(data)}"
    # MLM 那条 note 特别长，必须被截断，否则雷达图标签撑爆
    long_notes = [d for d in data if "MLM" in d["name"]]
    assert long_notes and len(long_notes[0]["name"]) <= 16, "超长引擎名应截断"


def test_to_html_invalid_input_returns_warning_page():
    """输入不合格（无 ai、有 note）应返回「无法判定」警告页，不渲染分数。"""
    html = ac.CheckReport(text_length=10, note="文本太短", ai=None).to_html()
    assert "无法判定" in html
    assert "文本太短" in html
    assert 'id="gauge"' not in html, "警告页不应渲染仪表盘"


def test_to_html_human_label_renders_student_friendly():
    """normal 模式不出现 strict 专属的学生解读/高压预警。"""
    ai = ac.AIResult(ai_probability=0.1, label="human", method="test", mode="normal")
    html = ac.CheckReport(text_length=200, ai=ai).to_html()
    assert "严苛找茬模式" not in html, "normal 模式不应标注严苛"


# ---------------------------------------------------------------
# server.py 契约（TestClient，不跑真实模型）
# ---------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "strict" in body["modes"]


def test_index_serves_visual_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Aurora Check" in r.text
    assert "echarts.min.js" in r.text, "首页应引入 ECharts"


def test_check_rejects_invalid_mode(client):
    r = client.post("/api/check", json={"text": "hello world " * 20, "mode": "bogus"})
    assert r.status_code == 400, f"非法 mode 应 400，实际 {r.status_code}"


def test_check_rejects_blank_text(client):
    # pydantic 层放行（长度>0），业务层应因 strip 后为空返回 400
    r = client.post("/api/check", json={"text": "   ", "mode": "strict"})
    assert r.status_code == 400, f"纯空格文本应 400，实际 {r.status_code}"
