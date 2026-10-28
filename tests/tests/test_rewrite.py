# -*- coding: utf-8 -*-
"""「AI 降维改写」Mock 版契约测试（3.12.0 新增）。

全部为快测：不加载深度学习模型，只验证 _mock_rewrite_sentence 的
纯规则降维行为，以及 to_html / to_markdown 对 rewrite_suggestion 的渲染。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import aurora_check as ac


# ---------------------------------------------------------------
# _mock_rewrite_sentence 纯函数
# ---------------------------------------------------------------

def test_rewrite_delve_into():
    assert ac._mock_rewrite_sentence("This essay will delve into the topic.") \
        == "This essay will talk about the topic."


def test_rewrite_utilize():
    assert ac._mock_rewrite_sentence("We should utilize this tool.") \
        == "We should use this tool."


def test_rewrite_in_my_opinion():
    assert ac._mock_rewrite_sentence("In my opinion, this method can help.") \
        == "I think this method can help."


def test_rewrite_long_sentence_splits():
    """超 20 词长句应在连词处硬拆成两个短句（连接词归第二句）。"""
    out = ac._mock_rewrite_sentence(
        "The approach is significant and subsequently fosters better results "
        "for students who engage with continuous improvement and professional "
        "development over time.")
    # significant→important、subsequently→later，并在 "and" 处拆句
    assert out.startswith("The approach is important."), f"实际: {out}"
    assert "And later" in out, f"连接词应归第二句，实际: {out}"


def test_rewrite_verb_agreement():
    """demonstrates→shows（主谓一致），不能机械砍成 show。"""
    out = ac._mock_rewrite_sentence("The overall conclusion demonstrates these ideas.")
    assert "shows" in out, f"demonstrates 应降维为 shows，实际: {out}"
    assert "conclusion show " not in out, "不应出现主谓不一致的 'conclusion show'"


def test_rewrite_no_change_returns_empty():
    """短句且无高级词/模板开头/超长时，返回空串表示「无需改写」。"""
    assert ac._mock_rewrite_sentence("This is a short sentence.") == ""


def test_rewrite_empty_input():
    assert ac._mock_rewrite_sentence("") == ""
    assert ac._mock_rewrite_sentence(None) == ""


# ---------------------------------------------------------------
# to_html / to_markdown 渲染 rewrite_suggestion
# ---------------------------------------------------------------

def _make_strict_report_with_rewrite() -> ac.CheckReport:
    ai = ac.AIResult(
        ai_probability=0.5,
        label="ai",
        method="ensemble v3.3[strict] (test)",
        mode="strict",
        student_verdict="高危（重度辅助痕迹，老师大概率怀疑）",
        engine_scores={"formulaic-patterns": 0.8, "paragraph-rhythm": 0.7},
        adopted_engines=[
            {"name": "formulaic-patterns", "score": 0.8, "weight": 0.12, "note": "公式化表达"},
        ],
        flagged_sentences=[
            {"text": "This essay will delve into the comprehensive framework.",
             "score": 0.8, "uncertainty": 0.05, "confidence": "high",
             "strategy": "含高级词汇「delve」，请换成大白话。",
             "rewrite_suggestion": "This essay will talk about the full framework."},
        ],
        confidence="medium",
    )
    return ac.CheckReport(text_length=300, ai=ai)


def test_to_html_renders_rewrite_suggestion():
    html = _make_strict_report_with_rewrite().to_html()
    assert "修改建议：" in html, "高危句下方应渲染「修改建议」"
    assert "class='rewrite'" in html, "修改建议应用绿色 .rewrite 样式"
    assert "talk about the full framework" in html, "应展示改写后的文本"


def test_to_markdown_renders_rewrite_suggestion():
    md = _make_strict_report_with_rewrite().to_markdown()
    assert "修改建议：" in md, "markdown 报告应渲染「修改建议」"
    assert "27ae60" in md, "markdown 端应用绿色 span 标记"
