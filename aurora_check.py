"""
极光查重 / Aurora Check — 核心模块 v2.0

用法：
    from aurora_check import run_check
    report = run_check(text="你的文章内容")
    print(report.to_markdown())

命令行：
    python aurora_check.py <文件> [--json] [--no-html]
    python aurora_check.py compare <文档A> <文档B>

设计原则：
    只输出能被证据支持的结论。文本过短、重复或非自然语言时拒绝判定，
    而不是给一个看似精确的分数。
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import zlib
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, NamedTuple, Optional

# ------------------------------------------------------------------
# HuggingFace 默认短超时。
# 未联网时 huggingface_hub 会按 1→2→4→8→16s 退避重试 5 次，单次检测
# 因此多等 2 分钟以上，并在 stderr 喷出一大段红色警告。
# 这里把超时压到秒级；真正需要模型时用户可自行覆盖这些变量。
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "4")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_VERBOSE = os.environ.get("AURORA_VERBOSE", "0") == "1"


# ============================================================
# 数据结构
# ============================================================

class EngineSpec(NamedTuple):
    """单个引擎的注册信息。

    原来是裸三元组 (+1, 0.25, "备注")，第三位写进去却没有任何地方读——
    等于注释伪装成数据，改的时候还得数位置。具名之后意图明确，
    解包写法 (d, w, n) 仍然兼容。
    """
    direction: int      # +1 分数高=像 AI，0 表示不参与
    weight: float       # 相对权重
    note: str = ""      # 给人看的说明，不参与计算


@dataclass
class AIResult:
    ai_probability: float       # 0~1
    label: str                 # human / mixed / ai
    method: str
    engine_scores: dict = field(default_factory=dict)
    flagged_sentences: list = field(default_factory=list)
    engine_disagreement: float = 0.0  # 引擎分歧度（加权标准差）
    confident_engines: list = field(default_factory=list)  # 高度确信AI的引擎
    conflicting_engines: list = field(default_factory=list)  # 分歧大的引擎对
    confidence: str = "low"          # high / medium / low
    adopted_engines: list = field(default_factory=list)   # 参与加权的引擎
    rejected_engines: list = field(default_factory=list)  # 被弃用的引擎及原因
    rewrite_hint: str = ""           # 改写嫌疑提示（非空表示命中）


@dataclass
class PlagiarismResult:
    external_similarity: float  # 0~1
    self_overlap: float          # 0~1
    matched_excerpts: list = field(default_factory=list)


@dataclass
class CitationResult:
    total_extracted: int = 0
    resolved: int = 0
    failed: list = field(default_factory=list)
    bibtex: str = ""


# 实测参数（40 篇人类文本零误判，按 rule of three 取 95% 上界）
_EST_TPR = 0.90      # AI 稿检出率，取保守值
_EST_FPR = 0.075     # 人类误判率上界


def _positive_predictive_value(base_rate: float = 0.10) -> float:
    """判为 AI 时，实际确为 AI 的概率（阳性预测值）。

    这一步很关键，但几乎所有检测工具都不告诉用户：
    检测器的"准确率"和"判 AI 时有多可信"是两件事。

    FPR 即使只有 7.5%，当 AI 稿在真实场景里只占 5% 时，
    被判 AI 的稿件里仍有六成是冤枉的——低基准率会把假阳性放大。
    这不是本工具的问题，贝叶斯下任何检测器都一样。
    """
    return (base_rate * _EST_TPR) / (base_rate * _EST_TPR + (1 - base_rate) * _EST_FPR)


@dataclass
class CheckReport:
    text_length: int = 0
    note: str = ""          # 输入不合格时的说明（此时 ai 为 None）
    ai: Optional[AIResult] = None
    plagiarism: Optional[PlagiarismResult] = None
    citations: Optional[CitationResult] = None

    def to_dict(self) -> dict:
        """导出为纯 dict，便于 JSON 输出"""
        return asdict(self)

    def to_markdown(self) -> str:
        lines = ["# 极光查重报告\n"]
        lines.append(f"- **正文长度：** {self.text_length} 字符\n")
        if self.note:
            lines.append(f"## ⛔ 无法判定\n\n{self.note}\n")
            lines.append("> 文本不满足最低统计要求，任何分数都不可靠，故不出具结论。\n")
            return "\n".join(lines)

        # AI 检测
        if self.ai:
            ai = self.ai
            lines.append("## AI 检测\n")
            lines.append(f"- **综合判定：** {_label_to_cn(ai.label)}")
            # 「概率」这个词是误导：它是多个信号的加权强度，不是校准过的发生概率。
            # 统计审慎的视角下必须改口，否则用户会把 73% 读成「73% 的可能性是 AI」。
            lines.append(f"- **AI 特征强度：** {ai.ai_probability:.1%}"
                         f"　（多个信号的加权结果，非发生概率）")
            lines.append(f"- **检测方法：** {ai.method}\n")
            _conf_cn = {"high": "高", "medium": "中", "low": "低"}.get(ai.confidence, ai.confidence)
            if ai.label == "ai":
                _ppv = _positive_predictive_value()
                lines.append(f"- **若据此认定，命中率约：** {_ppv:.0%}"
                             f"（假设你的场景中 AI 稿占 10%）")
                lines.append(f"  - 也就是说：被判 AI 的稿子里，约 {1-_ppv:.0%} 其实是人写的。"
                             "这一比例随 AI 稿占比下降而急剧恶化——"
                             "若占比 5%，命中率只剩 39%。")
                lines.append("  - 分数高只说明「文本特征像 AI」，不构成抄袭认定。")
                if ai.confidence == "low":
                    lines.append("  - ⚠️ 本条置信度为低，模型自身也不确定")
                lines.append("")

            if ai.rewrite_hint:
                lines.append(f"- **⚠️ 改写嫌疑：** {ai.rewrite_hint}")
                lines.append("")
            lines.append(f"- **置信度：** {_conf_cn}"
                         f"（引擎加权分歧度 {ai.engine_disagreement:.3f}）")
            if ai.confidence == "low":
                lines.append("  - ⚠️ 引擎分歧大，该判定不可靠，建议以人工阅读为准")
            lines.append("")

            if ai.adopted_engines:
                lines.append(f"- **参与加权引擎（{len(ai.adopted_engines)} 个，按影响排序）：**")
                for e in ai.adopted_engines:
                    bar = "█" * int(e["score"] * 10) + "░" * (10 - int(e["score"] * 10))
                    lines.append(f"  - {e['name']}：{e['score']:.1%} {bar} ×{e['weight']}")
                lines.append("")
            if ai.rejected_engines:
                lines.append(f"<details><summary>已弃用引擎（{len(ai.rejected_engines)} 个，不参与计算）</summary>\n")
                for e in ai.rejected_engines:
                    lines.append(f"- {e['name']}：{e['reason']}")
                lines.append("\n</details>\n")
            if ai.flagged_sentences:
                lines.append("- **高风险句（Top 5）：**")
                for s in ai.flagged_sentences[:5]:
                    confidence_icon = "🟢" if s.get("confidence") == "high" else ("🟡" if s.get("confidence") == "medium" else "🔴")
                    lines.append(f"  - {confidence_icon} ({s['score']:.2f}) {s['text'][:80]}...")
                lines.append("")
                lines.append("  - 🟢 高确信 | 🟡 中等确信 | 🔴 低确信（引擎分歧大）")
                lines.append("")

        # 相似度
        if self.plagiarism:
            p = self.plagiarism
            lines.append("## 相似度 / 查重\n")
            lines.append(f"- **外部相似度：** {p.external_similarity:.2%}")
            lines.append(f"- **章节自重叠：** {p.self_overlap:.2%}\n")

        # 引用验证
        if self.citations:
            c = self.citations
            lines.append("## 引用验证\n")
            lines.append(f"- **总共提取：** {c.total_extracted} 条")
            lines.append(f"- **解析成功：** {c.resolved} 条")
            if c.failed:
                lines.append(f"- **失败：** {len(c.failed)} 条")
                for f in c.failed[:5]:
                    lines.append(f"  - {f}")
            lines.append("")
            if c.bibtex:
                lines.append("<details><summary>规范 BibTeX</summary>\n")
                lines.append("```bibtex")
                lines.append(c.bibtex)
                lines.append("```\n</details>\n")

        lines.append("---\n_由极光查重 (Aurora Check) 生成_")
        return "\n".join(lines)

    def to_html(self, output_path: str = "") -> str:
        """生成 Turnitin 风格的 HTML 报告"""
        # 输入不合格：只输出警告页，不再渲染任何分数
        if self.note and not self.ai:
            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>无法判定 - Aurora Check</title>
<style>
  body {{ margin:0; padding:40px 20px; background:#f5f5f5;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }}
  .card {{ max-width:640px; margin:0 auto; background:#fff; border-radius:12px;
          border-left:6px solid #e74c3c; padding:36px 40px;
          box-shadow:0 2px 20px rgba(0,0,0,.08); }}
  h1 {{ font-size:22px; color:#e74c3c; margin:0 0 16px; }}
  p {{ color:#444; line-height:1.8; margin:0 0 12px; }}
  .hint {{ color:#888; font-size:13px; margin-top:20px; }}
</style></head>
<body><div class="card">
  <h1>⛔ 无法判定</h1>
  <p>{self.note}</p>
  <p class="hint">文本不满足最低统计要求，任何分数都不可靠，故不出具结论。</p>
  <p class="hint">正文长度：{self.text_length} 字符</p>
</div></body></html>
"""
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)
            return html


        ai_prob = self.ai.ai_probability if self.ai else 0
        ai_pct = int(round(ai_prob * 100))

        # v2：改用与 to_markdown 相同的 label，避免同一份结果
        # 在 Markdown 里说"人工写作"、在 HTML 里说"混合"。
        _LAB = {"ai": ("#e74c3c", "AI 特征显著"),
                "mixed": ("#f39c12", "特征混杂"),
                "human": ("#27ae60", "未见明显 AI 特征")}
        _key = self.ai.label if self.ai else "human"
        color, label = _LAB.get(_key, ("#f39c12", "混合（人机各半）"))

        # 引擎评分行
        engine_rows = ""
        if self.ai and self.ai.engine_scores:
            scores = list(self.ai.engine_scores.values())
            avg_score = sum(scores) / len(scores) if scores else 0
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores) if scores else 0
            std_dev = variance ** 0.5

            # 引擎分歧统计
            high_confidence = sum(1 for s in scores if s > 0.7 or s < 0.3)
            low_confidence = len(scores) - high_confidence
            consensus = "高共识" if std_dev < 0.2 else "中分歧" if std_dev < 0.35 else "高分歧"

            _adopted = self.ai.adopted_engines or [
                {"name": k, "score": v, "weight": 0} for k, v in self.ai.engine_scores.items()]
            for e in _adopted:
                name, score = e["name"], e["score"]
                pct = int(round(score * 100))
                bar_color = "#e74c3c" if score > 0.62 else "#f39c12" if score > 0.35 else "#27ae60"
                engine_rows += f"""
                <tr>
                    <td>{name}<span style="color:#aaa;font-size:11px;"> ×{e.get('weight', 0)}</span></td>
                    <td style="width: 200px;">
                        <div style="background: #eee; height: 20px; border-radius: 10px; overflow: hidden;">
                            <div style="background: {bar_color}; width: {pct}%; height: 100%;"></div>
                        </div>
                    </td>
                    <td style="text-align: right; font-weight: bold;">{pct}%</td>
                </tr>
                """
            if self.ai.rejected_engines:
                _rej = "".join(
                    f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;'
                    f'background:#eee;border-radius:10px;font-size:11px;color:#888;">'
                    f'{x["name"]}</span>'
                    for x in self.ai.rejected_engines)
                engine_rows += (f'<tr><td colspan="3" style="padding-top:16px;color:#888;font-size:12px;">'
                                f'已弃用（不参与计算）：<br>{_rej}</td></tr>')

            # 引擎分歧面板
            divergence_panel = f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="font-size: 14px; color: #666; margin-bottom: 10px;">引擎分歧分析</h3>
                <div style="display: flex; gap: 20px; font-size: 13px;">
                    <div>
                        <span style="color: #666;">共识强度:</span>
                        <span style="font-weight: bold; color: {color};">{consensus}</span>
                    </div>
                    <div>
                        <span style="color: #666;">标准差:</span>
                        <span style="font-weight: bold;">{std_dev:.3f}</span>
                    </div>
                    <div>
                        <span style="color: #666;">高确信引擎:</span>
                        <span style="font-weight: bold;">{high_confidence}/{len(scores)}</span>
                    </div>
                </div>
            </div>
            """
        else:
            divergence_panel = ""

        # 高风险句
        flagged_html = ""
        if self.ai and self.ai.flagged_sentences:
            for s in self.ai.flagged_sentences[:10]:
                score_pct = int(s["score"] * 100)
                # 置信度区间：±15%
                low = max(0, score_pct - 15)
                high = min(100, score_pct + 15)
                bg = "rgba(231, 76, 60, 0.15)" if score_pct > 50 else "rgba(243, 156, 18, 0.15)" if score_pct > 20 else "rgba(39, 174, 96, 0.15)"
                flagged_html += f"""
                <div style="background: {bg}; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid {color};">
                    <div style="font-size: 12px; color: #666; margin-bottom: 4px;">
                        AI 嫌疑度: {score_pct}%
                        <span style="color: #999; margin-left: 10px;">置信区间: [{low}% - {high}%]</span>
                    </div>
                    <div>{s['text']}</div>
                </div>
                """

        _conf_map = {"high": "高", "medium": "中", "low": "低"}
        _conf = _conf_map.get(self.ai.confidence, "—") if self.ai else "—"
        _hint_html = ""
        if self.ai and self.ai.rewrite_hint:
            _hint_html = (f'<div style="margin-top:14px;padding:12px 16px;background:#fff8e1;'
                          f'border-left:4px solid #f39c12;border-radius:6px;font-size:13px;'
                          f'color:#8a6d3b;text-align:left;">⚠️ 改写嫌疑：{self.ai.rewrite_hint}</div>')

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>极光查重报告 - Aurora Check</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 20px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 32px; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; }}
        .score-section {{ padding: 40px; text-align: center; }}
        .big-score {{ font-size: 72px; font-weight: bold; color: {color}; }}
        .score-label {{ font-size: 20px; color: #666; margin-top: 8px; }}
        .section {{ padding: 30px 40px; border-top: 1px solid #eee; }}
        .section h2 {{ font-size: 20px; margin-bottom: 20px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ font-weight: 600; color: #666; font-size: 14px; }}
        .footer {{ padding: 20px 40px; background: #f9f9f9; text-align: center; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>极光查重报告</h1>
            <p>Aurora Check - AI 文本检测</p>
        </div>

        <div class="score-section">
            <div class="big-score">{ai_pct}%</div>
            <div class="score-label">{label}</div>
            <div style="margin-top:6px;font-size:12px;color:#999;">特征强度，非发生概率；不构成抄袭认定</div>
            <div style="margin-top: 10px; font-size: 13px; color: #888;">
                置信度：{_conf}　·　采纳引擎：{len(self.ai.adopted_engines) if self.ai else 0}/{len(self.ai.engine_scores) if self.ai else 0}
            </div>
            {_hint_html}
            <p style="margin-top: 20px; color: #666; font-size: 14px;">正文长度: {self.text_length} 字符</p>
        </div>

        <div class="section">
            <h2>各引擎详细评分</h2>
            {divergence_panel}
            <table>
                <thead>
                    <tr>
                        <th>引擎名称</th>
                        <th>评分</th>
                        <th style="text-align: right;">百分比</th>
                    </tr>
                </thead>
                <tbody>
                    {engine_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>高风险句（AI 嫌疑）</h2>
            {flagged_html if flagged_html else '<p style="color: #999;">没有检测到高风险句子</p>'}
        </div>

        <div class="footer">
            由极光查重 (Aurora Check v2) 生成 · 共 29 个引擎，{len(self.ai.adopted_engines) if self.ai else 0} 个参与加权
        </div>
    </div>
</body>
</html>
"""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def _label_to_cn(label: str) -> str:
    return {
        "human": "未见明显 AI 特征",
        "mixed": "特征混杂",
        "ai": "AI 特征显著",
    }.get(label, label)


# ============================================================
# 文本提取
# ============================================================

def extract_text_from_file(file_path: str) -> str:
    """从 txt / md / docx 文件提取纯文本"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            # fallback: 直接读（会乱码，但至少不崩）
            with open(file_path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")

    # 默认按文本读
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# AI 检测引擎
# ============================================================

def _detect_with_lmscan(text: str) -> Optional[float]:
    """调用 lmscan CLI，返回 AI 概率"""
    try:
        result = subprocess.run(
            ["lmscan", "-", "--format", "json"],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("ai_probability", 0.5))
    except Exception:
        pass
    return None


def _detect_with_baseline(text: str) -> tuple[float, list]:
    """纯 Python 统计基线：句长方差 + 词型丰富度"""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.5, []

    # 句长
    sent_lengths = [int(_word_count(s)) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)

    # 词型丰富度
    words = text.lower().split()
    if not words:
        return 0.5, []
    ttr = len(set(words)) / len(words)

    # 打分：AI 文本倾向于句长均匀（方差小）、词型丰富度适中
    burstiness = variance / (avg_len ** 2 + 1)  # 越大越像人

    # 基于 burstiness 的线性打分：0.25 → 0.75, 0.55 → 0.45, 0.85 → 0.15
    if burstiness < 0.25:
        ai_score = 0.75
    elif burstiness < 0.50:
        ai_score = 0.55
    elif burstiness < 0.75:
        ai_score = 0.35
    else:
        ai_score = 0.15

    # TTR 微调：TTR 太高（>0.60）可能是 AI 堆砌词汇，加一点分
    if ttr > 0.60:
        ai_score += 0.08
    elif ttr < 0.40:
        ai_score -= 0.08

    # 高风险句：最短或最长的句子
    flagged = []
    for s in sorted(sentences, key=len)[:3]:
        flagged.append({"text": s, "score": round(ai_score, 2)})
    for s in sorted(sentences, key=len, reverse=True)[:2]:
        flagged.append({"text": s, "score": round(ai_score * 0.8, 2)})

    return min(max(ai_score, 0), 1), flagged


def _detect_perplexity_burstiness(text: str) -> Optional[float]:
    """困惑度 + 突发性近似（纯 Python）"""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return None

    lengths = [int(_word_count(s)) for s in sentences]
    avg = sum(lengths) / len(lengths)
    std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
    burstiness = std / (avg + 1)

    # 突发性低 = 更像 AI
    if burstiness < 0.25:
        ai_prob = 0.75
    elif burstiness < 0.45:
        ai_prob = 0.55
    elif burstiness < 0.65:
        ai_prob = 0.35
    else:
        ai_prob = 0.15

    return ai_prob


# ============================================================
# lynote 逆向集成的 4 个新信号
# ============================================================

# AI 常用短语列表（来自 lynote-ai/ai-text-detector 逆向，打磨扩充）
_AI_FORMULAIC_MARKERS = [
    # 原有模式
    r"\bas an ai\b", r"\bi cannot\b", r"\bit is important to note\b",
    r"\bdelve\b", r"\bnuanced\b", r"\btestament to\b", r"\bunlock\b",
    r"\bcomprehensive\b", r"\bin conclusion\b", r"\boverall\b",
    r"\bnot only .* but also\b", r"\bwhile .* it is also\b",
    r"\bin today's (rapidly )?(changing|evolving)\b",
    r"\bplays a (crucial|vital|pivotal) role\b",
    r"\blandscape\b", r"\brealm\b", r"\btapestry\b",
    # 新增模式
    r"\bit is worth noting\b", r"\bit is crucial to\b", r"\bit is essential to\b",
    r"\bit is important to understand\b", r"\bit is worth considering\b",
    r"\blet's dive in\b", r"\blet's explore\b", r"\blet's take a look\b",
    r"\bwhen it comes to\b", r"\bin the realm of\b", r"\bin the world of\b",
    r"\bparadigm shift\b", r"\bgame-changer\b", r"\bcutting-edge\b",
    r"\bgroundbreaking\b", r"\brevolutionary\b", r"\btransformative\b",
    r"\bseamless integration\b", r"\bseamless experience\b",
    r"\buser-friendly\b", r"\bcutting-edge technology\b",
    r"\bstate-of-the-art\b", r"\bnext-generation\b",
    # 继续扩充
    r"\bit is no secret that\b", r"\bit is widely recognized that\b",
    r"\bit is undeniable that\b", r"\bit is clear that\b",
    r"\bit is obvious that\b", r"\bit is evident that\b",
    r"\bwe live in an era\b", r"\bin today's world\b",
    r"\bas we all know\b", r"\bas you may know\b",
    r"\bhave you ever wondered\b", r"\bwhat if i told you\b",
    r"\blet's face it\b", r"\blet's be honest\b",
    r"\bthe future is\b", r"\bthe possibilities are\b",
    r"\bthe journey continues\b", r"\bthe story unfolds\b",
]

# 过渡词列表（来自 lynote 逆向，打磨扩充）
_TRANSITION_WORDS = [
    # 原有词
    "first", "second", "third", "moreover", "furthermore", "however", "therefore",
    "ultimately", "additionally", "in summary", "in conclusion", "on the other hand",
    "consequently", "nevertheless", "nonetheless", "thus", "hence",
    # 新增词
    "in addition", "besides", "also", "what's more",
    "in contrast", "conversely", "on the contrary",
    "as a result", "for this reason", "because of this",
    "in other words", "that is", "namely",
    "for example", "for instance", "such as",
    "in particular", "specifically", "notably",
    "generally", "typically", "usually",
    "essentially", "basically", "effectively",
    "finally", "lastly", "to sum up",
    "overall", "all things considered",
    "in short", "in brief",
    # 继续扩充
    "firstly", "secondly", "thirdly",
    "furthermore", "moreover", "additionally",
    "consequently", "therefore", "thus", "hence",
    "nevertheless", "nonetheless", "however",
    "meanwhile", "meantime", "in the meantime",
    "subsequently", "afterwards", "after that",
    "previously", "before that", "earlier",
    "currently", "at present", "nowadays",
    "eventually", "ultimately", "in the end",
    "luckily", "unfortunately", "sadly",
    "thankfully", "hopefully", "fortunately",
]


def _detect_formulaic_markers(text: str) -> float:
    """检测 AI 常用短语（逆向自 lynote，打磨版）"""
    text_lower = text.lower()
    marker_hits = sum(1 for pat in _AI_FORMULAIC_MARKERS if re.search(pat, text_lower, re.I | re.S))
    # 打磨：10 个以上标记 = 0.9 AI 概率
    return max(0.0, min(1.0, marker_hits / 10.0))


def _detect_transition_density(text: str) -> float:
    """检测过渡词密度（逆向自 lynote，打磨版）"""
    words = text.lower().split()
    word_count = len(words)
    if word_count < 50:
        return 0.3

    transition_hits = sum(text.lower().count(t) for t in _TRANSITION_WORDS)
    # 打磨：每 100 词 3 个过渡词 = 0.5
    density = transition_hits / word_count * 100
    return max(0.0, min(1.0, density / 6.0))


def _detect_compressibility(text: str) -> float:
    """检测压缩率（逆向自 lynote，精雕细琢版）- 可压缩性越高 = 越像 AI"""
    compressed = len(zlib.compress(text.encode("utf-8")))
    original = len(text.encode("utf-8"))
    ratio = compressed / max(1, original)
    # 精雕细琢：典型值调整
    # ratio 越低 = 越可压缩 = 越重复 = 越像 AI
    # 归一化：ratio=0.25 → 0.9 AI，ratio=0.65 → 0.1 AI
    return max(0.0, min(1.0, (0.50 - ratio) / 0.25))


def _detect_structured_shape(text: str) -> float:
    """检测结构化答案形状（逆向自 lynote，精雕细琢版）"""
    listish_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text))
    headingish_lines = len(re.findall(r"(?m)^\s{0,3}#{1,3}\s+|^\s{0,3}[A-Z][A-Za-z ]{3,}:\s*$", text))
    # 精雕细琢：加了加粗检测
    boldish_lines = len(re.findall(r"\*\*[^*]+\*\*", text))
    # 精雕细琢：加了表格检测
    tableish_lines = len(re.findall(r"(?m)^\s*\|.*\|\s*$", text))
    total = listish_lines + headingish_lines + boldish_lines + tableish_lines
    # 精雕细琢：8 个以上结构化行 = 0.9
    return max(0.0, min(1.0, total / 8.0))


# ============================================================
# sloptotal 逆向集成：Formulaic 公式化表达检测
# ============================================================

_AI_OPENERS = [
    # 原有模式
    r"in today'?s (?:rapidly )?(?:evolving|changing|digital|modern)",
    r"in (?:the|a) (?:world|era|age|landscape) (?:where|of|that)",
    r"when it comes to",
    r"in the realm of",
    r"let'?s (?:dive|explore|take a (?:closer )?look|unpack|break down)",
    r"imagine (?:this|a world)",
    r"as (?:technology|the world|we|society) continues to",
    r"the (?:rise|emergence|advent|proliferation) of",
    # 新增模式
    r"it'?s no secret that",
    r"we live in an era where",
    r"in recent years,",
    r"with the rapid advancement of",
    r"as artificial intelligence (?:continues|becomes)",
    r"the world is changing (?:rapidly|fast|dramatically)",
    r"in an increasingly (?:digital|connected|globalized)",
    r"have you ever wondered",
    r"what if i told you",
    r"let'?s face it,",
    # 继续扩充
    r"in today'?s world,",
    r"as we all know,",
    r"it'?s widely recognized that",
    r"there is no denying that",
    r"it'?s undeniable that",
    r"we'?ve all heard about",
    r"you might be wondering",
    r"the question is not whether",
    r"the real question is",
    r"let'?s start with",
]

_AI_CLOSERS = [
    # 原有模式
    r"in conclusion,?",
    r"(?:in summary|to summarize|to sum up),?",
    r"(?:ultimately|at the end of the day),?",
    r"as we (?:navigate|move forward|look ahead|continue)",
    r"by (?:embracing|leveraging|harnessing|adopting)",
    r"it'?s clear that",
    # 新增模式
    r"the future is (?:bright|promising|exciting)",
    r"only time will tell",
    r"it remains to be seen",
    r"as we move forward,",
    r"this is just the beginning",
    r"the possibilities are endless",
    r"we'?re only scratching the surface",
    r"the journey continues",
    r"in the grand scheme of things",
    r"all things considered,",
    # 继续扩充
    r"the bottom line is",
    r"the key takeaway is",
    r"the main point is",
    r"the important thing to remember is",
    r"in the end,",
    r"at the end of the day,",
    r"when all is said and done,",
    r"to put it in a nutshell,",
    r"in a word,",
    r"in short,",
]

_FILLER_PATTERNS = [
    # 原有模式
    r"not only\b.{3,60}\bbut also",
    r"on (?:the )?one hand\b.{3,120}\bon the other(?: hand)?",
    r"it(?:'?s| is) (?:important|worth|crucial|essential) to (?:note|remember|mention)",
    r"first and foremost",
    r"last but not least",
    r"without (?:a )?doubt",
    # 新增模式
    r"it goes without saying that",
    r"needless to say,",
    r"as a matter of fact,",
    r"in other words,",
    r"that is to say,",
    r"to put it simply,",
    r"basically,",
    r"essentially,",
    r"effectively,",
    r"in essence,",
    # 继续扩充
    r"it'?s worth mentioning that",
    r"it'?s important to understand that",
    r"it'?s crucial to recognize that",
    r"it'?s essential to acknowledge that",
    r"we must also remember that",
    r"it should be noted that",
    r"it can be argued that",
    r"some might say that",
    r"it'?s often said that",
    r"many people believe that",
]


def _detect_formulaic(text: str) -> float:
    """检测 AI 公式化表达（逆向自 sloptotal FormulaicEngine）"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)

    # 检测开头套路（前 300 字符）
    opener_score = 0.0
    opener_text = text_lower[:300]
    for pattern in _AI_OPENERS:
        if re.search(pattern, opener_text):
            opener_score = 1.0
            break

    # 检测结尾套路（后 400 字符）
    closer_score = 0.0
    closer_text = text_lower[-400:]
    for pattern in _AI_CLOSERS:
        if re.search(pattern, closer_text):
            closer_score = 1.0
            break

    # 检测填充模式
    filler_count = 0
    for pattern in _FILLER_PATTERNS:
        filler_count += len(re.findall(pattern, text_lower))
    filler_density = (filler_count / word_count) * 1000
    filler_score = min(filler_density / 5.0, 1.0)

    # 加权打分
    final_score = (
        opener_score * 0.20
        + closer_score * 0.15
        + filler_score * 0.65
    )
    return round(final_score, 3)


# ============================================================
# sloptotal 逆向集成：Linguistic 语言标志检测
# ============================================================

_AI_MARKERS_WEIGHTED = [
    # Tier 1: Very strong AI signals (weight 3)
    ("delve", 3),
    ("tapestry", 3),
    ("multifaceted", 3),
    ("ever-evolving", 3),
    ("thought-provoking", 3),
    ("it's important to note", 3),
    ("it's worth noting", 3),
    ("in today's digital age", 3),
    ("in today's rapidly evolving", 3),
    ("let's dive in", 3),
    # Tier 2: Strong signals (weight 2)
    ("landscape", 2),
    ("navigate", 2),
    ("leverage", 2),
    ("foster", 2),
    ("pivotal", 2),
    ("nuanced", 2),
    ("robust", 2),
    ("holistic", 2),
    ("synergy", 2),
    ("paradigm", 2),
    ("encompass", 2),
    ("intricate", 2),
    ("comprehensive", 2),
    ("underscores", 2),
    ("underscore", 2),
    ("realm", 2),
    ("cornerstone", 2),
    ("underpinning", 2),
    ("facilitating", 2),
    ("harnessing", 2),
    ("spearheading", 2),
    ("revolutionize", 2),
    ("groundbreaking", 2),
    ("cutting-edge", 2),
    ("game-changer", 2),
    ("deep dive", 2),
    ("at the forefront", 2),
    ("at its core", 2),
    ("in the realm of", 2),
    ("it is crucial", 2),
    ("it is essential", 2),
    ("plays a crucial role", 2),
    ("serves as a testament", 2),
    ("is a testament to", 2),
    ("a myriad of", 2),
    ("plethora", 2),
    # Tier 3: Moderate signals (weight 1)
    ("crucial", 1),
    ("enhance", 1),
    ("dynamic", 1),
    ("innovative", 1),
    ("streamline", 1),
    ("optimize", 1),
    ("elevate", 1),
    ("empower", 1),
    ("stakeholder", 1),
    ("ecosystem", 1),
    ("actionable", 1),
    ("seamless", 1),
    ("seamlessly", 1),
    ("furthermore", 1),
    ("moreover", 1),
    ("consequently", 1),
    ("nevertheless", 1),
    ("in conclusion", 1),
    ("ultimately", 1),
    ("in essence", 1),
    ("it is important to", 1),
    ("not only", 1),
    ("but also", 1),
    ("on the other hand", 1),
    ("having said that", 1),
    ("that being said", 1),
    ("with that in mind", 1),
    ("in this context", 1),
    ("in light of", 1),
    ("as we navigate", 1),
    ("when it comes to", 1),
    ("it goes without saying", 1),
    ("needless to say", 1),
    ("at the end of the day", 1),
    ("the bottom line", 1),
    ("key takeaway", 1),
    ("food for thought", 1),
    ("resonate", 1),
    ("aligns with", 1),
    ("bolster", 1),
    ("catalyst", 1),
    ("testament", 1),
    ("arguably", 1),
    ("notably", 1),
    ("specifically", 1),
    ("essentially", 1),
    ("fundamentally", 1),
    ("inherently", 1),
    ("intricacies", 1),
    # 新增 Tier 1（打磨扩充）
    ("unlock", 3),
    ("unlock the potential", 3),
    ("unlock opportunities", 3),
    ("navigate the complexities", 3),
    ("navigate the challenges", 3),
    ("embark on a journey", 3),
    ("a transformative experience", 3),
    ("a paradigm shift", 3),
    ("a new era of", 3),
    ("the dawn of", 3),
    # 新增 Tier 2（打磨扩充）
    ("elevate", 2),
    ("elevate the experience", 2),
    ("empower", 2),
    ("empower individuals", 2),
    ("streamline", 2),
    ("streamline processes", 2),
    ("optimize", 2),
    ("optimize performance", 2),
    ("foster", 2),
    ("foster collaboration", 2),
    ("foster innovation", 2),
    ("facilitate", 2),
    ("facilitate communication", 2),
    ("harness", 2),
    ("harness the power", 2),
    ("leverage", 2),
    ("leverage technology", 2),
    ("synergize", 2),
    ("create synergies", 2),
    ("drive", 2),
    ("drive growth", 2),
    ("drive innovation", 2),
    # 新增 Tier 3（打磨扩充）
    ("in today's world", 1),
    ("in the modern era", 1),
    ("as we move forward", 1),
    ("as we look ahead", 1),
    ("the future of", 1),
    ("the potential of", 1),
    ("the power of", 1),
    ("the importance of", 1),
    ("the value of", 1),
    ("the benefits of", 1),
    ("the challenges of", 1),
    ("the opportunities of", 1),
    ("it's no secret", 1),
    ("it's worth noting", 1),
    ("it's important to understand", 1),
    ("it's crucial to recognize", 1),
    ("it's essential to acknowledge", 1),
    # 新增中文 AI 语言标志
    ("值得注意的是", 3),
    ("需要指出的是", 3),
    ("至关重要的是", 3),
    ("不可或缺的是", 3),
    ("众所周知", 2),
    ("毋庸置疑", 2),
    ("不言而喻", 2),
    ("显而易见", 2),
    ("总而言之", 2),
    ("综上所述", 2),
    ("由此可见", 2),
    ("因此", 1),
    ("然而", 1),
    ("此外", 1),
    ("而且", 1),
    ("同时", 1),
    ("最后", 1),
    ("首先", 1),
    ("其次", 1),
    ("再次", 1),
]


def _detect_linguistic_markers(text: str) -> float:
    """检测 AI 语言标志（逆向自 sloptotal LinguisticEngine）"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)

    total_weight = 0
    for marker, weight in _AI_MARKERS_WEIGHTED:
        count = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
        total_weight += count * weight

    # Density: weighted hits per 1000 words
    density = (total_weight / word_count) * 1000
    # Score mapping: 0 density = 0.0, 15+ density = 1.0
    score = min(density / 15.0, 1.0)
    return round(score, 3)


# ============================================================
# sloptotal 逆向集成：Vocabulary 词汇丰富度检测
# ============================================================

def _detect_vocabulary_richness(text: str) -> float:
    """检测词汇丰富度（打磨版，逆向自 sloptotal VocabularyEngine）"""

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 20:
        return 0.0

    total = len(words)
    freq = Counter(words)
    unique = len(freq)

    # 1. Root TTR（词型丰富度）
    root_ttr = unique / math.sqrt(total)
    # 人类文本：root_ttr 通常 5-8+，AI 倾向 4-6
    ttr_score = max(0, 1.0 - ((root_ttr - 3.5) / 4.0))
    ttr_score = min(max(ttr_score, 0.0), 1.0)

    # 2. Hapax legomena ratio（只出现一次的词的比例）
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / unique if unique > 0 else 0
    # 人类文本：hapax ratio 通常 0.5-0.7，AI 文本：0.3-0.5
    hapax_score = max(0, 1.0 - ((hapax_ratio - 0.3) / 0.4))
    hapax_score = min(max(hapax_score, 0.0), 1.0)

    # 3. Yule's K（词汇多样性）
    freq_of_freq = Counter(freq.values())
    m1 = total
    m2 = sum(i * i * v for i, v in freq_of_freq.items())
    yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0
    # 人类文本：K ~ 80-120，AI 文本：K ~ 120-200+
    k_score = min(max((yules_k - 80) / 120, 0.0), 1.0)

    # 4. 平均词长（打磨新增）
    avg_word_length = sum(len(w) for w in words) / total
    # 人类文本：平均词长 4.5-5.5，AI 文本：5.0-6.0（AI 更喜欢用长词）
    length_score = min(max((avg_word_length - 4.5) / 2.0, 0.0), 1.0)

    # 5. 长词比例（>6个字母）
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / total
    # 人类文本：长词比例 10-20%，AI 文本：20-35%
    long_score = min(max((long_word_ratio - 0.10) / 0.20, 0.0), 1.0)

    # 6. 短词比例（<3个字母）
    short_words = sum(1 for w in words if len(w) < 3)
    short_word_ratio = short_words / total
    # 人类文本：短词比例 20-30%，AI 文本：15-25%（AI 更喜欢用中等长度的词）
    short_score = min(max((0.30 - short_word_ratio) / 0.15, 0.0), 1.0)

    final_score = ttr_score * 0.20 + hapax_score * 0.20 + k_score * 0.15 + length_score * 0.15 + long_score * 0.15 + short_score * 0.15
    return round(final_score, 3)


# ============================================================
# sloptotal 逆向集成：Structural 结构特征检测
# ============================================================

def _detect_structural_patterns(text: str) -> float:
    """检测结构特征（逆向自 sloptotal StructuralEngine）"""

    signals = []
    word_count = max(int(_word_count(text)), 1)

    # 1. Em-dash 频率
    em_dashes = len(re.findall(r"\w—\w", text))
    em_dash_rate = (em_dashes / word_count) * 1000
    em_score = min(em_dash_rate / 6.0, 1.0)

    # 2. 句长均匀度
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if len(sentences) >= 3:
        lengths = [int(_word_count(s)) for s in sentences]
        mean_len = statistics.mean(lengths)
        stdev = statistics.stdev(lengths)
        cv = stdev / mean_len if mean_len > 0 else 0
        uniformity_score = max(0, 1.0 - (cv / 0.6))
    else:
        uniformity_score = 0.0

    # 3. 列表密度
    lines = text.split("\n")
    list_lines = sum(
        1
        for l in lines
        if re.match(r"\s*[-•*]\s", l) or re.match(r"\s*\d+[.)]\s", l)
    )
    list_per_1k = (list_lines / word_count) * 1000
    list_score = min(list_per_1k / 30.0, 1.0)

    # 4. 段落长度均匀度
    paragraphs = [
        p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5
    ]
    if len(paragraphs) >= 3:
        para_lengths = [int(_word_count(p)) for p in paragraphs]
        para_mean = statistics.mean(para_lengths)
        para_stdev = statistics.stdev(para_lengths)
        para_cv = para_stdev / para_mean if para_mean > 0 else 0
        para_score = max(0, 1.0 - (para_cv / 0.5))
    else:
        para_score = 0.0

    # 5. 加粗/标题密度
    bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
    heading_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    format_density = ((bold_count + heading_count) / word_count) * 1000
    format_score = min(format_density / 10.0, 1.0)

    # 6. 冒号密度（最强 AI 信号）
    colon_count = text.count(":")
    colon_density = (colon_count / word_count) * 1000
    # AI: 18-39/1k，人类：0-17/1k
    colon_score = min(max(colon_density - 8, 0) / 15.0, 1.0)

    # 7. 冒号定义模式（"术语：解释"）
    colon_defs = len(
        re.findall(r"[A-Z][a-z]+(?:\s[A-Z]?[a-z]+){0,3}:\s[A-Z]", text)
    )
    colon_def_score = min(colon_defs / 3.0, 1.0)

    # 8. 括号使用频率（打磨新增）
    paren_count = len(re.findall(r"[()（）]", text))
    paren_density = (paren_count / word_count) * 1000
    # AI：10-20 per 1k，人类：3-8 per 1k
    paren_score = min(max((paren_density - 5) / 10.0, 0.0), 1.0)

    # 9. 引号使用频率（打磨新增）
    quote_count = len(re.findall(r"[''\"\"'']", text))
    quote_density = (quote_count / word_count) * 1000
    # AI：8-15 per 1k，人类：3-7 per 1k
    quote_score = min(max((quote_density - 4) / 8.0, 0.0), 1.0)

    # 10. 斜杠使用频率
    slash_count = len(re.findall(r"/", text))
    slash_density = (slash_count / word_count) * 1000
    # AI：5-10 per 1k，人类：1-3 per 1k
    slash_score = min(max((slash_density - 2) / 6.0, 0.0), 1.0)

    # 11. 分号使用频率
    semicolon_count = len(re.findall(r";", text))
    semicolon_density = (semicolon_count / word_count) * 1000
    # AI：3-8 per 1k，人类：0-2 per 1k
    semicolon_score = min(max((semicolon_density - 1) / 5.0, 0.0), 1.0)

    # 12. 破折号使用频率
    dash_count = len(re.findall(r"--|—", text))
    dash_density = (dash_count / word_count) * 1000
    # AI：2-5 per 1k，人类：0-1 per 1k
    dash_score = min(max((dash_density - 0.5) / 3.0, 0.0), 1.0)

    # 加权打分——冒号信号权重最高
    final_score = (
        em_score * 0.04
        + uniformity_score * 0.05
        + list_score * 0.06
        + para_score * 0.04
        + format_score * 0.03
        + colon_score * 0.15
        + colon_def_score * 0.15
        + paren_score * 0.08
        + quote_score * 0.08
        + slash_score * 0.05
        + semicolon_score * 0.15
        + dash_score * 0.12
    )
    return round(final_score, 3)


# ============================================================
# sloptotal 逆向集成：Sentiment & Hedging 情感与对冲检测
# ============================================================

_HEDGING_PHRASES = [
    # 原有词
    "it's worth noting",
    "it is worth noting",
    "it's important to remember",
    "it is important to remember",
    "it's important to note",
    "it is important to note",
    "it's crucial to",
    "it is crucial to",
    "it's essential to",
    "it is essential to",
    "it should be noted",
    "one might argue",
    "it could be argued",
    "some might say",
    "there are those who",
    "many experts believe",
    "studies suggest",
    "research indicates",
    "while there are",
    "although there are",
    "that said",
    "having said that",
    "that being said",
    "with that being said",
    "be that as it may",
    "regardless",
    "irrespective of",
    # 新增词
    "it's important to understand",
    "it is important to understand",
    "it's crucial to recognize",
    "it is crucial to recognize",
    "it's essential to acknowledge",
    "it is essential to acknowledge",
    "it should be mentioned",
    "it should be pointed out",
    "one could argue",
    "one may argue",
    "it can be argued",
    "some would say",
    "many would argue",
    "there are those who believe",
    "many experts suggest",
    "studies indicate",
    "research suggests",
    "while it is true that",
    "although it is true that",
    "having said this",
    "that being this",
    "with that being this",
    "be that as it may",
    "nevertheless",
    "nonetheless",
    "in any case",
    "at any rate",
]

_BALANCE_PATTERNS = [
    r"on (?:the )?one hand\b.{5,150}\bon the other(?: hand)?",
    r"while .{5,80}, (?:it |there |we |they )",
    r"although .{5,80}, (?:it |there |we |they )",
    r"(?:pros|advantages|benefits) .{5,200}(?:cons|disadvantages|drawbacks)",
]

_VAGUE_POSITIVITY = [
    # 原有词
    "incredibly important",
    "extremely valuable",
    "truly remarkable",
    "absolutely essential",
    "deeply meaningful",
    "truly transformative",
    "game-changing",
    "world-class",
    "best-in-class",
    "make a difference",
    "positive impact",
    "meaningful change",
    "better future",
    "brighter future",
    "greater good",
    "incredible potential",
    "exciting opportunity",
    "powerful tool",
    "valuable resource",
    "essential guide",
    # 新增词（打磨扩充）
    "a new era",
    "a new paradigm",
    "a game-changer",
    "a breakthrough",
    "a major milestone",
    "a significant step forward",
    "a key factor",
    "a crucial element",
    "a vital component",
    "a foundational pillar",
    "a cornerstone of",
    "a testament to",
    "a symbol of",
    "a representation of",
    "an embodiment of",
    "the future of",
    "the next generation",
    "the cutting edge",
    "the forefront of",
    "the leading edge",
    # 继续扩充
    "a powerful force",
    "a driving force",
    "a catalyst for change",
    "a catalyst for growth",
    "a catalyst for innovation",
    "a beacon of hope",
    "a beacon of light",
    "a ray of hope",
    "a breath of fresh air",
    "a world of possibilities",
    "a world of opportunities",
    "a world of potential",
    "a world of difference",
    "a positive impact",
    "a lasting impact",
    "a profound impact",
    "a significant impact",
    "a meaningful impact",
    "a transformative impact",
]

_FILLER_QUALIFIERS = [
    # 原有词
    "very",
    "really",
    "extremely",
    "incredibly",
    "absolutely",
    "definitely",
    "certainly",
    "undoubtedly",
    "undeniably",
    "quite",
    "rather",
    "fairly",
    "somewhat",
    "relatively",
    # 新增词（打磨扩充）
    "super",
    "ultra",
    "mega",
    "highly",
    "deeply",
    "profoundly",
    "tremendously",
    "immensely",
    "enormously",
    "massively",
    "significantly",
    "substantially",
    "considerably",
    "noticeably",
    "markedly",
]


def _detect_sentiment_hedging(text: str) -> float:
    """检测情感与对冲模式（逆向自 sloptotal SentimentEngine）"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)

    # 1. 过度对冲短语
    hedge_count = 0
    for phrase in _HEDGING_PHRASES:
        hedge_count += text_lower.count(phrase)
    hedge_density = (hedge_count / word_count) * 1000
    hedge_score = min(hedge_density / 4.0, 1.0)

    # 2. 强制平衡结构
    balance_count = 0
    for pattern in _BALANCE_PATTERNS:
        balance_count += len(re.findall(pattern, text_lower, re.DOTALL))
    balance_score = min(balance_count / 3.0, 1.0)

    # 3. 模糊积极表达
    vague_count = 0
    for phrase in _VAGUE_POSITIVITY:
        vague_count += text_lower.count(phrase)
    vague_density = (vague_count / word_count) * 1000
    vague_score = min(vague_density / 3.0, 1.0)

    # 4. 填充限定词滥用
    qualifier_count = 0
    for word in _FILLER_QUALIFIERS:
        qualifier_count += len(re.findall(r"\b" + word + r"\b", text_lower))
    qualifier_density = (qualifier_count / word_count) * 1000
    # 人类用 ~5-10 per 1k，AI ~15-25+
    qualifier_score = max(0, min((qualifier_density - 8) / 15.0, 1.0))

    # 5. 解释性 this/these 模式
    explanatory_verbs = len(
        re.findall(
            r"\bThis (?:means|ensures|allows|enables|helps|provides|creates|offers|makes|includes|involves|requires)\b",
            text,
        )
    )
    explanatory_nouns = len(
        re.findall(
            r"\b[Tt]his (?:approach|method|process|tool|feature|system|framework|strategy|solution)\b",
            text,
        )
    )
    explanatory_these = len(
        re.findall(
            r"\b[Tt]hese (?:tools|features|systems|methods|approaches|strategies|solutions|techniques|elements|factors|components|resources)\b",
            text,
        )
    )
    explanatory_count = explanatory_verbs + explanatory_nouns + explanatory_these
    # AI: 3-8 per article，人类：0-1
    explanatory_score = min(explanatory_count / 4.0, 1.0)

    # 加权打分
    final_score = (
        hedge_score * 0.30
        + balance_score * 0.15
        + vague_score * 0.20
        + qualifier_score * 0.15
        + explanatory_score * 0.20
    )
    return round(final_score, 3)


def _detect_human_imperfection(text: str) -> float:
    """
    人类写作痕迹检测器 v2（重写）
    返回：AI 概率（越高越像 AI）

    旧版只统计"打字错误"（双空格、小写 i、缺句末标点、中英标点混用）。
    这类特征在 Word 排版的规范文本里恒为 0，于是任何经过拼写检查的文本
    都会拿满分「完美 = AI」。实测旧版给人类样本 0.75、AI 样本 0.88，
    几乎没有区分力。

    新版改用语篇级信号，对"格式规范但确实是人所写"的文本同样有效：
      1. 句长变异系数   —— 人写起伏大，AI 均匀
      2. 过渡词密度     —— AI 偏爱 furthermore / moreover / therefore
      3. 实词重复率     —— 人写更容易反复用同一个词
      4. 模糊限定语密度 —— 人写爱用 very / maybe / I think / a lot
      5. 句首多样性     —— 人写常用 This / It / And / But 起句
    """
    sentences = [s for s in _split_sentences(text) if int(_word_count(s)) >= 3]
    if len(sentences) < 4:
        return 0.5

    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)
    lens = [int(_word_count(s)) for s in sentences]

    # 1) 句长变异系数
    mean_len = sum(lens) / len(lens)
    var = sum((l - mean_len) ** 2 for l in lens) / len(lens)
    cv = (var ** 0.5) / mean_len if mean_len else 0.0
    cv_ai = max(0.0, min(1.0, (0.70 - cv) / 0.45))

    # 2) 过渡词密度
    _trans = ("furthermore", "moreover", "additionally", "consequently",
              "therefore", "however", "in addition", "overall",
              "in conclusion", "firstly", "secondly", "thirdly", "finally", "thus")
    trans = sum(text.lower().count(w) for w in _trans)
    trans_density = trans / n_words * 1000
    trans_ai = max(0.0, min(1.0, (trans_density - 2.0) / 12.0))

    # 3) 实词重复率
    content = [w for w in words if len(w) > 3]
    cnt = Counter(content)
    rep = sum(c - 1 for c in cnt.values() if c > 2) / max(len(content), 1)
    rep_ai = max(0.0, min(1.0, (0.10 - rep) / 0.10))

    # 4) 模糊限定语密度
    _vague = (" very ", " really ", " quite ", " maybe ", " a lot ", " lots of ",
              "i think ", "i believe ", "in my opinion", " some ", " kind of ",
              " actually ", " of course ", " and so on ")
    tl = " " + text.lower() + " "
    vague = sum(tl.count(w) for w in _vague)
    vague_density = vague / n_words * 1000
    vague_ai = max(0.0, min(1.0, 1.0 - vague_density / 8.0))

    # 5) 句首多样性
    starters = {"this", "it", "and", "but", "so", "that", "these", "also", "then"}
    first_words = [s.split()[0].lower().strip(",.") for s in sentences if s.split()]
    lead_ratio = sum(1 for w in first_words if w in starters) / max(len(first_words), 1)
    lead_ai = max(0.0, min(1.0, (0.30 - lead_ratio) / 0.30))

    ai_prob = (cv_ai * 0.30 + trans_ai * 0.22 + rep_ai * 0.18
               + vague_ai * 0.20 + lead_ai * 0.10)
    return round(max(0.05, min(0.95, ai_prob)), 3)


# ============================================================
# 情感曲线检测器（Emotion Curve Detector）
# 创新角度：AI 情感很平，人类有起伏
# ============================================================

_POSITIVE_WORDS = [
    # 英文正面词（按强度分级，精雕细琢扩充）
    # 强正面
    "excellent", "amazing", "wonderful", "fantastic", "outstanding",
    "brilliant", "magnificent", "phenomenal", "terrific", "superb",
    "exceptional", "remarkable", "impressive", "delighted", "thrilled",
    "incredible", "awesome", "perfect", "extraordinary", "spectacular",
    # 中等正面
    "great", "good", "positive", "happy", "excited", "joyful",
    "pleased", "satisfied", "grateful", "thankful", "optimistic",
    "hopeful", "confident", "proud", "cheerful", "upbeat",
    # 弱正面
    "nice", "fine", "okay", "decent", "pleasant",
    "beneficial", "helpful", "useful", "valuable", "worthwhile",
    # 中文正面词
    "优秀", "出色", "很棒", "很好", "积极", "开心", "进步",
    "成功", "杰出", "卓越", "优秀", "出色", "棒", "好",
    "高兴", "快乐", "满意", "感谢", "乐观", "希望",
    "自信", "自豪", "愉快", "不错", "可以",
]

_NEGATIVE_WORDS = [
    # 英文负面词（按强度分级，精雕细琢扩充）
    # 强负面
    "terrible", "awful", "horrible", "dreadful", "appalling",
    "pathetic", "useless", "worthless", "miserable", "devastating",
    "disgusting", "repulsive", "offensive", "vile", "abysmal",
    # 中等负面
    "bad", "poor", "negative", "sad", "angry", "frustrated",
    "disappointed", "disappointing", "upset", "annoyed", "irritated",
    "worried", "anxious", "afraid", "scared", "fearful",
    # 弱负面
    "mediocre", "subpar", "inferior", "unsatisfactory", "unpleasant",
    "boring", "tedious", "dull", "monotonous", "tedious",
    # 中文负面词
    "差", "糟糕", "不好", "负面", "失望", "沮丧",
    "失败", "差劲", "劣质", "糟", "坏",
    "难过", "生气", "烦躁", "担心", "焦虑",
    "害怕", "恐惧", "一般", "普通", "无聊",
]

# 否定词（会反转情感，精雕细琢扩充）
_NEGATIONS = [
    # 英文否定词
    "not", "no", "never", "don't", "doesn't", "isn't", "aren't", "won't",
    "can't", "cannot", "couldn't", "wouldn't", "shouldn't",
    "haven't", "hasn't", "hadn't", "didn't", "wasn't", "weren't",
    # 中文否定词
    "不", "没", "无", "非", "未",
    "不是", "没有", "不会", "不能", "不要",
    "并非", "并非", "毫不", "绝不", "从不",
]


def _detect_paragraph_rhythm(text: str) -> float:
    """段落节奏检测。

    生成式模型的段落长度高度均匀（每段 4~5 句、长度接近），
    人类写作的段落长短悬殊——有的段落只有一句，有的写成一整屏。

    实测（中文 8 篇 AI / 英文 2 篇 AI / 41 篇人类）：
      AI   段落长度 CV 0.15 ~ 0.47
      人类          CV 0.39 ~ 2.51
    不依赖任何词表，换模型、换语言都还有效。
    """
    # 只统计「真正的正文段落」：按词数筛，不按字符数。
    # 原来用 len(...) > 40 字符，封皮行（"Marking Criteria"）、
    # 13 词的小标题都能通过，和 300 词的注释段混在一起，
    # 把段落长度方差人为拉大，同一篇文章换个排版就换一个分数。
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.split()) >= 40]
    # 段落数不足 5 时 CV 本身就不稳（样本少 → 极值少 → CV 天然偏小），
    # 且注释书目这类体裁按评分要求每段长度就必须接近，属于体裁特征而非 AI 特征。
    # 实测：三篇 Annotated Bibliography 的 CV 只有 0.02~0.06，全被拉满 100%。
    # 这种情况退回中性值，不参与判断。
    if len(paras) < 5:
        return 0.5
    lens = [_word_count(p) for p in paras]
    mean_len = sum(lens) / len(lens)
    if not mean_len:
        return 0.5
    cv = statistics.pstdev(lens) / mean_len
    return round(max(0.0, min(1.0, (0.55 - cv) / 0.40)), 3)


def _detect_emotion_curve(text: str) -> float:
    """
    情感曲线检测器
    原理：AI 写文章情感很平，人类写文章情感有起伏
    改进：扩充词表、考虑否定词、调整归一化
    """
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    # 计算每句话的情感分
    emotion_scores = []
    for sent in sentences:
        sent_lower = sent.lower()
        words = sent_lower.split()

        pos_count = 0
        neg_count = 0

        for i, word in enumerate(words):
            # 检查是否是否定词后面的词
            is_negated = i > 0 and words[i-1] in _NEGATIONS

            for w in _POSITIVE_WORDS:
                if w in word:
                    if is_negated:
                        neg_count += 1  # 否定词+正面词 = 负面
                    else:
                        pos_count += 1
                    break

            for w in _NEGATIVE_WORDS:
                if w in word:
                    if is_negated:
                        pos_count += 1  # 否定词+负面词 = 正面
                    else:
                        neg_count += 1
                    break

        # 情感分：-1 到 1
        if pos_count + neg_count > 0:
            emotion = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            emotion = 0.0
        emotion_scores.append(emotion)

    # 计算情感曲线的标准差（起伏程度）
    if len(emotion_scores) < 2:
        return 0.5

    mean_emotion = statistics.mean(emotion_scores)
    std_emotion = statistics.stdev(emotion_scores)

    # 标准差越小 = 情感越平 = 越像 AI
    # 标准差越大 = 情感起伏越大 = 越像人
    # 所以：std_emotion 越小 = AI 分数越高
    # 归一化（打磨过的参数）：
    # std=0.1 → 0.8（很平，像 AI）
    # std=0.3 → 0.4（一般）
    # std=0.5 → 0.0（起伏大，像人）
    ai_score = max(0.0, min(1.0, 1.0 - (std_emotion / 0.4)))
    return round(ai_score, 3)


def _detect_specific_details(text: str) -> float:
    """
    具体细节密度检测器 v2

    v1 失效原因：用 \b[A-Z][a-z]+ [A-Z][a-z]+\b 抓人名，把句首大写 + 下个词
    大写（"The Authors" / "This Study" / "In This"）全算成专有名词。
    一份 700 词的学术稿匹配出 382 例，密度 545/1k，是阈值的 27 倍，
    于是 1 - density/20 恒为负，所有文本都被压成 0.0。

    v2 三条更干净的线索：
      1. 数字密度（含百分比、年份、样本量）—— AI 偏抽象，少给具体数值
      2. 非句首专名 —— 跳过句首可排除语法性大写
      3. 引号内的具体表述 —— 人写更常引用他人原话/具体说法
    """
    sentences = [s for s in _split_sentences(text) if s.strip()]
    n_words = max(int(_word_count(text)), 1)

    # 1) 数字（含 148 / 87% / 2025 这类具体量）
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))

    # 2) 非句首专名
    proper = 0
    for s in sentences:
        tokens = s.split()
        for i, w in enumerate(tokens):
            if i == 0:
                continue                      # 句首大写属于语法，不算专名
            core = w.strip(".,;:()\"'")
            if re.match(r"^[A-Z][a-z]{2,}$", core):
                proper += 1

    # 3) 引号内的具体表述
    quoted = len(re.findall(r"[\"“][^\"”]{8,80}[\"”]", text))

    density = (numbers * 3 + proper * 2 + quoted * 3) / n_words * 1000
    # 人写学术：25~60 per 1k；AI：5~15 per 1k
    return round(max(0.0, min(1.0, 1.0 - density / 45.0)), 3)


def _detect_self_references(text: str) -> float:
    """
    自我指涉检测器 v2

    v1 失效原因：只统计 "I think / in my opinion" 的绝对密度。
    在学术文本里，无论人写还是 AI 写，这类词都很少；而且 AI 一旦生成
    "In my opinion, ..." 这种模板句，密度立刻冲高 —— 实测 AI 样本
    反而拿到 1.0（被判更像 AI），而人写的学术注释同样接近 1.0。
    这个信号在学术语域里不成立。

    v2 改成两级：
      强自反 —— "I do not understand" / "I found that" / "I was surprised"
                这类带具体认知过程的表述，AI 几乎不会产出；
      弱模板 —— "I think" / "in my opinion"，人机通用，只给少量权重。
    """
    n_words = max(int(_word_count(text)), 1)
    low = " " + text.lower() + " "

    def _hit(phrases):
        c = 0
        for p in phrases:
            c += len(re.findall(r"(?<![a-z])" + re.escape(p), low))
        return c

    strong = _hit(_SELF_REF_STRONG)
    weak = _hit(_SELF_REF_WEAK)

    strong_density = strong / n_words * 1000
    weak_density = weak / n_words * 1000

    human_signal = (min(strong_density / 6.0, 1.0) * 0.8
                    + min(weak_density / 12.0, 1.0) * 0.2)
    return round(max(0.0, min(1.0, 1.0 - human_signal)), 3)


# v2 词表：强自反指具体认知行为，弱模板指通用套语
_SELF_REF_STRONG = [
    "i do not understand", "i don't understand", "i did not understand",
    "i found that", "i found it", "i noticed", "i realised", "i realized",
    "i was surprised", "i was confused", "i am not sure", "i'm not sure",
    "in my own experience", "from my own experience", "i remember",
    "i have seen", "i could not", "i couldn't", "i tried", "i learned that",
    "i learnt that", "to be honest", "i am not clear", "i'm not clear",
]
_SELF_REF_WEAK = [
    "i think", "i believe", "i feel", "in my opinion", "in my view",
    "from my perspective", "personally", "i would say", "as far as i",
]


# ============================================================
# 逻辑连接词密度检测器（Logical Connectors Detector）
# 创新角度：AI 用很多 therefore/however，人类用得少
# 打磨版：扩充词表、分类检测
# ============================================================

# v2：只保留真正标志学术语域的低频/多词连接词。
# but / also / while / yet / though / although 属于基础功能词，
# 任何文本都高频，纳入词表会让引擎恒为满分。
_ACADEMIC_CONNECTORS = [
    "therefore", "consequently", "thus", "hence", "accordingly",
    "furthermore", "moreover", "additionally", "in addition",
    "nevertheless", "nonetheless", "however",
    "in contrast", "on the other hand", "conversely",
    "as a result", "as a consequence", "for this reason",
    "in conclusion", "in summary", "overall", "to sum up",
    "further", "notably", "specifically", "in particular",
]


def _detect_logical_connectors(text: str) -> float:
    """
    逻辑连接词检测器 v2

    v1 失效原因：词表把 but / also / while / yet / though / although 这类
    极高词频的功能词也算作"逻辑连接词"。任何自然英语文本里它们的密度都
    远超阈值，于是 min(density/15, 1.0) 恒被顶到 1.0 —— 实测所有样本
    都是 100%，等于没测。

    另外 v1 用 str.count 做子串匹配，"but" 会把 "attribute" 里的 but
    也数进去。

    v2 只保留真正区分学术语域的多词/低频连接词，并加词边界匹配。
    """
    n_words = max(int(_word_count(text)), 1)
    low = " " + text.lower() + " "

    count = 0
    for phrase in _ACADEMIC_CONNECTORS:
        # (?<![a-z]) ... (?![a-z]) 保证整词匹配
        count += len(re.findall(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low))

    density = count / n_words * 1000
    # 人写学术：2~6 per 1k；AI：8~20 per 1k
    return round(max(0.0, min(1.0, (density - 2.0) / 12.0)), 3)


# ============================================================
# RoBERTa ChatGPT 检测器（逆向自 sloptotal ClassifierChatGPTEngine）
# ============================================================

_roberta_model = None
_roberta_tokenizer = None
_roberta_ai_idx = None


_roberta_failed = False


def _hf_reachable(host: str = "huggingface.co", timeout: float = 0.6) -> bool:
    """0.6 秒 TCP 探测。

    huggingface_hub 的连接重试不读 HF_HUB_DOWNLOAD_TIMEOUT，
    离线时会按 1→2→4→8→16s 退避 5 次，单次检测多等两分钟。
    这里先做一次快速连通性判断，不通就直接放弃该引擎。
    """
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def _load_roberta_model():
    """懒加载 RoBERTa ChatGPT 检测器

    v2：失败后写入 _roberta_failed，后续调用直接返回，不再重复联网重试，
        也不再向 stderr 打印长篇警告（设 AURORA_VERBOSE=1 可恢复）。
    """
    global _roberta_model, _roberta_tokenizer, _roberta_ai_idx, _roberta_failed
    if _roberta_failed:
        return None, None, None
    if _roberta_model is None:
        if not _hf_reachable():
            _roberta_failed = True
            print("[aurora] 连不上 huggingface.co，跳过 roberta-chatgpt 引擎"
                  "（其余统计引擎照常工作，不影响结论方向）", file=sys.stderr)
            return None, None, None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
            _roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _roberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _roberta_model.eval()

            # 自动检测哪个标签是 AI
            id2label = _roberta_model.config.id2label
            for idx, label in id2label.items():
                label_lower = str(label).lower()
                if "chatgpt" in label_lower or "ai" in label_lower or "fake" in label_lower:
                    _roberta_ai_idx = int(idx)
                    break
            if _roberta_ai_idx is None:
                _roberta_ai_idx = max(int(k) for k in id2label.keys())
        except Exception as e:
            _roberta_failed = True
            if _VERBOSE:
                print(f"[aurora] RoBERTa 模型不可用，已跳过该引擎：{e}", file=sys.stderr)
            else:
                print("[aurora] RoBERTa 模型不可用（离线），已跳过 roberta-chatgpt 引擎。"
                      "设 AURORA_VERBOSE=1 查看详情。", file=sys.stderr)
            return None, None, None
    return _roberta_model, _roberta_tokenizer, _roberta_ai_idx


# ============================================================
# 隐藏字符检测器（Hidden Characters Detector）
# 创新角度：AI生成的文本可能会有水印（零宽字符、不可见字符）
# 原理：检测文本里有没有隐藏的、不可见的字符
# ============================================================

# 常见的隐藏字符（AI水印，精雕细琢扩充）
_HIDDEN_CHARACTERS = [
    # 零宽字符
    "\u200b",  # 零宽空格
    "\u200c",  # 零宽非连接符
    "\u200d",  # 零宽连接符
    "\u2060",  # 字连接符
    "\ufeff",  # 零宽非断空格
    # 软连字符
    "\u00ad",  # 软连字符
    # 其他不可见字符
    "\u200e",  # 从左到右标记
    "\u200f",  # 从右到左标记
    "\u202a",  # 从左到右嵌入
    "\u202b",  # 从右到左嵌入
    "\u202c",  # 弹出方向格式
    "\u202d",  # 从左到右覆盖
    "\u202e",  # 从右到左覆盖
    "\u2066",  # 从左到右隔离
    "\u2067",  # 从右到左隔离
    "\u2068",  # 第一类强隔离
    "\u2069",  # 弹出隔离
    # 其他隐藏字符
    "\u034f",  # 组合图形连接符
    "\u0483",  # 组合变音符号
    "\u0485",  # 组合西里尔变音符号
]


def _detect_hidden_characters(text: str) -> float:
    """
    隐藏字符检测器 - 全新角度！
    原理：AI生成的文本可能会有水印（零宽字符、不可见字符）
    人类写的文本不会有这些隐藏字符
    所以：隐藏字符越多 = 越像 AI
    """
    word_count = max(int(_word_count(text)), 1)

    # 统计隐藏字符数量
    hidden_count = 0
    for char in _HIDDEN_CHARACTERS:
        hidden_count += text.count(char)

    # 隐藏字符密度
    hidden_density = (hidden_count / word_count) * 1000

    # 人类：0 per 1k，AI：1-5 per 1k（如果有水印的话）
    # 所以：隐藏字符越多 = 越像 AI
    ai_score = min(hidden_density / 3.0, 1.0)
    return round(ai_score, 3)


# ============================================================
# 句首词分布检测器（Sentence Starter Distribution）
# 全新角度！人类写作的句首词分布很随机，AI写作的句首词分布很规律
# ============================================================

# AI 常用的句首词
_AI_SENTENCE_STARTERS = [
    # 过渡词开头
    "first", "second", "third", "finally", "lastly",
    "however", "therefore", "moreover", "furthermore", "additionally",
    "in addition", "in contrast", "in conclusion", "in summary",
    "on the other hand", "on one hand",
    "as a result", "consequently", "thus", "hence",
    "nevertheless", "nonetheless", "meanwhile",
    # 公式化开头
    "it is", "it's", "there are", "there is",
    "this is", "this demonstrates", "this shows",
    "in today's", "in the world", "in the realm",
    "when it comes to", "let's", "let us",
    # 其他AI常用开头
    "one of", "another", "some", "many", "most",
    "studies show", "research indicates", "experts believe",
]


def _detect_starter_patterns(text: str) -> float:
    """
    开头模式检测器（合并句首词分布 + 段落开头模式）
    原理：人类写作的句首词分布很随机，AI写作的句首词分布很规律
    """
    # 句子开头检测
    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        starters = []
        for sent in sentences:
            words = sent.strip().split()
            if words:
                first_word = words[0].lower().strip(".,!?;:'\"()")
                starters.append(first_word)

        if starters:
            ai_starter_count = 0
            for starter in starters:
                for ai_starter in _AI_SENTENCE_STARTERS:
                    if starter.startswith(ai_starter) or starter == ai_starter:
                        ai_starter_count += 1
                        break

            ai_starter_ratio = ai_starter_count / len(starters)
            starter_score = min(max((ai_starter_ratio - 0.15) / 0.30, 0.0), 1.0)
        else:
            starter_score = 0.5
    else:
        starter_score = 0.5

    # 段落开头检测
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5]
    if len(paragraphs) >= 3:
        opener_count = 0
        for para in paragraphs:
            words = para.split()
            if words:
                first_word = words[0].lower().strip(".,!?;:'\"()")
                for opener in _AI_SENTENCE_STARTERS:
                    if first_word.startswith(opener) or first_word == opener:
                        opener_count += 1
                        break

        opener_ratio = opener_count / len(paragraphs)
        paragraph_score = min(max((opener_ratio - 0.25) / 0.40, 0.0), 1.0)
    else:
        paragraph_score = 0.5

    # 加权平均
    final_score = starter_score * 0.6 + paragraph_score * 0.4
    return round(final_score, 3)


# ============================================================
# 标点符号使用模式检测器（Punctuation Patterns）
# 全新角度！人类和AI的标点使用习惯不同
# ============================================================

def _detect_punctuation_patterns(text: str) -> float:
    """标点使用模式检测器（合并感叹句/疑问句 + 标点符号使用）"""
    word_count = max(int(_word_count(text)), 1)

    # 统计感叹号和问号的密度
    exclamation_density = (text.count("!") / word_count) * 1000
    question_density = (text.count("?") / word_count) * 1000

    # 人类：感叹号和问号多，AI：感叹号和问号少
    exclamation_score = min(max((5.0 - exclamation_density) / 5.0, 0.0), 1.0)
    question_score = min(max((8.0 - question_density) / 8.0, 0.0), 1.0)

    # 统计感叹句和疑问句的比例
    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        exclamation_count = sum(1 for s in sentences if "!" in s)
        question_count = sum(1 for s in sentences if "?" in s)
        ratio = (exclamation_count + question_count) / len(sentences)
        sentence_score = min(max((0.10 - ratio) / 0.08, 0.0), 1.0)
    else:
        sentence_score = 0.5

    final_score = exclamation_score * 0.3 + question_score * 0.3 + sentence_score * 0.4
    return round(final_score, 3)


# ============================================================
# 长句/短句交替模式检测器（Long/Short Sentence Alternation）
# 全新角度！人类写作有长短句交替
# ============================================================

def _detect_sentence_length_distribution(text: str) -> float:
    """句子长度分布检测器（合并长句/短句交替 + 滑动窗口）"""
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    # 统计每个句子的长度
    lengths = [int(_word_count(s)) for s in sentences]

    # 统计相邻句子的长度差（长句/短句交替）
    diffs = []
    for i in range(len(lengths) - 1):
        diffs.append(abs(lengths[i] - lengths[i+1]))
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    alternation_score = min(max((7.0 - avg_diff) / 5.0, 0.0), 1.0)

    # 统计句子长度的变异系数（滑动窗口）
    mean_len = statistics.mean(lengths)
    stdev = statistics.stdev(lengths)
    cv = stdev / mean_len if mean_len > 0 else 0
    uniformity_score = max(0, 1.0 - (cv / 0.6))

    final_score = alternation_score * 0.5 + uniformity_score * 0.5
    return round(final_score, 3)


# ============================================================
# PAWN 困惑度注意力加权检测器
# 参考 PAWN 方法：对困惑度进行注意力加权，低困惑度token权重更高
# ============================================================

def _detect_pawn_perplexity(text: str) -> float:
    """PAWN 困惑度注意力加权检测器"""

    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    # 计算词频
    word_freq = Counter(words)
    total_words = len(words)

    # 计算每个词的log概率（简化版）
    log_probs = []
    for word in words:
        freq = word_freq.get(word, 0)
        # 加平滑
        prob = (freq + 0.5) / (total_words + 0.5 * len(word_freq))
        log_prob = math.log(prob)
        log_probs.append(log_prob)

    if not log_probs:
        return 0.5

    # PAWN：对低困惑度（高log概率）的token赋予更高权重
    # 也就是说，那些"太常见"的词对AI检测的贡献更大
    avg_log_prob = sum(log_probs) / len(log_probs)

    # 注意力权重：每个token的权重 = exp(log_prob) / sum(exp(log_prob))
    # 简化版：直接用log_prob作为权重
    weights = [max(0, lp - avg_log_prob) for lp in log_probs]  # 只取高于平均的部分
    total_weight = sum(weights)

    if total_weight == 0:
        return 0.5

    # 加权平均困惑度
    weighted_avg = sum(lp * w for lp, w in zip(log_probs, weights)) / total_weight

    # 归一化：加权困惑度越高（越负）= 越像人；加权困惑度越低（越接近0）= 越像AI
    ai_score = max(0.0, min(1.0, (-2.0 - weighted_avg) / 3.0))

    return round(ai_score, 3)


# ============================================================
# 对抗扰动程度检测器（Adversarial Perturbation Detector）
# 参考"Modeling the Attack"思路：计算文本距离"原始AI版本"的扰动程度
# 扰动越小、越"顺滑"，AI嫌疑反而越高
# ============================================================

def _detect_adversarial_perturbation(text: str) -> float:
    """对抗扰动程度检测器"""

    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    # 1. 检测词汇多样性（扰动会增加词汇多样性）
    word_freq = Counter(words)
    unique_ratio = len(word_freq) / len(words)

    # 2. 检测句子长度方差（扰动会让句子更随机）
    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        sent_lengths = [int(_word_count(s)) for s in sentences]
        mean_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        cv = math.sqrt(variance) / mean_len if mean_len > 0 else 0
    else:
        cv = 0.5

    # 3. 检测标点符号多样性（扰动会增加标点多样性）
    punct_types = set()
    for char in text:
        if char in ".,!?;:()[]{}\"'":
            punct_types.add(char)
    punct_diversity = len(punct_types) / 10.0  # 最多10种标点

    # 综合扰动程度：多样性越高 = 扰动越大 = AI嫌疑越低
    # 扰动程度 = 词汇多样性 * 0.4 + 句子长度方差 * 0.3 + 标点多样性 * 0.3
    perturbation = (unique_ratio * 0.4 + cv * 0.3 + punct_diversity * 0.3)

    # 归一化：扰动程度越高 = 越像人 = AI嫌疑越低
    # 扰动程度 0.2 → AI嫌疑 0.8（没扰动，很顺滑）
    # 扰动程度 0.6 → AI嫌疑 0.2（扰动很大，像人写的）
    ai_score = max(0.0, min(1.0, (0.5 - perturbation) / 0.3))

    return round(ai_score, 3)


# ============================================================
# 语言检测器（Language Detector）
# 自动判断文本是中文还是英文
# ============================================================

def _detect_language(text: str) -> str:
    """检测文本语言"""
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字符数量
    english_chars = len(re.findall(r'[a-zA-Z]', text))

    if chinese_chars > english_chars:
        return "zh"
    else:
        return "en"


# ============================================================
# 中文AI检测器（简化版）
# 针对中文文本的AI特征检测
# ============================================================

# 中文AI常用短语
_AI_CHINESE_MARKERS = [
    "值得注意的是",
    "需要指出的是",
    "至关重要的是",
    "综上所述",
    "总而言之",
    "由此可见",
    "众所周知",
    "毋庸置疑",
    "不言而喻",
    "显而易见",
    "首先",
    "其次",
    "再次",
    "最后",
    "此外",
    "而且",
    "同时",
    "因此",
    "然而",
]


_ZH_TEMPLATE = [
    "首先", "其次", "再次", "最后", "综上所述", "总而言之", "值得注意的是",
    "需要指出的是", "与此同时", "此外", "一方面", "另一方面", "由此可见",
    "总的来说", "简而言之", "不难发现", "众所周知", "毫无疑问", "进一步说",
    "在此基础上", "从某种意义上说",
]
# 公文/官腔词：中文 AI 特别喜欢，人类口语写作极少出现
_ZH_JARGON = [
    "赋能", "抓手", "闭环", "生态", "维度", "层面", "机制", "体系", "格局", "路径",
    "举措", "支撑", "驱动", "融合", "协同", "优化", "提升", "构建", "助力", "纵深",
    "引领", "夯实", "着力", "系统性", "整体性", "协同性",
    # 空形容词：本语料人类样本零出现（R→∞）
    "显著提升", "大幅改善", "有效优化", "深远影响", "有力支撑", "重要保障",
    "卓越成效", "全面提升",
]
# 万能开头 / 套话收尾：人类样本几乎不出现（R→∞ / R=108）
_ZH_BOILERPLATE = [
    "随着", "在当今", "近年来", "众所周知", "综上所述", "总而言之",
    "总的来说", "一言以蔽之", "由此可见", "需要指出的是",
]
# 翻译腔：从英文硬翻的句式（R≈1.7）
_ZH_TRANSLATIONESE = [
    "对于", "这意味着", "而言", "当", "基于",
]
# 口语标记：人写中文几乎必现
_ZH_COLLOQUIAL = [
    "我觉得", "我感觉", "其实", "挺", "特别", "真的", "咱们", "咋", "吧", "呢",
    "啊", "嘛", "搞", "啥", "挺", "说白了", "反正", "估计", "好像", "有点",
]


_PPL_MODEL = None
_PPL_TOKENIZER = None
_PPL_TRIED = False


def _find_local_model():
    """按优先级找本地模型目录。"""
    cands = [
        os.environ.get("AURORA_PPL_MODEL", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qwen2.5-0.5b"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "config.json")):
            return c
    return ""


def _load_ppl_model():
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_TRIED
    if _PPL_TRIED:
        return _PPL_MODEL, _PPL_TOKENIZER
    _PPL_TRIED = True
    path = _find_local_model()
    if not path:
        return None, None
    try:
        import warnings, logging
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _PPL_TOKENIZER = AutoTokenizer.from_pretrained(path)
        _PPL_MODEL = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
        _PPL_MODEL.eval()
        if _VERBOSE:
            print(f"[aurora] 困惑度模型已加载：{path}", file=sys.stderr)
    except Exception as e:
        if _VERBOSE:
            print(f"[aurora] 模型加载失败：{e}", file=sys.stderr)
        _PPL_MODEL = None
        _PPL_TOKENIZER = None
    return _PPL_MODEL, _PPL_TOKENIZER


def _detect_lm_perplexity(text: str):
    """语言模型困惑度 → AI 概率。

    这是所有信号里对"改写"最鲁棒的一项：它不看用了什么词、怎么断句，
    只看整段文字对语言模型来说有多可预测。改写能骗过表面特征，
    但骗不过"这个词后面接什么"的概率分布。

    实测（Qwen2.5-0.5B，500 token 截断）：
      标准 AI 输出   困惑度  6.3 ~ 13.0
      伪装成人类的 AI        14.4 ~ 59.7
      真实人类               22.9 ~ 116.5

    没有本地模型时返回 None，该引擎自动不参与。
    """
    # 当前挂载的是中文模型（Qwen2.5），对英文的困惑度不可靠：
    # 实测英文人类文本同样被判"很顺"，等于给所有人加分。
    # 英文侧仍走原有统计信号；将来接入英文模型后可放开。
    if _detect_language(text) != "zh":
        return None

    model, tok = _load_ppl_model()
    if model is None:
        return None
    try:
        import torch
        ids = tok(text[:1200], return_tensors="pt", truncation=True, max_length=500)["input_ids"]
        if ids.shape[1] < 30:
            return None
        with torch.no_grad():
            ppl = float(torch.exp(model(input_ids=ids, labels=ids).loss))
        if ppl <= 0:
            return None
        # 单边证据：只有"太顺"才构成证据。
        # 困惑度高只能说明这段不像模型写的，不能反证是人写的
        # —— 人写得好一样很顺。因此超过阈值直接不参与，返回 None。
        if ppl > 25.0:
            return None
        score = (math.log(25) - math.log(ppl)) / (math.log(25) - math.log(6.5))
        return round(max(0.0, min(1.0, score)), 3)
    except Exception:
        return None


def _detect_zh_char_entropy(text: str) -> float:
    """中文字符级 3-gram 熵。

    目前唯一"难以通过改写绕过"的中文信号：它衡量整篇用字的可预测程度，
    加几句口语、编几个数字都改变不了整体分布。

    实测（8 篇标准 AI / 4 篇伪装成人类的 AI / 24 篇真实人类）：
      AI        熵 8.38 ~ 8.66
      人类      熵 10.54
    人类用字更分散，模型更依赖高频搭配。
    """
    if _detect_language(text) != "zh":
        return 0.5
    han = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if len(han) < 300:
        return 0.5
    n = 3
    grams = [han[i:i + n] for i in range(len(han) - n + 1)]
    total = len(grams)
    counter = Counter(grams)
    entropy = -sum((v / total) * math.log2(v / total) for v in counter.values())
    return round(max(0.0, min(1.0, (11.5 - entropy) / 3.5)), 3)


def _detect_chinese_ai(text: str) -> float:
    """中文 AI 文本检测。

    英文那套词表（delve / furthermore / testament to）对中文完全无效，
    原实现只匹配十余个短语，权重也是 0，实测中文 AI 与人类样本区间完全重叠
    （AI 均值 0.241 / 人类 0.143）。

    这里改用六组中文特有信号，权重按实测区分力（Cohen's d）分配：
      模板连接词  +      官腔抽象词  +
      口语标记    -      具体数字    -
      句长变异    -      引号使用    -
    """
    if _detect_language(text) != "zh":
        return 0.5

    n = max(_word_count(text), 1)
    sentences = [x for x in _split_sentences(text) if len(x) >= 8]
    if len(sentences) < 4:
        return 0.5

    lens = [_word_count(x) for x in sentences]
    mean_len = sum(lens) / len(lens)
    cv = statistics.pstdev(lens) / mean_len if mean_len else 0

    def density(words):
        return sum(text.count(w) for w in words) / n * 1000

    s_tpl = min(density(_ZH_TEMPLATE) / 8.0, 1.0)
    s_jargon = min(density(_ZH_JARGON) / 10.0, 1.0)
    # 万能开头与套话收尾：AI 密度极高，人类几乎为零
    s_boiler = min(density(_ZH_BOILERPLATE) / 6.0, 1.0)
    s_colloq = max(0.0, 1.0 - density(_ZH_COLLOQUIAL) / 4.0)
    s_number = max(0.0, 1.0 - (len(re.findall(r"\d", text)) / n * 1000) / 8.0)
    s_cv = max(0.0, min(1.0, (0.62 - cv) / 0.35))
    s_quote = max(0.0, 1.0 - (len(re.findall(r"[“”「」『』]", text)) / n * 1000) / 6.0)

    # 权重按"可伪造性"重新分配：
    #   数字 / 口语 / 引号  是显式特征，写手用提示词加几句就绕过了；
    #   句长节奏          需要逐句控制长度，很难长期伪装；
    #   模板词 / 官腔     属于习惯性输出，中等可伪造。
    score = (s_tpl * 0.12 + s_jargon * 0.14 + s_boiler * 0.10
             + s_colloq * 0.12 + s_number * 0.10 + s_cv * 0.30 + s_quote * 0.12)
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
# 统计水印检测器（简化版）
# 检测统计水印（green/red-list）
# ============================================================

def _detect_statistical_watermark(text: str) -> float:
    """统计水印检测器（简化版）"""
    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    # 简化版统计水印检测：
    # 检测是否有异常的token分布（绿色token比例过高）
    word_freq = Counter(words)

    # 找出出现次数最多的词（绿色候选）
    top_words = word_freq.most_common(5)
    top_freq = sum(count for word, count in top_words) / len(words)

    # 绿色token比例异常高 = 可能有统计水印
    # 人类：top 5 词频率 ~10-15%，AI：~15-25%
    watermark_score = min(max((top_freq - 0.12) / 0.15, 0.0), 1.0)

    return round(watermark_score, 3)


# ============================================================
# SpecDetect 信号处理检测器（简化版）
# 对困惑度序列做频域分析，捕捉AI文本在频谱上的异常模式
# ============================================================

def _detect_spec_detect(text: str) -> float:
    """SpecDetect 信号处理检测器（简化版）"""

    words = text.lower().split()
    if len(words) < 30:
        return 0.5

    # 计算每个词的log概率（困惑度序列）
    word_freq = Counter(words)
    total_words = len(words)

    log_probs = []
    for word in words:
        freq = word_freq.get(word, 0)
        prob = (freq + 0.5) / (total_words + 0.5 * len(word_freq))
        log_probs.append(math.log(prob))

    # 简化版频域分析：计算序列的方差和自相关性
    # AI文本的困惑度序列更平滑（方差小，自相关性高）
    # 人类文本的困惑度序列更随机（方差大，自相关性低）

    # 1. 计算方差
    mean = sum(log_probs) / len(log_probs)
    variance = sum((lp - mean) ** 2 for lp in log_probs) / len(log_probs)

    # 2. 计算自相关性（相邻词的相关性）
    autocorr = 0
    for i in range(len(log_probs) - 1):
        autocorr += (log_probs[i] - mean) * (log_probs[i+1] - mean)
    autocorr = autocorr / (variance * len(log_probs)) if variance > 0 else 0

    # 归一化：方差越小、自相关性越高 = 越像AI
    # 人类：方差 ~2-4，AI：方差 ~1-2
    variance_score = min(max((3.0 - variance) / 2.0, 0.0), 1.0)

    # 人类：自相关性 ~0.1-0.3，AI：自相关性 ~0.3-0.5
    autocorr_score = min(max((autocorr - 0.2) / 0.3, 0.0), 1.0)

    final_score = variance_score * 0.5 + autocorr_score * 0.5
    return round(final_score, 3)


# ============================================================
# DNA-DetectLLM（简化版）
# 把检测看作"修复"过程：计算文本与"理想AI序列"的距离
# ============================================================

def _detect_dna_detect(text: str) -> float:
    """DNA-DetectLLM 简化版"""

    words = text.lower().split()
    if len(words) < 30:
        return 0.5

    # 计算词频
    word_freq = Counter(words)
    total_words = len(words)

    # 构造"理想AI序列"：最常见的词排列
    # 理想AI序列的特点：词频分布更平滑，没有罕见词
    # 计算修复代价：每个词与理想分布的偏离程度

    repair_cost = 0
    for word in words:
        freq = word_freq.get(word, 0)
        # 理想概率：平滑分布
        ideal_prob = 1.0 / len(word_freq)
        # 实际概率
        actual_prob = freq / total_words
        # 修复代价：实际概率与理想概率的差异
        repair_cost += abs(actual_prob - ideal_prob)

    # 平均修复代价
    avg_repair_cost = repair_cost / len(words)

    # 归一化：修复代价越低 = 越像AI（越接近理想分布）
    # 人类：修复代价 ~0.02-0.04，AI：修复代价 ~0.01-0.02
    ai_score = max(0.0, min(1.0, (0.03 - avg_repair_cost) / 0.02))

    return round(ai_score, 3)


# ============================================================
# 重复词检测器（Repeated Words Detector）
# 全新角度！AI写作会重复使用某些词
# ============================================================

def _detect_vocabulary_diversity(text: str) -> float:
    """词汇多样性检测器（合并重复词 + 词汇丰富度）"""
    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    word_freq = Counter(words)

    # 重复词检测
    top_words = word_freq.most_common(10)
    top_freq = sum(count for word, count in top_words) / len(words)
    repeated_score = min(max((top_freq - 0.15) / 0.15, 0.0), 1.0)

    # 词汇丰富度检测
    unique = len(word_freq)
    total = len(words)
    root_ttr = unique / math.sqrt(total)
    ttr_score = max(0, 1.0 - ((root_ttr - 3.5) / 4.0))
    ttr_score = min(max(ttr_score, 0.0), 1.0)

    final_score = repeated_score * 0.4 + ttr_score * 0.6
    return round(final_score, 3)


def _detect_with_roberta(text: str) -> Optional[float]:
    """用 RoBERTa 模型检测 AI 概率"""
    model, tokenizer, ai_idx = _load_roberta_model()
    if model is None:
        return None

    import torch

    # 分块处理（每块 512 token）
    chunks = []
    words = text.split()
    chunk_size = 80  # 约 80 词一块
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 10:
            chunks.append(chunk)

    if not chunks:
        return None

    scores = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        scores.append(probs[ai_idx].item())

    return sum(scores) / len(scores) if scores else None


# ============================================================
# 本地文档相似度比对
# ============================================================

def _compute_text_similarity(text1: str, text2: str) -> float:
    """计算两段文本的相似度（基于 n-gram 重叠）"""
    def get_ngrams(text, n=3):
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i:i+n]))
        return ngrams

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1.intersection(ngrams2)
    # Jaccard 相似度
    similarity = len(intersection) / len(ngrams1.union(ngrams2))
    return similarity


def compare_with_reference(doc_path: str, ref_path: str) -> dict:
    """比对当前文档和参考文档的相似度"""
    doc_text = extract_text_from_file(doc_path)
    ref_text = extract_text_from_file(ref_path)

    # 提取正文（排除引用）
    doc_prose, _ = _extract_prose_only(doc_text)
    ref_prose, _ = _extract_prose_only(ref_text)

    # 计算整体相似度
    overall_similarity = _compute_text_similarity(doc_prose, ref_prose)

    # 找出高重叠的句子
    doc_sentences = _split_sentences(doc_prose)
    ref_sentences = _split_sentences(ref_prose)

    overlapping_sentences = []
    for doc_sent in doc_sentences:
        for ref_sent in ref_sentences:
            sim = _compute_text_similarity(doc_sent, ref_sent)
            if sim > 0.5:
                overlapping_sentences.append({
                    "doc_sentence": doc_sent,
                    "ref_sentence": ref_sent,
                    "similarity": sim,
                })
                break  # 每个文档句子只匹配一次

    return {
        "overall_similarity": overall_similarity,
        "overlapping_count": len(overlapping_sentences),
        "overlapping_sentences": overlapping_sentences[:10],
    }


def _word_count(text: str) -> float:
    """估算词数，兼容中英文。

    英文按空格分词；中文没有空格，`text.split()` 会把 1600 字的文章
    算成十几个"词"，导致所有「每千词密度」指标虚高几百倍，
    校验环节还会把正常中文稿误判为"过短"。

    中文按 1.6 字/词折算（现代汉语平均词长约 1.5~1.7 字）。
    """
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z][A-Za-z'\-]*", text))
    return en + zh / 1.6


def _split_sentences(text: str) -> list:
    """句子切分，保留句末标点。

    原实现用 re.split 直接按标点切开，句号被当作分隔符丢弃，
    中文报告里句子会连成一片（"…生活方式和工作模式 从智能手机…"）。
    """
    text = re.sub(r"\n+", " ", text)
    parts = re.findall(r"[^.!?。！？]+[.!?。！？]*", text)
    return [x.strip() for x in parts if len(x.strip()) > 10]


def _looks_like_reference_line(line: str) -> bool:
    """判断一段是否像参考文献条目。

    why：原来只靠 "References" 标题定位参考文献起点。
    annotated bibliography 这类作业通常直接列条目、没有标题，
    于是所有参考文献被当成正文段落参与计算——
    一个 300 词的注释段紧挨着一个 33 词的文献条目，
    段落长度方差被人为拉大，paragraph-rhythm 从 1.00 掉到 0.00，
    总分凭空低 5 个百分点。同一份文字换个排版就换一个分数。

    这里改用条目自身的特征识别，与文档排版无关：
      1. 含 DOI / doi: 链接
      2. APA 作者格式：Surname, A. B. ... (2020).
      3. 卷(期), 页码 的模式
    """
    if re.search(r"https?://doi\.org|doi:\s*10\.", line, re.I):
        return True
    if re.match(r"^[A-Z][\w'\-]+,\s+[A-Za-z]\.", line) and re.search(r"\(\d{4}[a-z]?\)", line):
        return True
    if re.search(r"\b\d+\s*\(\d+\)\s*,\s*\d+\s*[-–]\s*\d+", line):
        return True
    return False


def _extract_prose_only(text: str) -> tuple:
    """自动排除引用/参考文献部分，只返回正文（prose）"""
    # 找到 References / Bibliography / Works Cited 的位置
    ref_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Works\s+Cited\s*\n',
        r'\n\s*Reference\s+List\s*\n',
    ]
    ref_start = len(text)
    for pat in ref_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ref_start = min(ref_start, m.start())

    prose = text[:ref_start]
    references = text[ref_start:]

    # 也排除 DOI 行、URL 行
    prose_lines = []
    for line in prose.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 排除 DOI 行
        if re.match(r'^https?://doi\.org', line, re.IGNORECASE):
            continue
        if re.match(r'^doi:', line, re.IGNORECASE):
            continue
        # 排除标题行（短的、全大写的）
        if len(line) < 60 and line.isupper():
            continue
        # 排除参考文献条目（不依赖 "References" 标题是否存在）
        if _looks_like_reference_line(line):
            continue
        prose_lines.append(line)

    return '\n'.join(prose_lines), references


def _compute_perplexity_ngram(text: str, n: int = 3) -> float:
    """基于 n-gram 的简化困惑度计算（打磨版，不需要下载大模型）"""
    words = text.lower().split()
    if len(words) < n + 5:
        return 0.5

    # 构建 n-gram 频率表（打磨：用 trigram 更准）
    ngrams = Counter()
    unigrams = Counter()

    for w in words:
        unigrams[w] += 1
    for i in range(len(words) - n + 1):
        ngrams[tuple(words[i:i+n])] += 1

    # 计算平均负对数似然
    log_prob_sum = 0.0
    count = 0
    total_words = len(words)

    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        prev = words[i]
        curr = words[i+1]

        ngram_count = ngrams.get(ngram, 0)
        unigram_count = unigrams.get(prev, 0)

        if unigram_count > 0:
            # 加 0.5 平滑（打磨：改进平滑算法）
            prob = (ngram_count + 0.5) / (unigram_count + 0.5 * len(unigrams))
        else:
            prob = 1.0 / total_words

        log_prob_sum -= math.log(prob)
        count += 1

    if count == 0:
        return 0.5

    avg_log_prob = log_prob_sum / count

    # 归一化
    # 低困惑度（2-3）= 可预测 = AI 写的
    # 高困惑度（5-7）= 不可预测 = 人写的
    # 我们输出 AI 概率：困惑度越低，AI 概率越高
    ai_prob = max(0.1, min(0.9, (4.5 - avg_log_prob) / 2.0 + 0.3))
    return round(ai_prob, 3)


def _sliding_window_detection(sentences: list, window_size: int = 7) -> list:
    """段落级滑动窗口检测：7句为一个窗口，带上下文"""
    if len(sentences) < window_size:
        return [0.5] * len(sentences)

    window_scores = []
    for i in range(len(sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        window = sentences[start:end]
        window_text = ' '.join(window)

        # 计算这个窗口的 burstiness
        lengths = [int(_word_count(s)) for s in window]
        avg = sum(lengths) / len(lengths)
        std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
        burstiness = std / (avg + 1)

        # 精雕细琢：加了句长均匀度检测
        if avg > 0:
            uniformity = 1 - (std / avg)
        else:
            uniformity = 0

        # 窗口内的 AI 概率
        if burstiness < 0.35 and uniformity > 0.65:
            window_ai = 0.75
        elif burstiness < 0.55:
            window_ai = 0.55
        else:
            window_ai = 0.25

        window_scores.append(window_ai)

    return window_scores


def validate_text(text: str) -> tuple:
    """输入质量校验。

    统计型检测器对过短、重复、非自然语言的文本会给出看似确定的分数，
    这在学术诚信场景里比"报错"更危险 —— 老师会把噪声当结论。
    不合格时直接拒绝判定。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, "文本为空"
    chars = len(stripped)
    wc = _word_count(stripped)

    if chars < 200:
        return False, f"文本过短（{chars} 字符）；至少需要 200 字符，约 3~4 个段落"
    if wc < 50:
        return False, f"篇幅过短（约 {int(wc)} 词）；至少需要 50 词"

    sentences = [s for s in _split_sentences(stripped) if int(_word_count(s)) >= 2]
    if len(sentences) < 5:
        return False, f"完整句子仅 {len(sentences)} 句；至少需要 5 句才能估计句长分布"

    letters = sum(1 for c in stripped if c.isalpha())
    if letters / chars < 0.40:
        return False, "字母比例过低，疑似非自然语言文本（表格、代码或符号串）"

    # 中英文分别统计词汇多样性（中文按字符二元组近似）
    if _detect_language(stripped) == "zh":
        chars_only = re.sub(r"[^\u4e00-\u9fff]", "", stripped)
        grams = [chars_only[i:i+2] for i in range(len(chars_only) - 1)]
        diversity = len(set(grams)) / max(len(grams), 1)
        if diversity < 0.05:
            return False, (f"用词重复度异常（{diversity:.1%}）；"
                           "疑似同一句反复粘贴，不能作为检测对象")
    else:
        words = stripped.split()
        uniq = len({w.lower().strip(".,!?;:") for w in words})
        diversity = uniq / max(len(words), 1)
        if diversity < 0.15:
            return False, (f"词汇多样性仅 {diversity:.0%}，重复度异常；"
                           "疑似同一句反复粘贴，不能作为检测对象")
    if diversity < 0.15:
        return False, (f"词汇多样性仅 {diversity:.0%}，重复度异常；"
                       "疑似同一句反复粘贴，不能作为检测对象")
    return True, ""


def _compute_engine_scores(prose_text: str):
    """跑全部检测引擎，只产出原始读数，不做任何判断。

    原来这一段和权重表、聚合、报告组装全塞在 run_ai_detection 里（326 行），
    改一个阈值要重跑全部计算，也没法单独测某个引擎。
    切开之后这里只负责「测到什么」，意味着什么交给后面的 _aggregate。

    返回 (engine_scores, sentences, window_scores)。后两者是逐句标注要用的
    中间产物，一并带出去，避免重复切句。
    """
    engine_scores = {}
    # 引擎 1: lmscan（对正文跑）
    lmscan_score = _detect_with_lmscan(prose_text)
    if lmscan_score is not None:
        engine_scores["lmscan"] = lmscan_score

    # 引擎 2: n-gram 困惑度（新增，真正的语言模型困惑度）
    pp_ngram_score = _compute_perplexity_ngram(prose_text, n=3)
    engine_scores["ngram-perplexity"] = pp_ngram_score

    # 引擎 3: 滑动窗口突发度（新增，段落级上下文检测）
    sentences = _split_sentences(prose_text)
    window_scores = _sliding_window_detection(sentences, window_size=5)
    avg_window_score = sum(window_scores) / len(window_scores) if window_scores else 0.5
    engine_scores["sliding-window"] = avg_window_score

    # 引擎 4: 传统 perplexity-burstiness（保留）
    pp_score = _detect_perplexity_burstiness(prose_text)
    if pp_score is not None:
        engine_scores["burstiness-stat"] = pp_score

    # 引擎 5: baseline（保留兜底）
    baseline_score, flagged = _detect_with_baseline(prose_text)
    engine_scores["statistical-baseline"] = baseline_score

    # 引擎 6-9: lynote 逆向集成的 4 个新信号
    engine_scores["formulaic-markers"] = _detect_formulaic_markers(prose_text)
    engine_scores["transition-density"] = _detect_transition_density(prose_text)
    engine_scores["compressibility"] = _detect_compressibility(prose_text)
    engine_scores["structured-shape"] = _detect_structured_shape(prose_text)

    # 引擎 10: sloptotal 逆向集成的 Formulaic 公式化检测
    engine_scores["formulaic-patterns"] = _detect_formulaic(prose_text)

    # 引擎 11: sloptotal 逆向集成的 Linguistic 语言标志检测
    engine_scores["linguistic-markers"] = _detect_linguistic_markers(prose_text)

    # 引擎 12: sloptotal 逆向集成的 Vocabulary 词汇丰富度检测
    engine_scores["vocabulary-richness"] = _detect_vocabulary_richness(prose_text)

    # 引擎 13: sloptotal 逆向集成的 Structural 结构特征检测
    engine_scores["structural-patterns"] = _detect_structural_patterns(prose_text)

    # 引擎 14: sloptotal 逆向集成的 Sentiment & Hedging 情感与对冲检测
    engine_scores["sentiment-hedging"] = _detect_sentiment_hedging(prose_text)

    # 引擎 15: RoBERTa ChatGPT 检测器（真正的 ML 分类器）
    roberta_score = _detect_with_roberta(prose_text)
    if roberta_score is not None:
        engine_scores["roberta-chatgpt"] = roberta_score

    # 引擎 16: 人类瑕疵检测器（全新角度！越完美越像 AI）
    engine_scores["human-imperfection"] = _detect_human_imperfection(prose_text)

    # 段落节奏（不依赖词表，跨语言有效）
    engine_scores["paragraph-rhythm"] = _detect_paragraph_rhythm(prose_text)

    # 中文字符熵（对改写最鲁棒）
    engine_scores["zh-char-entropy"] = _detect_zh_char_entropy(prose_text)

    # 语言模型困惑度（装了本地模型才启用）
    ppl_score = _detect_lm_perplexity(prose_text)
    if ppl_score is not None:
        engine_scores["lm-perplexity"] = ppl_score

    # 引擎 17: 情感曲线检测器（AI 情感很平，人类有起伏）
    engine_scores["emotion-curve"] = _detect_emotion_curve(prose_text)

    # 引擎 18: 具体细节密度检测器（人类有很多具体数字/人名/地点）
    engine_scores["specific-details"] = _detect_specific_details(prose_text)

    # 引擎 19: 自我引用检测器（人类会说"我认为"、"我觉得"）
    engine_scores["self-references"] = _detect_self_references(prose_text)

    # 引擎 20: 逻辑连接词密度检测器（AI 用很多 therefore/however）
    engine_scores["logical-connectors"] = _detect_logical_connectors(prose_text)

    # 引擎 21: 隐藏字符检测器（检测AI水印：零宽字符、不可见字符）
    engine_scores["hidden-characters"] = _detect_hidden_characters(prose_text)

    # 引擎 22: 开头模式检测器（合并句首词分布 + 段落开头模式）
    engine_scores["starter-patterns"] = _detect_starter_patterns(prose_text)

    # 引擎 23: 标点使用模式检测器（合并感叹句/疑问句 + 标点符号使用）
    engine_scores["punctuation-patterns"] = _detect_punctuation_patterns(prose_text)

    # 引擎 24: 句子长度分布检测器（合并长句/短句交替 + 滑动窗口）
    engine_scores["sentence-length"] = _detect_sentence_length_distribution(prose_text)

    # 引擎 25: 词汇多样性检测器（合并重复词 + 词汇丰富度）
    engine_scores["vocabulary-diversity"] = _detect_vocabulary_diversity(prose_text)


    # 引擎 28: PAWN 困惑度注意力加权检测器（零样本范式）
    engine_scores["pawn-perplexity"] = _detect_pawn_perplexity(prose_text)

    # 引擎 29: 对抗扰动程度检测器（对抗鲁棒性）
    engine_scores["adversarial-perturbation"] = _detect_adversarial_perturbation(prose_text)

    # 引擎 30: 中文AI检测器（多语言支持）
    engine_scores["chinese-ai"] = _detect_chinese_ai(prose_text)

    # 引擎 31: 统计水印检测器（统计水印）
    engine_scores["statistical-watermark"] = _detect_statistical_watermark(prose_text)

    # 引擎 32: SpecDetect 信号处理检测器（零样本范式）
    engine_scores["spec-detect"] = _detect_spec_detect(prose_text)

    # 引擎 33: DNA-DetectLLM（零样本范式）
    engine_scores["dna-detect"] = _detect_dna_detect(prose_text)
    return engine_scores, sentences, window_scores


def run_ai_detection(text: str) -> AIResult:
    """跑所有可用的 AI 检测引擎，返回综合结果"""
    engine_scores = {}

    # 步骤 0: 自动排除引用/参考文献部分
    prose_text, references = _extract_prose_only(text)
    if len(prose_text.strip()) < 50:
        prose_text = text  # 如果提取后太短，就用原文

    engine_scores, sentences, window_scores = _compute_engine_scores(prose_text)

    # ============================================================
    # 集成 v2：显式引擎注册表 + 稳健聚合
    # ------------------------------------------------------------
    # 旧实现的问题：
    #   weights.get(name, 0.1)  —— 未列进表的引擎自动拿 0.1 权重，
    #   比表里一半引擎还高。sliding-window / vocabulary-richness /
    #   vocabulary-diversity 三个噪声引擎就这样各拿 0.1，
    #   把所有文本的分数都压到 0.41~0.45 之间。
    #
    # 新规则：
    #   1) 未注册的引擎权重为 0（不再有默认值兜底）
    #   2) 输出恰好等于 0 / 0.5 / 1 的引擎视为"恒定输出"，自动降到 15%
    #   3) 加权均值 + 一次离群剔除
    #   4) 输出置信度（基于采纳引擎的加权标准差）
    #
    # 权重来自 6 个已知来源样本上的实测区分力（人写 vs AI 写）。
    # ============================================================
    # 名字: (方向, 权重, 备注)  方向 +1 表示分数高 = 像 AI
    ENGINE_SPEC = {
        # —— 实测区分力强 ——
        "roberta-chatgpt":       EngineSpec(+1, 0.25, "ML 分类器（需联网下载模型）"),
        "lmscan":                EngineSpec(+1, 0.15, "外部 CLI"),
        "linguistic-markers":    EngineSpec(+1, 0.12, "AI 高频词表"),
        "formulaic-patterns":    EngineSpec(+1, 0.12, "公式化表达"),
        "emotion-curve":         EngineSpec(+1, 0.10, "情感曲线平坦度"),
        "transition-density":    EngineSpec(+1, 0.08, "过渡词密度"),
        "sentiment-hedging":     EngineSpec(+1, 0.08, "对冲语密度"),
        # —— 中等 ——
        "sentence-length":       EngineSpec(+1, 0.07, "句长分布"),
        "burstiness-stat":       EngineSpec(+1, 0.07, "突发度"),
        "statistical-baseline":  EngineSpec(+1, 0.06, "统计基线"),
        "sliding-window":        EngineSpec(+1, 0.06, "段落滑动窗口"),
        "human-imperfection":    EngineSpec(+1, 0.06, "人味特征（v2 重写）"),
        "paragraph-rhythm":      EngineSpec(+1, 0.07, "段落长度均匀度（跨语言）"),
        "zh-char-entropy":       EngineSpec(+0, 0.0, "仅中文启用，见语言自适应"),
        "lm-perplexity":         EngineSpec(+1, 0.22, "语言模型困惑度（需本地模型）"),
        "formulaic-markers":     EngineSpec(+1, 0.05, "套话标记"),
        "vocabulary-richness":   EngineSpec(+1, 0.05, "词汇丰富度"),
        # —— 弱 ——
        "starter-patterns":      EngineSpec(+1, 0.04, "句首模式"),
        "specific-details":      EngineSpec(+1, 0.02, "具体细节密度（LLM 已普遍生成数值，弱信号）"),
        "adversarial-perturbation": EngineSpec(+1, 0.04, "对抗扰动"),
        "compressibility":       EngineSpec(+1, 0.03, "压缩率"),
        "pawn-perplexity":       EngineSpec(+1, 0.03, "PAWN"),
        "ngram-perplexity":      EngineSpec(+1, 0.02, "对短文本不可靠，仅作参考"),
        # —— 已实测失效，权重置 0（仍显示，便于对照）——
        "self-references":       (0, 0.0, "AI 同样高频使用 I think，无区分力"),
        "logical-connectors":    (0, 0.0, "几乎恒为 1.0"),
        "vocabulary-diversity":  (0, 0.0, "方向相反"),
        "structural-patterns":   (0, 0.0, "区分力弱"),
        "structured-shape":      (0, 0.0, "恒为 0"),
        "hidden-characters":     (0, 0.0, "恒为 0"),
        "punctuation-patterns":  (0, 0.0, "恒为 1.0"),
        "statistical-watermark": (0, 0.0, "方向相反"),
        "spec-detect":           (0, 0.0, "近乎恒定"),
        "dna-detect":            (0, 0.0, "恒为 1.0，实现有误"),
        "chinese-ai":            (0, 0.0, "仅中文文本生效"),
    }

    # 语言自适应：中文文本下，英文词表类引擎基本失效，必须降权，
    # 否则它们的噪声会盖过唯一有效的中文引擎。
    _lang = _detect_language(prose_text)
    if _lang == "zh":
        _EN_ONLY = ("linguistic-markers", "formulaic-markers", "formulaic-patterns",
                    "sentiment-hedging", "transition-density", "starter-patterns",
                    "self-references", "logical-connectors", "ngram-perplexity",
                    "pawn-perplexity", "adversarial-perturbation")
        for _k in _EN_ONLY:
            if _k in ENGINE_SPEC:
                _d, _w, _n = ENGINE_SPEC[_k]
                ENGINE_SPEC[_k] = EngineSpec(_d, round(_w * 0.2, 4), _n + "（对中文无效，降权）")
        ENGINE_SPEC["chinese-ai"] = EngineSpec(+1, 0.24, "中文 AI 特征")
        ENGINE_SPEC["zh-char-entropy"] = EngineSpec(+1, 0.20, "中文字符熵（抗改写）")

    adopted = {}       # name -> (score, weight)
    rejected = {}      # name -> 原因
    for name, score in engine_scores.items():
        spec = ENGINE_SPEC.get(name)
        if spec is None:
            rejected[name] = "未注册"
            continue
        direction, w, _note = spec
        if direction == 0 or w <= 0:
            rejected[name] = "实测无效"
            continue
        s = score if direction > 0 else 1.0 - score
        # 只把"精确 0.5"视为引擎未触发（多数函数在文本不足时 return 0.5）。
        # 0.0 和 1.0 都是合法信号：很多引擎对极端文本会给出饱和值，
        # 早期版本把 1.0 也一并降权，导致 AI 检测中最有价值的满分引擎
        # 反而被压到 0.015，属回归。已被弃用的引擎权重在注册表里就是 0，
        # 不依赖这条规则兜底。
        if abs(s - 0.5) < 1e-9:
            w *= 0.15
        adopted[name] = (s, w)

    if adopted:
        # 第一轮加权均值
        tw = sum(w for _, w in adopted.values())
        ws = sum(s * w for s, w in adopted.values())
        mean1 = ws / tw if tw else 0.5
        # 剔除偏离均值 0.45 以上的引擎后重算
        filt = {k: v for k, v in adopted.items() if abs(v[0] - mean1) <= 0.45}
        if not filt:
            filt = adopted
        tw = sum(w for _, w in filt.values())
        ws = sum(s * w for s, w in filt.values())
        final_score = ws / tw if tw else 0.5
        # 加权标准差 → 置信度
        wvar = sum(w * (s - final_score) ** 2 for s, w in filt.values()) / tw if tw else 0
        wstd = wvar ** 0.5
        confidence = "high" if wstd < 0.15 else ("medium" if wstd < 0.25 else "low")
    else:
        final_score, wstd, confidence = 0.5, 0.0, "low"

    engine_scores = dict(engine_scores)  # 保留全部，供报告展示

    # 判定
    # 阈值随权重体系重标定：人写文本天然得分偏低，故下界放宽
    # 阈值在 41 篇真实语料上重标定：
    #   人类上界 0.385 / AI 下界 0.488，中间留出 mixed 带供人工复核
    if final_score > 0.58:
        label = "ai"
    elif final_score < 0.33:
        label = "human"
    else:
        label = "mixed"

    # 引擎分歧度：沿用加权标准差（仅统计被采纳的引擎）
    disagreement = wstd

    # 高度确信AI的引擎（>0.7）
    confident_engines = [
        {"name": name, "score": round(score, 3)}
        for name, score in engine_scores.items()
        if score > 0.7
    ]
    confident_engines.sort(key=lambda x: x["score"], reverse=True)

    # 分歧大的引擎对（最高和最低分差 > 0.5）
    sorted_engines = sorted(engine_scores.items(), key=lambda x: x[1])
    conflicting_engines = []
    if len(sorted_engines) >= 2:
        lowest = sorted_engines[0]
        highest = sorted_engines[-1]
        if highest[1] - lowest[1] > 0.5:
            conflicting_engines = [
                {"lowest": {"name": lowest[0], "score": round(lowest[1], 3)}},
                {"highest": {"name": highest[0], "score": round(highest[1], 3)}},
            ]

    # 用滑动窗口的逐句分数做高风险句
    flagged_sentences = []
    for i, s in enumerate(sentences):
        if i < len(window_scores) and window_scores[i] > 0.5:
            # 计算句子级不确定性（基于窗口分数的标准差）
            window_start = max(0, i - 2)
            window_end = min(len(window_scores), i + 3)
            local_window = window_scores[window_start:window_end]
            if len(local_window) > 1:
                local_mean = sum(local_window) / len(local_window)
                local_var = sum((x - local_mean) ** 2 for x in local_window) / len(local_window)
                uncertainty = local_var ** 0.5
            else:
                uncertainty = 0.0
            
            flagged_sentences.append({
                "text": s[:100],
                "score": round(window_scores[i], 3),
                "uncertainty": round(uncertainty, 3),
                "confidence": "high" if uncertainty < 0.1 else ("medium" if uncertainty < 0.2 else "low"),
            })
    flagged_sentences.sort(key=lambda x: x["score"], reverse=True)

    # —— 改写嫌疑：词汇层仍带 AI 特征，但句式/情感层被打散 ——
    # 这是"AI 生成后人工改写"或改写工具处理后最常见的信号分裂形态。
    # 单看总分永远判不出来，因为改写本来就压低了绝大多数特征。
    rewrite_hint = ""
    _lmk = engine_scores.get("linguistic-markers")
    _fmp = engine_scores.get("formulaic-patterns")
    # 该判据基于英文词表（linguistic-markers 对中文恒为 0 或饱和），
    # 中文文本下会稳定误报，故只在英文语境启用。
    if (_lang == "en" and _lmk is not None and _lmk > 0.8
            and _fmp is not None and _fmp < 0.25):
        rewrite_hint = (
            f"词汇层仍有 AI 特征（linguistic-markers {_lmk:.0%}），"
            f"但句式与模板层已被打散（formulaic-patterns {_fmp:.0%}）。"
            "典型形态为「AI 生成后人工改写」或改写工具处理，"
            "此类文本的总分通常会被压低，需单独判断。"
        )

    return AIResult(
        ai_probability=round(final_score, 4),
        label=label,
        method=f"ensemble v2 ({len(adopted)}/{len(engine_scores)} engines adopted, confidence={confidence}, references excluded)",
        engine_scores=engine_scores,
        flagged_sentences=flagged_sentences[:10],
        engine_disagreement=round(disagreement, 4),
        confident_engines=confident_engines[:5],
        conflicting_engines=conflicting_engines,
        confidence=confidence,
        adopted_engines=sorted(
            [{"name": k, "score": round(v[0], 3), "weight": round(v[1], 3)}
             for k, v in adopted.items()],
            key=lambda x: x["weight"] * abs(x["score"] - final_score), reverse=True),
        rejected_engines=[{"name": k, "reason": v} for k, v in rejected.items()],
        rewrite_hint=rewrite_hint,
    )


# ============================================================
# 相似度检测
# ============================================================

def run_plagiarism_check(text: str) -> PlagiarismResult:
    """检测文本自重叠（简单版，没有外部语料库）"""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    if len(paragraphs) < 2:
        return PlagiarismResult(external_similarity=0.0, self_overlap=0.0)

    # 简单的词袋余弦相似度
    max_overlap = 0.0
    matched = []

    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):
            sim = _cosine_sim(paragraphs[i], paragraphs[j])
            if sim > max_overlap:
                max_overlap = sim
                matched = [
                    {"para1": paragraphs[i][:80] + "...", "sim": round(sim, 3)},
                    {"para2": paragraphs[j][:80] + "..."},
                ]

    return PlagiarismResult(
        external_similarity=0.0,  # 没有外部语料库
        self_overlap=round(max_overlap, 4),
        matched_excerpts=matched[:5],
    )


def _cosine_sim(text1: str, text2: str) -> float:
    """简单的词袋余弦相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    # Jaccard 相似度（比余弦更简单，效果差不多）
    return len(intersection) / len(union)


# ============================================================
# 引用验证（onecite）
# ============================================================

def _doi_to_bibtex(doi: str) -> str:
    """用 doi.org 的内容协商取 BibTeX，不依赖任何外部命令。

    原实现只调 onecite CLI。这个命令默认没装，subprocess 抛异常后
    所有 DOI 都被记成"解析失败"——报告会把有效的 DOI 说成有问题，
    把"工具没装"伪装成"文献有问题"，属于最坏的一类误报。
    """
    import urllib.request

    url = "https://doi.org/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"Accept": "application/x-bibtex"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", "replace")
        return body.strip() if "@" in body else ""
    except Exception:
        return ""


def run_citation_check(text: str) -> CitationResult:
    """提取引用并逐一验证：优先 onecite，缺失时回落到 doi.org 内容协商。"""
    import shutil

    # 提取 DOI，末尾的标点不算
    dois = re.findall(r"10\.\d{4,}/[^\s\]\)>,;]+", text)
    dois = list(dict.fromkeys(dois))  # 去重并保留顺序

    if not dois:
        return CitationResult(total_extracted=0, resolved=0)

    has_onecite = shutil.which("onecite") is not None
    resolved = 0
    failed = []
    bibtex_entries = []

    for doi in dois[:20]:  # 最多处理 20 条
        entry = ""
        if has_onecite:
            try:
                result = subprocess.run(
                    ["onecite", "process", doi, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    entry = result.stdout.strip()
            except Exception:
                entry = ""
        if not entry:
            entry = _doi_to_bibtex(doi)
        if entry:
            resolved += 1
            bibtex_entries.append(entry)
        else:
            failed.append(doi)

    return CitationResult(
        total_extracted=len(dois),
        resolved=resolved,
        failed=failed,
        bibtex="\n\n".join(bibtex_entries),
    )


# ============================================================
# 主入口
# ============================================================

def run_check(text: str = "", file_path: str = "") -> CheckReport:
    """
    一键跑完整查重流程

    参数：
        text: 直接传入的文本（和 file_path 二选一）
        file_path: 从文件读取（txt/md/docx）

    返回：
        CheckReport 对象，用 .to_markdown() 转报告
    """
    # 获取文本
    if file_path:
        text = extract_text_from_file(file_path)

    ok, reason = validate_text(text)
    if not ok:
        return CheckReport(text_length=len((text or "").strip()), note=reason)

    # 跑各模块
    ai_result = run_ai_detection(text)
    plagiarism_result = run_plagiarism_check(text)
    citation_result = run_citation_check(text)

    return CheckReport(
        text_length=len(text),
        ai=ai_result,
        plagiarism=plagiarism_result,
        citations=citation_result,
    )


# CLI 入口
if __name__ == "__main__":
    import sys
    _flags = {a for a in sys.argv[1:] if a.startswith("--")}
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.argv = [sys.argv[0]] + _args

    if len(sys.argv) < 2:
        print("用法:")
        print("  AI 检测:   python aurora_check.py <文件路径> [--json] [--no-html]")
        print("  文档比对:  python aurora_check.py compare <当前文档> <参考文档>")
        print("  环境变量:  AURORA_VERBOSE=1 显示模型加载细节")
        sys.exit(1)

    # 文档比对模式
    if sys.argv[1] == "compare" and len(sys.argv) >= 4:
        result = compare_with_reference(sys.argv[2], sys.argv[3])
        print("=" * 50)
        print("文档比对结果")
        print("=" * 50)
        print(f"整体相似度: {result['overall_similarity']:.1%}")
        print(f"重叠句子数: {result['overlapping_count']}")
        print()
        if result["overlapping_sentences"]:
            print("高重叠句子 (Top 10):")
            for i, s in enumerate(result["overlapping_sentences"][:10]):
                print(f"  {i+1}. 相似度 {s['similarity']:.1%}")
                print(f"     当前文档: {s['doc_sentence'][:80]}...")
                print(f"     参考文档: {s['ref_sentence'][:80]}...")
                print()
        sys.exit(0)

    # 普通 AI 检测模式
    report = run_check(file_path=sys.argv[1])
    if "--json" in _flags:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        sys.exit(0)
    print(report.to_markdown())

    # 生成 HTML 报告
    if "--no-html" not in _flags:
        base_name = os.path.splitext(sys.argv[1])[0]
        html_path = base_name + "_aurora_report.html"
        report.to_html(output_path=html_path)
        print(f"\nHTML 报告已生成: {html_path}")
