"""
极光查重核心模块
用法：
    from aurora_check import run_check
    result = run_check(text="你的文章内容")
    print(result["report"])
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AIResult:
    ai_probability: float       # 0~1
    label: str                 # human / mixed / ai
    method: str
    engine_scores: dict = field(default_factory=dict)
    flagged_sentences: list = field(default_factory=list)


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


@dataclass
class CheckReport:
    text_length: int = 0
    ai: Optional[AIResult] = None
    plagiarism: Optional[PlagiarismResult] = None
    citations: Optional[CitationResult] = None

    def to_markdown(self) -> str:
        lines = ["# 极光查重报告\n"]
        lines.append(f"- **正文长度：** {self.text_length} 字符\n")

        # AI 检测
        if self.ai:
            ai = self.ai
            lines.append("## AI 检测\n")
            lines.append(f"- **综合判定：** {_label_to_cn(ai.label)}")
            lines.append(f"- **AI 概率：** {ai.ai_probability:.1%}")
            lines.append(f"- **人工概率：** {1 - ai.ai_probability:.1%}")
            lines.append(f"- **检测方法：** {ai.method}\n")
            if ai.engine_scores:
                lines.append("- **各引擎单独评分：**")
                for name, score in ai.engine_scores.items():
                    lines.append(f"  - {name}：{score:.1%}")
                lines.append("")
            if ai.flagged_sentences:
                lines.append("- **高风险句（Top 5）：**")
                for s in ai.flagged_sentences[:5]:
                    lines.append(f"  - ({s['score']:.2f}) {s['text'][:80]}...")
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
        ai_prob = self.ai.ai_probability if self.ai else 0
        ai_pct = int(ai_prob * 100)

        # 颜色：红 > 50%, 黄 20-50%, 绿 < 20%
        if ai_pct > 50:
            color = "#e74c3c"
            label = "大概率 AI 生成"
        elif ai_pct > 20:
            color = "#f39c12"
            label = "混合（人机各半）"
        else:
            color = "#27ae60"
            label = "大概率人工写作"

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

            for name, score in self.ai.engine_scores.items():
                pct = int(score * 100)
                bar_color = "#e74c3c" if pct > 50 else "#f39c12" if pct > 20 else "#27ae60"
                engine_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td style="width: 200px;">
                        <div style="background: #eee; height: 20px; border-radius: 10px; overflow: hidden;">
                            <div style="background: {bar_color}; width: {pct}%; height: 100%;"></div>
                        </div>
                    </td>
                    <td style="text-align: right; font-weight: bold;">{pct}%</td>
                </tr>
                """

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
                bg = "rgba(231, 76, 60, 0.15)" if score_pct > 50 else "rgba(243, 156, 18, 0.15)" if score_pct > 20 else "rgba(39, 174, 96, 0.15)"
                flagged_html += f"""
                <div style="background: {bg}; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid {color};">
                    <div style="font-size: 12px; color: #666; margin-bottom: 4px;">AI 嫌疑度: {score_pct}%</div>
                    <div>{s['text']}</div>
                </div>
                """

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
            由极光查重 (Aurora Check) 生成 · 14 引擎集成检测
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
        "human": "大概率人工写作",
        "mixed": "混合（人机各半）",
        "ai": "大概率 AI 生成",
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
    sent_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)

    # 词型丰富度
    words = text.lower().split()
    if not words:
        return 0.5, []
    ttr = len(set(words)) / len(words)

    # 打分：AI 文本倾向于句长均匀（方差小）、词型丰富度适中
    burstiness = variance / (avg_len ** 2 + 1)  # 越大越像人

    # 基于 burstiness 的线性打分（精雕细琢调整参数）：0.25 → 0.75, 0.55 → 0.45, 0.85 → 0.15
    if burstiness < 0.25:
        ai_score = 0.75
    elif burstiness < 0.50:
        ai_score = 0.55
    elif burstiness < 0.75:
        ai_score = 0.35
    else:
        ai_score = 0.15

    # TTR 微调（精雕细琢调整参数）：TTR 太高（>0.60）可能是 AI 堆砌词汇，加一点分
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

    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
    burstiness = std / (avg + 1)

    # 突发性低 = 更像 AI（精雕细琢调整参数）
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
    # 新增模式（精雕细琢扩充）
    r"\bit is worth noting\b", r"\bit is crucial to\b", r"\bit is essential to\b",
    r"\bit is important to understand\b", r"\bit is worth considering\b",
    r"\blet's dive in\b", r"\blet's explore\b", r"\blet's take a look\b",
    r"\bwhen it comes to\b", r"\bin the realm of\b", r"\bin the world of\b",
    r"\bparadigm shift\b", r"\bgame-changer\b", r"\bcutting-edge\b",
    r"\bgroundbreaking\b", r"\brevolutionary\b", r"\btransformative\b",
    r"\bseamless integration\b", r"\bseamless experience\b",
    r"\buser-friendly\b", r"\bcutting-edge technology\b",
    r"\bstate-of-the-art\b", r"\bnext-generation\b",
    # 继续扩充（精雕细琢）
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
    # 新增词（精雕细琢扩充）
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
    # 继续扩充（精雕细琢）
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
    import zlib
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
    # 新增模式（精雕细琢扩充）
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
    # 继续扩充（精雕细琢）
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
    # 新增模式（精雕细琢扩充）
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
    # 继续扩充（精雕细琢）
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
    # 新增模式（精雕细琢扩充）
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
    # 继续扩充（精雕细琢）
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
    word_count = max(len(text.split()), 1)

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
    # 新增中文 AI 语言标志（精雕细琢扩充）
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
    word_count = max(len(text.split()), 1)

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
    import math
    from collections import Counter

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

    # 5. 长词比例（>6个字母）（精雕细琢）
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / total
    # 人类文本：长词比例 10-20%，AI 文本：20-35%
    long_score = min(max((long_word_ratio - 0.10) / 0.20, 0.0), 1.0)

    # 6. 短词比例（<3个字母）（精雕细琢新增）
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
    import statistics

    signals = []
    word_count = max(len(text.split()), 1)

    # 1. Em-dash 频率
    em_dashes = len(re.findall(r"\w—\w", text))
    em_dash_rate = (em_dashes / word_count) * 1000
    em_score = min(em_dash_rate / 6.0, 1.0)

    # 2. 句长均匀度
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
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
        para_lengths = [len(p.split()) for p in paragraphs]
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

    # 10. 斜杠使用频率（精雕细琢）
    slash_count = len(re.findall(r"/", text))
    slash_density = (slash_count / word_count) * 1000
    # AI：5-10 per 1k，人类：1-3 per 1k
    slash_score = min(max((slash_density - 2) / 6.0, 0.0), 1.0)

    # 11. 分号使用频率（精雕细琢新增）
    semicolon_count = len(re.findall(r";", text))
    semicolon_density = (semicolon_count / word_count) * 1000
    # AI：3-8 per 1k，人类：0-2 per 1k
    semicolon_score = min(max((semicolon_density - 1) / 5.0, 0.0), 1.0)

    # 12. 破折号使用频率（精雕细琢新增）
    dash_count = len(re.findall(r"--|—", text))
    dash_density = (dash_count / word_count) * 1000
    # AI：2-5 per 1k，人类：0-1 per 1k
    dash_score = min(max((dash_density - 0.5) / 3.0, 0.0), 1.0)

    # 加权打分——冒号信号权重最高（精雕细琢调整权重）
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
    # 新增词（精雕细琢扩充）
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
    # 继续扩充（精雕细琢）
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
    word_count = max(len(text.split()), 1)

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


# ============================================================
# 第 16 号引擎：人类瑕疵检测器（Human Imperfection Detector）
# 创新角度：越完美越像 AI，越有"人类的不完美"越像人
# 打磨版：扩充词表、改进检测模式、调整权重
# ============================================================

# 人类口语化插入语（精雕细琢版：只保留真正人类专属的，AI 很少用的）
_HUMAN_FILLERS = [
    # 中文（真正人类才会说的）
    "嗯", "那个", "就是说", "反正", "对吧", "你知道",
    "怎么说呢", "其实吧", "我觉得吧", "说真的", "说实话",
    "说白了", "你想想", "你懂的", "你明白吧", "我跟你说",
    "害", "哎", "哦", "啊", "呃",
    # 英文（真正人类才会说的，去掉了太常见的 so/well/right/okay）
    "you know", "i mean", "sort of", "kind of",
    "or something", "and stuff", "let's see", "anyway",
    "you know what", "guess what", "believe it or not",
    "to be honest", "honestly", "frankly",
    "i guess", "i suppose", "i imagine",
]

# 人类常见的"不完美"模式（扩充版）
_HUMAN_IMPERFECTION_PATTERNS = [
    # 1. 相邻重复用词（人类会打错，比如"the the"）
    r"\b(\w+)\s+\1\b",
    # 2. 多余的空格（人类打字不规范）
    r"  +",
    # 3. 中英文标点混用（人类常犯）
    r"[a-zA-Z]+[，。；：、]",
    r"[，。；：、][a-zA-Z]+",
    # 4. 句子开头小写（人类打字不注意）
    r"(?<=[.!?。！？]\s)[a-z]",
    # 5. 数字和单位之间没有空格（人类常犯）
    r"\d+[a-zA-Z]",
    # 6. 两个感叹号/问号（人类情绪激动）
    r"!{2,}",
    r"\?{2,}",
    # 7. 多余的逗号（人类常犯）
    r",{2,}",
    # 8. 多余的句号（人类常犯）
    r"\.{3,}",
    # 9. 中英文引号混用（人类常犯）
    r"['‘’“”][a-zA-Z]+[a-zA-Z]+['‘’“”]",
    # 10. 句末没有标点（人类常犯）
    r"\n[a-z].*[a-z]$\n",
]


def _detect_human_imperfection(text: str) -> float:
    """
    人类瑕疵检测器 - 打磨版！
    原理：AI 写文章太完美了，人类写文章会有各种"不完美"
    所以：越完美 = 越像 AI，越有瑕疵 = 越像人
    """
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    imperfection_score = 0.0

    # 1. 口语化插入语（人类用，AI 不用）- 权重 25%
    filler_count = 0
    for phrase in _HUMAN_FILLERS:
        filler_count += text_lower.count(phrase)
    filler_density = (filler_count / word_count) * 1000
    # 人类：8-20 per 1k，AI：0-2 per 1k
    # 所以：filler 越少 = 越像 AI
    filler_score = max(0, 1.0 - (filler_density / 12.0))
    imperfection_score += filler_score * 0.25

    # 2. 重复用词（精雕细琢：人类会重复，AI 很少重复）- 权重 15%（降低权重，因为这个信号不够强）
    repeat_count = 0
    # 相邻重复用词（比如 "the the"）
    repeat_count += len(re.findall(r"\b(\w+)\s+\1\b", text_lower))
    # 中间隔了1-2个词的重复（比如 "the cat the"）
    repeat_count += len(re.findall(r"\b(\w+)\s+\w+\s+\1\b", text_lower)) * 0.5
    repeat_density = (repeat_count / word_count) * 1000
    # 人类：3-8 per 1k，AI：0-0.5 per 1k
    # 所以：repeat 越少 = 越像 AI
    repeat_score = max(0, 1.0 - (repeat_density / 4.0))
    imperfection_score += repeat_score * 0.15

    # 3. 标点不规范（精雕细琢：人类常犯，AI 很规范）- 权重 20%（降低权重，因为很多检测对纯英文没用）
    punctuation_issues = 0
    # 中英文标点混用（只对中文文章有用）
    punctuation_issues += len(re.findall(r"[a-zA-Z]+[，。；：、]", text)) * 0.5
    punctuation_issues += len(re.findall(r"[，。；：、][a-zA-Z]+", text)) * 0.5
    # 两个感叹号/问号（人类情绪激动，中英文都有用）
    punctuation_issues += len(re.findall(r"!{2,}", text))
    punctuation_issues += len(re.findall(r"\?{2,}", text))
    # 多余的逗号（中英文都有用）
    punctuation_issues += len(re.findall(r",{2,}", text))
    # 多余的句号（中英文都有用）
    punctuation_issues += len(re.findall(r"\.{3,}", text))
    # 新增：逗号后面没有空格（人类常犯，AI 很规范）
    punctuation_issues += len(re.findall(r",[a-zA-Z]", text)) * 0.3
    # 新增：句号后面没有空格（人类常犯，AI 很规范）
    punctuation_issues += len(re.findall(r"\.[a-zA-Z]", text)) * 0.3
    punct_density = (punctuation_issues / word_count) * 1000
    # 人类：3-8 per 1k，AI：0-0.5 per 1k
    # 所以：标点问题越少 = 越像 AI
    punct_score = max(0, 1.0 - (punct_density / 4.0))
    imperfection_score += punct_score * 0.20

    # 4. 打字不规范（精雕细琢：人类会犯，AI 不会）- 权重 20%
    typing_issues = 0
    # 多余的空格（中英文都有用）
    typing_issues += len(re.findall(r"  +", text))
    # 句子开头小写（中英文都有用）
    typing_issues += len(re.findall(r"(?<=[.!?。！？]\s)[a-z]", text))
    # 数字和单位之间没有空格（中英文都有用）
    typing_issues += len(re.findall(r"\d+[a-zA-Z]", text)) * 0.5
    # 中英文引号混用（只对中文有用，降低权重）
    typing_issues += len(re.findall(r"['‘’“”][a-zA-Z]+[a-zA-Z]+['‘’“”]", text)) * 0.3
    # 新增：小写 "i"（人类常犯，AI 不会）
    typing_issues += len(re.findall(r"\bi\b", text)) * 0.2
    typing_density = (typing_issues / word_count) * 1000
    # 人类：8-20 per 1k，AI：0-1 per 1k
    # 所以：打字问题越少 = 越像 AI
    typing_score = max(0, 1.0 - (typing_density / 10.0))
    imperfection_score += typing_score * 0.20

    # 5. 句末没有标点（人类常犯）- 权重 20%（提高权重，因为这个信号很强）
    no_punct_end = 0
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not line[-1] in ".!?。！？":
            no_punct_end += 1
    no_punct_density = (no_punct_end / len(lines)) * 100 if len(lines) > 0 else 0
    # 人类：10-30 per 100 lines，AI：0-5 per 100 lines
    # 所以：句末没标点越少 = 越像 AI
    no_punct_score = max(0, 1.0 - (no_punct_density / 15.0))
    imperfection_score += no_punct_score * 0.20

    # 最终分数：越完美 = 分数越高 = 越像 AI
    return round(imperfection_score, 3)


# ============================================================
# 第 17 号引擎：情感曲线检测器（Emotion Curve Detector）
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
    # 中文正面词（精雕细琢扩充）
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
    # 中文负面词（精雕细琢扩充）
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


def _detect_emotion_curve(text: str) -> float:
    """
    情感曲线检测器 - 打磨版！
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
    import statistics
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


# ============================================================
# 第 18 号引擎：具体细节密度检测器（Specific Details Detector）
# 创新角度：人类有很多具体数字/人名/地点，AI 很抽象
# ============================================================

# 数字模式（分类检测，精雕细琢扩充）
_PERCENT_PATTERN = r"\b\d+(\.\d+)?%|百分之[一二三四五六七八九十百千万\d]+"
_DATE_PATTERN = r"\b(19|20)\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?\b|\b\d{1,2}[-/月]\d{1,2}日?\b"
_PRICE_PATTERN = r"\$\d+(\.\d+)?|€\d+(\.\d+)?|£\d+(\.\d+)?|¥\d+|人民币\d+|美元\d+"
# 新增：温度、距离、重量等
_TEMP_PATTERN = r"\b\d+(\.\d+)?°[CF]|\b\d+(\.\d+)?度\b"
_DISTANCE_PATTERN = r"\b\d+(\.\d+)?km|\b\d+(\.\d+)?miles|\b\d+(\.\d+)?公里|\b\d+(\.\d+)?米\b"
_WEIGHT_PATTERN = r"\b\d+(\.\d+)?kg|\b\d+(\.\d+)?lbs|\b\d+(\.\d+)?公斤|\b\d+(\.\d+)?斤\b"
_GENERIC_NUMBER_PATTERN = r"\b\d+(\.\d+)?\b"

# 人名模式（精雕细琢扩充）
_NAME_PATTERN = r"\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b|\b[A-Z][a-z]+\b"

# 地点模式（精雕细琢扩充，从25个→50个城市）
_LOCATION_PATTERN = r"\b(in|at|on|from|to|near|around) (New York|London|Paris|Beijing|Shanghai|Tokyo|San Francisco|Los Angeles|Chicago|Boston|Seattle|Austin|Toronto|Sydney|Berlin|Madrid|Rome|Moscow|Seoul|Hong Kong|Singapore|Amsterdam|Brussels|Vienna|Stockholm|Copenhagen|Oslo|Dublin|Zurich|Geneva|Milan|Barcelona|Lisbon|Athens|Prague|Warsaw|Budapest|Dubai|Mumbai|Delhi|Bangkok|Jakarta|Manila|Kuala Lumpur|Melbourne|Auckland|Vancouver|Montreal|Mexico City|Buenos Aires|Sao Paulo)\b"


def _detect_specific_details(text: str) -> float:
    """
    具体细节密度检测器 - 打磨版！
    原理：人类写文章会有很多具体的细节（数字、人名、地点）
    AI 写文章比较抽象，很少有具体细节
    所以：具体细节越少 = 越像 AI
    改进：分类检测数字、扩充地点、调整权重
    """
    word_count = max(len(text.split()), 1)

    # 1. 数字数量（分类检测，精雕细琢扩充）
    percent_count = len(re.findall(_PERCENT_PATTERN, text, re.IGNORECASE))
    date_count = len(re.findall(_DATE_PATTERN, text))
    price_count = len(re.findall(_PRICE_PATTERN, text, re.IGNORECASE))
    temp_count = len(re.findall(_TEMP_PATTERN, text, re.IGNORECASE))
    distance_count = len(re.findall(_DISTANCE_PATTERN, text, re.IGNORECASE))
    weight_count = len(re.findall(_WEIGHT_PATTERN, text, re.IGNORECASE))
    generic_number_count = len(re.findall(_GENERIC_NUMBER_PATTERN, text))

    # 数字总分（百分比、日期、温度、距离、权重更高）
    number_total = percent_count * 3 + date_count * 3 + price_count * 2 + temp_count * 2 + distance_count * 2 + weight_count * 2 + generic_number_count
    number_density = (number_total / word_count) * 1000

    # 2. 人名数量
    name_count = len(re.findall(_NAME_PATTERN, text))
    name_density = (name_count / word_count) * 1000

    # 3. 具体地点数量
    location_count = len(re.findall(_LOCATION_PATTERN, text, re.IGNORECASE))
    location_density = (location_count / word_count) * 1000

    # 总具体细节密度（加权）
    total_density = number_density * 0.5 + name_density * 0.3 + location_density * 0.2

    # 人类：15-30 per 1k，AI：3-8 per 1k
    # 所以：density 越低 = 越像 AI
    ai_score = max(0.0, min(1.0, 1.0 - (total_density / 20.0)))
    return round(ai_score, 3)


# ============================================================
# 第 19 号引擎：自我引用检测器（Self-Reference Detector）
# 创新角度：人类会说"我认为"、"我觉得"，AI 很少
# ============================================================

_SELF_REFERENCES = [
    # 英文自我引用（按强度分级，精雕细琢扩充）
    # 强自我引用
    "i think", "i believe", "i feel", "i mean", "in my opinion",
    "from my perspective", "personally", "i would say",
    "in my view", "from my point of view", "as far as i'm concerned",
    "i personally", "i personally think", "i personally believe",
    "i personally feel", "i personally believe",
    # 中等自我引用
    "i guess", "i suppose", "i imagine", "i suspect",
    "i reckon", "i assume", "i presume", "i gather",
    "i understand", "i believe that", "i think that",
    # 弱自我引用
    "in my experience", "from my experience", "based on my experience",
    "as i see it", "the way i see it", "to my mind",
    "i'd say", "i'd say that", "i would argue",
    # 中文自我引用（精雕细琢扩充）
    "我认为", "我觉得", "在我看来", "我个人", "我觉得吧",
    "我认为吧", "我个人认为", "我觉得应该",
    "我想", "我猜", "我估计", "我感觉",
    "在我看来", "从我来看", "以我的经验",
    "我觉得吧", "我认为吧", "我个人觉得",
    "我想吧", "我猜吧", "我估计吧",
    "依我看", "照我说", "要我说",
]


def _detect_self_references(text: str) -> float:
    """
    自我引用检测器 - 打磨版！
    原理：人类写文章会有很多自我表达（"我认为"、"我觉得"）
    AI 写文章比较客观，很少有自我表达
    所以：自我引用越少 = 越像 AI
    改进：扩充词表、调整归一化
    """
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    self_ref_count = 0
    for phrase in _SELF_REFERENCES:
        self_ref_count += text_lower.count(phrase)

    self_ref_density = (self_ref_count / word_count) * 1000

    # 人类：5-12 per 1k，AI：0-1 per 1k
    # 所以：density 越低 = 越像 AI
    ai_score = max(0.0, min(1.0, 1.0 - (self_ref_density / 8.0)))
    return round(ai_score, 3)


# ============================================================
# 第 20 号引擎：逻辑连接词密度检测器（Logical Connectors Detector）
# 创新角度：AI 用很多 therefore/however，人类用得少
# 打磨版：扩充词表、分类检测
# ============================================================

_LOGICAL_CONNECTORS = [
    # 英文逻辑连接词（按类型分类，精雕细琢扩充）
    # 因果类
    "therefore", "consequently", "thus", "hence", "accordingly",
    "as a result", "for this reason", "because of this",
    "as a consequence", "in consequence", "due to this",
    "owing to this", "thanks to this", "on account of this",
    # 转折类
    "however", "nevertheless", "nonetheless", "on the other hand",
    "in contrast", "conversely", "yet", "but",
    "on the contrary", "in comparison", "by contrast",
    "while", "whereas", "although", "though",
    # 递进类
    "moreover", "furthermore", "additionally", "in addition",
    "besides", "also", "what's more",
    "further", "furthermore", "more importantly",
    "not only that", "on top of that", "as well as",
    # 总结类
    "in conclusion", "in summary", "to summarize", "overall",
    "in short", "in brief", "to sum up", "ultimately",
    "all in all", "all things considered", "in the end",
    "to conclude", "to wrap up", "in the final analysis",
    # 举例类
    "for example", "for instance", "such as",
    "namely", "specifically", "in particular",
    # 顺序类
    "first", "second", "third", "finally",
    "firstly", "secondly", "thirdly", "lastly",
    # 中文逻辑连接词（精雕细琢扩充）
    "因此", "然而", "此外", "而且", "总之", "综上所述",
    "所以", "但是", "另外", "同时", "最后", "综上",
    "因为", "由于", "既然", "既然如此",
    "虽然", "尽管", "即使", "哪怕",
    "不仅", "而且", "不但", "并且",
    "首先", "其次", "然后", "接着",
    "例如", "比如", "诸如", "举例来说",
]


def _detect_logical_connectors(text: str) -> float:
    """
    逻辑连接词密度检测器 - 打磨版！
    原理：AI 写文章会用很多逻辑连接词（therefore, however, moreover）
    人类写文章用得比较少
    所以：逻辑连接词越多 = 越像 AI
    改进：扩充词表、调整归一化
    """
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    connector_count = 0
    for phrase in _LOGICAL_CONNECTORS:
        connector_count += text_lower.count(phrase)

    connector_density = (connector_count / word_count) * 1000

    # AI：10-20 per 1k，人类：3-7 per 1k
    # 所以：density 越高 = 越像 AI
    ai_score = min(connector_density / 15.0, 1.0)
    return round(ai_score, 3)


# ============================================================
# RoBERTa ChatGPT 检测器（逆向自 sloptotal ClassifierChatGPTEngine）
# ============================================================

_roberta_model = None
_roberta_tokenizer = None
_roberta_ai_idx = None


def _load_roberta_model():
    """懒加载 RoBERTa ChatGPT 检测器"""
    global _roberta_model, _roberta_tokenizer, _roberta_ai_idx
    if _roberta_model is None:
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
            print(f"RoBERTa model load failed: {e}")
            return None, None, None
    return _roberta_model, _roberta_tokenizer, _roberta_ai_idx


# ============================================================
# 第 21 号引擎：隐藏字符检测器（Hidden Characters Detector）
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
    # 其他不可见字符（精雕细琢新增）
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
    # 其他隐藏字符（精雕细琢新增）
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
    word_count = max(len(text.split()), 1)

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
# 第 22 号引擎：句首词分布检测器（Sentence Starter Distribution）
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
# 第 23 号引擎：标点符号使用模式检测器（Punctuation Patterns）
# 全新角度！人类和AI的标点使用习惯不同
# ============================================================

def _detect_punctuation_patterns(text: str) -> float:
    """标点使用模式检测器（合并感叹句/疑问句 + 标点符号使用）"""
    word_count = max(len(text.split()), 1)

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
# 第 24 号引擎：段落长度分布检测器（Paragraph Length Distribution）
# 全新角度！AI的段落长度更均匀
# ============================================================

def _detect_paragraph_length_distribution(text: str) -> float:
    """段落长度分布检测器"""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5]
    if len(paragraphs) < 3:
        return 0.5

    # 统计段落长度的变异系数
    import statistics
    para_lengths = [len(p.split()) for p in paragraphs]
    mean_len = statistics.mean(para_lengths)
    stdev = statistics.stdev(para_lengths)
    cv = stdev / mean_len if mean_len > 0 else 0

    # 人类：cv ~0.5-0.8，AI：cv ~0.2-0.4
    uniformity_score = max(0, 1.0 - (cv / 0.6))
    return round(uniformity_score, 3)


# ============================================================
# 第 25 号引擎：感叹句/疑问句比例检测器（Exclamation/Question Ratio）
# 全新角度！人类写作有更多感叹和疑问
# ============================================================

def _detect_exclamation_question_ratio(text: str) -> float:
    """感叹句/疑问句比例检测器"""
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    exclamation_count = sum(1 for s in sentences if "!" in s)
    question_count = sum(1 for s in sentences if "?" in s)
    total = len(sentences)

    # 人类：感叹句+疑问句比例 ~10-20%
    # AI：感叹句+疑问句比例 ~2-5%
    ratio = (exclamation_count + question_count) / total
    # ratio越高 = 越像人
    ai_score = min(max((0.10 - ratio) / 0.08, 0.0), 1.0)
    return round(ai_score, 3)


# ============================================================
# 第 26 号引擎：同义词使用频率检测器（Synonym Usage Frequency）
# 全新角度！AI更喜欢用同义词
# ============================================================

# 常见的同义词对
_SYNONYM_PAIRS = [
    ("good", "great"), ("good", "excellent"), ("good", "wonderful"),
    ("bad", "terrible"), ("bad", "awful"), ("bad", "horrible"),
    ("big", "large"), ("big", "huge"), ("big", "enormous"),
    ("small", "tiny"), ("small", "little"), ("small", "miniature"),
    ("happy", "glad"), ("happy", "joyful"), ("happy", "delighted"),
    ("sad", "unhappy"), ("sad", "depressed"), ("sad", "sorrowful"),
    ("important", "crucial"), ("important", "vital"), ("important", "essential"),
    ("difficult", "hard"), ("difficult", "challenging"), ("difficult", "tough"),
    ("easy", "simple"), ("easy", "effortless"), ("easy", "straightforward"),
    ("fast", "quick"), ("fast", "rapid"), ("fast", "swift"),
]


def _detect_synonym_usage(text: str) -> float:
    """同义词使用频率检测器"""
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    synonym_count = 0
    for word1, word2 in _SYNONYM_PAIRS:
        if word1 in text_lower and word2 in text_lower:
            synonym_count += 1

    # 同义词密度（每100词）
    synonym_density = (synonym_count / word_count) * 100

    # 人类：~1-2 per 100 words，AI：~3-5 per 100 words
    ai_score = min(max((synonym_density - 1.5) / 3.0, 0.0), 1.0)
    return round(ai_score, 3)


# ============================================================
# 第 27 号引擎：长句/短句交替模式检测器（Long/Short Sentence Alternation）
# 全新角度！人类写作有长短句交替
# ============================================================

def _detect_sentence_length_distribution(text: str) -> float:
    """句子长度分布检测器（合并长句/短句交替 + 滑动窗口）"""
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    # 统计每个句子的长度
    lengths = [len(s.split()) for s in sentences]

    # 统计相邻句子的长度差（长句/短句交替）
    diffs = []
    for i in range(len(lengths) - 1):
        diffs.append(abs(lengths[i] - lengths[i+1]))
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    alternation_score = min(max((7.0 - avg_diff) / 5.0, 0.0), 1.0)

    # 统计句子长度的变异系数（滑动窗口）
    import statistics
    mean_len = statistics.mean(lengths)
    stdev = statistics.stdev(lengths)
    cv = stdev / mean_len if mean_len > 0 else 0
    uniformity_score = max(0, 1.0 - (cv / 0.6))

    final_score = alternation_score * 0.5 + uniformity_score * 0.5
    return round(final_score, 3)


# ============================================================
# 第 28 号引擎：PAWN 困惑度注意力加权检测器
# 参考 PAWN 方法：对困惑度进行注意力加权，低困惑度token权重更高
# ============================================================

def _detect_pawn_perplexity(text: str) -> float:
    """PAWN 困惑度注意力加权检测器"""
    import math
    from collections import Counter

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
# 第 29 号引擎：对抗扰动程度检测器（Adversarial Perturbation Detector）
# 参考"Modeling the Attack"思路：计算文本距离"原始AI版本"的扰动程度
# 扰动越小、越"顺滑"，AI嫌疑反而越高
# ============================================================

def _detect_adversarial_perturbation(text: str) -> float:
    """对抗扰动程度检测器"""
    import math
    from collections import Counter

    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    # 1. 检测词汇多样性（扰动会增加词汇多样性）
    word_freq = Counter(words)
    unique_ratio = len(word_freq) / len(words)

    # 2. 检测句子长度方差（扰动会让句子更随机）
    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        sent_lengths = [len(s.split()) for s in sentences]
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
# 第 28 号引擎：重复词检测器（Repeated Words Detector）
# 全新角度！AI写作会重复使用某些词
# ============================================================

def _detect_vocabulary_diversity(text: str) -> float:
    """词汇多样性检测器（合并重复词 + 词汇丰富度）"""
    import math
    words = text.lower().split()
    if len(words) < 20:
        return 0.5

    from collections import Counter
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


# ============================================================
# 第 29 号引擎：数字/百分比使用检测器（Numbers/Percentages Usage）
# 全新角度！AI更喜欢用数字和百分比
# ============================================================

def _detect_specific_details(text: str) -> float:
    """具体细节密度检测器（合并数字/百分比 + 具体细节）"""
    word_count = max(len(text.split()), 1)

    # 数字检测
    numbers = len(re.findall(r'\b\d+\b', text))
    percentages = len(re.findall(r'\b\d+(\.\d+)?%', text))
    number_density = (numbers / word_count) * 1000
    percentage_density = (percentages / word_count) * 1000
    number_score = min(max((number_density - 30) / 40.0, 0.0), 1.0)
    percentage_score = min(max((percentage_density - 5) / 10.0, 0.0), 1.0)

    # 具体细节检测（人名、地名、组织名等）
    # 简单检测：大写字母开头的词
    proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    proper_noun_density = (proper_nouns / word_count) * 1000
    # 人类：~50-100 per 1k words，AI：~20-50 per 1k words
    proper_noun_score = min(max((80 - proper_noun_density) / 60.0, 0.0), 1.0)

    final_score = number_score * 0.3 + percentage_score * 0.2 + proper_noun_score * 0.5
    return round(final_score, 3)


# ============================================================
# 第 30 号引擎：引用模式检测器（Citation Patterns Detector）
# 全新角度！AI的引用方式更公式化
# ============================================================

# AI常用的引用模式
_CITATION_PATTERNS = [
    r'according to \w+ \(\d{4}\)',
    r'\w+ et al\. \(\d{4}\)',
    r'\w+ \(\d{4}\) (found|showed|demonstrated|argued|claimed)',
    r'studies (show|show that|have shown)',
    r'research (indicates|suggests|has shown)',
    r'experts (believe|agree|argue)',
]


def _detect_formulaic_patterns(text: str) -> float:
    """公式化模式检测器（合并引用模式 + 公式化表达）"""
    word_count = max(len(text.split()), 1)

    # 引用模式检测
    citation_count = 0
    for pattern in _CITATION_PATTERNS:
        citation_count += len(re.findall(pattern, text, re.IGNORECASE))
    citation_density = (citation_count / word_count) * 1000
    citation_score = min(max((citation_density - 2.0) / 5.0, 0.0), 1.0)

    # 公式化表达检测（检测AI常用短语）
    formulaic_count = 0
    for phrase in _AI_FORMULAIC_MARKERS:
        formulaic_count += len(re.findall(phrase, text, re.IGNORECASE))
    formulaic_density = (formulaic_count / word_count) * 1000
    formulaic_score = min(max((formulaic_density - 5.0) / 15.0, 0.0), 1.0)

    final_score = citation_score * 0.4 + formulaic_score * 0.6
    return round(final_score, 3)


# ============================================================
# 第 31 号引擎：段落开头模式检测器（Paragraph Opener Patterns）
# 全新角度！AI的段落开头更规律
# ============================================================

# AI常用的段落开头
_PARAGRAPH_OPENERS = [
    "first", "second", "third", "finally", "lastly",
    "however", "therefore", "moreover", "furthermore",
    "in addition", "in contrast", "in conclusion",
    "on the other hand", "on one hand",
    "as a result", "consequently", "thus", "hence",
    "it is", "there are", "this is",
    "one of", "another", "some", "many",
]


def _detect_paragraph_openers(text: str) -> float:
    """段落开头模式检测器"""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5]
    if len(paragraphs) < 3:
        return 0.5

    opener_count = 0
    for para in paragraphs:
        words = para.split()
        if words:
            first_word = words[0].lower().strip(".,!?;:'\"()")
            for opener in _PARAGRAPH_OPENERS:
                if first_word.startswith(opener) or first_word == opener:
                    opener_count += 1
                    break

    opener_ratio = opener_count / len(paragraphs)

    # 人类：~20-30%，AI：~50-70%
    ai_score = min(max((opener_ratio - 0.25) / 0.40, 0.0), 1.0)
    return round(ai_score, 3)


# ============================================================
# 第 32 号引擎：句尾词分布检测器（Sentence Ending Distribution）
# 全新角度！人类写作的句尾词分布很随机，AI写作的句尾词分布很规律
# ============================================================

def _detect_sentence_ending_distribution(text: str) -> float:
    """句尾词分布检测器"""
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    # 统计每个句子的最后一个词
    endings = []
    for sent in sentences:
        words = sent.strip().split()
        if words:
            last_word = words[-1].lower().strip(".,!?;:'\"()")
            endings.append(last_word)

    if not endings:
        return 0.5

    # 统计句尾词的多样性
    from collections import Counter
    ending_freq = Counter(endings)
    diversity = len(ending_freq) / len(endings)

    # 人类：diversity ~0.8-0.9，AI：diversity ~0.6-0.7
    diversity_score = min(max((0.85 - diversity) / 0.25, 0.0), 1.0)

    # 统计重复句尾词的比例
    repeated_endings = sum(count for count in ending_freq.values() if count > 1)
    repeat_ratio = repeated_endings / len(endings)

    repeat_score = min(max((repeat_ratio - 0.15) / 0.25, 0.0), 1.0)

    final_score = diversity_score * 0.5 + repeat_score * 0.5
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


def _split_sentences(text: str) -> list:
    """简单的句子切分"""
    text = re.sub(r'\n+', ' ', text)
    sentences = re.split(r'[.!?。！？]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


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
        prose_lines.append(line)

    return '\n'.join(prose_lines), references


def _compute_perplexity_ngram(text: str, n: int = 3) -> float:
    """基于 n-gram 的简化困惑度计算（打磨版，不需要下载大模型）"""
    words = text.lower().split()
    if len(words) < n + 5:
        return 0.5

    # 构建 n-gram 频率表（打磨：用 trigram 更准）
    from collections import Counter
    ngrams = Counter()
    unigrams = Counter()

    for w in words:
        unigrams[w] += 1
    for i in range(len(words) - n + 1):
        ngrams[tuple(words[i:i+n])] += 1

    # 计算平均负对数似然
    import math
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

    # 归一化（精雕细琢：调整范围）
    # 低困惑度（2-3）= 可预测 = AI 写的
    # 高困惑度（5-7）= 不可预测 = 人写的
    # 我们输出 AI 概率：困惑度越低，AI 概率越高
    ai_prob = max(0.1, min(0.9, (4.5 - avg_log_prob) / 2.0 + 0.3))
    return round(ai_prob, 3)


def _sliding_window_detection(sentences: list, window_size: int = 7) -> list:
    """段落级滑动窗口检测（精雕细琢版）：7句为一个窗口，带上下文"""
    if len(sentences) < window_size:
        return [0.5] * len(sentences)

    window_scores = []
    for i in range(len(sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        window = sentences[start:end]
        window_text = ' '.join(window)

        # 计算这个窗口的 burstiness
        lengths = [len(s.split()) for s in window]
        avg = sum(lengths) / len(lengths)
        std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
        burstiness = std / (avg + 1)

        # 精雕细琢：加了句长均匀度检测
        if avg > 0:
            uniformity = 1 - (std / avg)
        else:
            uniformity = 0

        # 窗口内的 AI 概率（精雕细琢：结合 burstiness 和 uniformity）
        if burstiness < 0.35 and uniformity > 0.65:
            window_ai = 0.75
        elif burstiness < 0.55:
            window_ai = 0.55
        else:
            window_ai = 0.25

        window_scores.append(window_ai)

    return window_scores


def run_ai_detection(text: str) -> AIResult:
    """跑所有可用的 AI 检测引擎，返回综合结果"""
    engine_scores = {}

    # 步骤 0: 自动排除引用/参考文献部分
    prose_text, references = _extract_prose_only(text)
    if len(prose_text.strip()) < 50:
        prose_text = text  # 如果提取后太短，就用原文

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

    # 引擎 26: 具体细节密度检测器（合并数字/百分比 + 具体细节）
    engine_scores["specific-details"] = _detect_specific_details(prose_text)

    # 引擎 27: 公式化模式检测器（合并引用模式 + 公式化表达）
    engine_scores["formulaic-patterns"] = _detect_formulaic_patterns(prose_text)

    # 引擎 28: PAWN 困惑度注意力加权检测器（零样本范式）
    engine_scores["pawn-perplexity"] = _detect_pawn_perplexity(prose_text)

    # 引擎 29: 对抗扰动程度检测器（对抗鲁棒性）
    engine_scores["adversarial-perturbation"] = _detect_adversarial_perturbation(prose_text)

    # 综合分数（加权平均：RoBERTa 权重最高，因为是真正的 ML 分类器）
    # 合并后的权重：强引擎权重更高，弱引擎权重更低，确保不互相冲突
    weights = {
        # 强引擎（ML分类器、成熟工具）- 权重最高
        "roberta-chatgpt": 0.25,
        "lmscan": 0.15,
        # 中等引擎（创新引擎、老牌引擎）
        "human-imperfection": 0.08,
        "emotion-curve": 0.05,
        "specific-details": 0.05,
        "self-references": 0.04,
        "logical-connectors": 0.04,
        "linguistic-markers": 0.06,
        "structural-patterns": 0.04,
        # 合并后的引擎
        "starter-patterns": 0.03,
        "punctuation-patterns": 0.02,
        "sentence-length": 0.02,
        "vocabulary-diversity": 0.02,
        "formulaic-patterns": 0.05,
        # 弱引擎（小指标）
        "hidden-characters": 0.01,
        "formulaic-markers": 0.01,
        "sentiment-hedging": 0.01,
        "transition-density": 0.01,
        "compressibility": 0.01,
        "structured-shape": 0.01,
        "ngram-perplexity": 0.03,
        "pawn-perplexity": 0.04,  # PAWN 零样本引擎，权重稍高
        "adversarial-perturbation": 0.03,  # 对抗扰动检测器
        "burstiness-stat": 0.01,
        "statistical-baseline": 0.01,
    }
    total_weight = 0
    weighted_sum = 0
    for name, score in engine_scores.items():
        w = weights.get(name, 0.1)
        weighted_sum += score * w
        total_weight += w
    final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    # 判定
    if final_score > 0.6:
        label = "ai"
    elif final_score < 0.4:
        label = "human"
    else:
        label = "mixed"

    # 用滑动窗口的逐句分数做高风险句
    flagged_sentences = []
    for i, s in enumerate(sentences):
        if i < len(window_scores) and window_scores[i] > 0.5:
            flagged_sentences.append({
                "text": s[:100],
                "score": round(window_scores[i], 3)
            })
    flagged_sentences.sort(key=lambda x: x["score"], reverse=True)

    return AIResult(
        ai_probability=round(final_score, 4),
        label=label,
        method=f"ensemble ({len(engine_scores)} engines, references excluded)",
        engine_scores=engine_scores,
        flagged_sentences=flagged_sentences[:10],
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

def run_citation_check(text: str) -> CitationResult:
    """提取引用并调用 onecite 验证"""
    # 提取 DOI
    dois = re.findall(r'10\.\d{4,}/[^\s\]>\)]+', text)
    dois = list(set(dois))  # 去重

    if not dois:
        return CitationResult(total_extracted=0, resolved=0)

    resolved = 0
    failed = []
    bibtex_entries = []

    for doi in dois[:20]:  # 最多处理 20 条
        try:
            result = subprocess.run(
                ["onecite", "process", doi, "--quiet"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                resolved += 1
                bibtex_entries.append(result.stdout.strip())
            else:
                failed.append(doi)
        except Exception:
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

    if not text or len(text.strip()) < 50:
        return CheckReport(text_length=0)

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
    if len(sys.argv) < 2:
        print("用法:")
        print("  AI 检测: python aurora_check.py <文件路径>")
        print("  文档比对: python aurora_check.py compare <当前文档> <参考文档>")
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
    print(report.to_markdown())

    # 生成 HTML 报告
    base_name = os.path.splitext(sys.argv[1])[0]
    html_path = base_name + "_aurora_report.html"
    report.to_html(output_path=html_path)
    print(f"\nHTML 报告已生成: {html_path}")
