"""Aurora Check — core detection module (optimized edition, v3.23).

Usage (Python):

    from aurora_check import run_check
    report = run_check(text="Your article text")
    print(report.to_markdown())

Usage (CLI):

    python aurora_check.py <file> [--json] [--no-html]
    python aurora_check.py compare <doc_a> <doc_b>

Design principle:
    Only emit conclusions that the evidence can support.
    When the text is too short, too repetitive, or not natural language,
    this module refuses to judge rather than producing a deceptively
    precise score.

v3.16 优化摘要（功能与 API 完全兼容）:
  1. 热路径正则全部预编译为模块级常量。
  2. ENGINE_SPEC 基础注册表移至模块级；中文自适应变体按语言缓存。
  3. 修复 validate_text 中词汇多样性检查重复执行的 bug。
  4. 新增分层异常体系（AuroraError 及子类）。
  5. 语言检测结果在检测主流程内复用传递。
  6. 大词表匹配改为预编译 Pattern 列表。

v3.17 新增:
  7. 模型生命周期管理：空闲超过 AURORA_MODEL_TTL（默认 1800 秒）
     自动回收（del + gc.collect + torch.cuda.empty_cache），释放后
     懒加载标志复位，下次访问自动重载；推理期用 in_use 守卫防止
     回收线程动正在使用的模型；加载加锁防止并发重复加载。
  8. _detect_mlm_perplexity 批量推理（AURORA_PPL_BATCH，默认 8）。
  9. 公开接口 release_idle_models(force=True) 可手动强制回收。

v3.18（Code Review 修复）:
 10. 【P0】_detect_mlm_perplexity 批量窗口构造补上 mask：
     目标 token 必须替换成 [MASK]，否则模型"看着答案做题"，
     CE≈0、pseudo_pp≈1、引擎恒定输出 ≈0.3 中性噪声，彻底失效。
 11. 【P1-1】三个 _load_* 改为「失败即粘性」：加载尝试过一次后
     （无论成败）直接返回当前结果，不再每次重跑 2.5s 网络探测。
     需要重试时调用 release_idle_models(force=True) 或重启进程。
 12. 【P1-2】MLM snippet 改为按字符截断再 tokenize：中文没有
     空格，原先 split 后 words[:400] 对中文覆盖失真、门槛过严。
 13. 【P1-3】to_html 中用户原文 / 策略 / 改写建议一律 html.escape，
     杜绝本地 XSS（开源仓库必备）。
 14. 【P2-1】权重 0 的引擎直接跳过计算（不再白算），并补登记进
     「已弃用引擎」报告面板。
 15. 【P2-2】_detect_specific_details 删除从未使用的 lang 死参数。
 16. 【P2-3】情感曲线否定判断先剥标点（"don't," 此前匹配不上
     "don't"）。
 17. 【P2-4】_detect_language 改为拉丁字母数 vs 汉字数公平对比
     （原版拿汉字数比英文单词数，把英文打了 4~5 折）。
 18. 【P2-5 修正】LRU 缓存 key 直接就是原文 text：str 自带缓存
     哈希，无需再算 SHA-256。初版把 sha256(text) 和原文一起塞进
     key，既没省 hash 又让缓存 key 更长，未省反费。
 19. 【P1-1 配套】release_idle_models(force=True) 补上三个懒
     加载标志位（_ml_load_done / _mlm_load_done / _PPL_TRIED）
     的硬重置：「失败即粘性」下加载失败不注册 release 回调，
     原 force=True 触发不了重载；现在是断网→联网后重新加载
     模型的唯一入口。

v3.19（Code Review 修复，功能与 API 完全兼容）:
 20. 【P1】_detect_with_ml 中文分块修复：原先 text.split() 按空白
     切词，中文整段是「一个词」，全文被塞进单块再被 512 token
     截断——长中文文档只检测了开头。现按语言自适应：英文按
     80 词分块，中文按 160 字符分块（ModernBERT 512 token ≈
     中文 ~500 字，留余量），逐块独立打分取均值。
 21. 【P2-1】run_citation_check DOI 解析并行化（5 并发）：
     20 个 DOI 串行最坏 20×(15s+12s)≈9 分钟，压到 ~1/5 墙钟；
     解析顺序与输出格式不变。
 22. 【P2-2】compare_with_reference 预计算参考句 3-gram 集合：
     原先 O(n×m) 次重复分词建集 → 只建一次，长文档比对提速
     约 2 倍，结果完全一致。
 23. 【P2-3】_detect_emotion_curve 情感词匹配改为单条预编译
     alternation 正则：去掉「逐 token × 逐词表」双层子串扫描，
     长文明显提速，计数语义完全不变（仍是子串命中即计 1）。
 24. 【P3-1】to_markdown BibTeX 输出补上 ```bibtex 代码围栏
     （原版反引号丢失，"bibtex" 渲染成裸文本）。
 25. 【P3-2】清理：移除未使用的 _AI_CHINESE_MARKERS 导入；删除
     _load_ml_models 里多余的 global _ml_models（该函数只原地
     变更 dict，从不重绑定）；_STARTER_INDEX 类型标注改为实际的
     Dict[str, List[str]]。

v3.20（外部评审修复：P0 全部 + P1 6 项 + P2 3 项）:
 26. 【P0-1】外部查重不再谎报 0.0：Crossref query.bibliographic 是
     元数据级模糊匹配而非全文索引，"未命中"≠"无抄袭"。此前无命中
     返回 0.0 会被渲染成"外部相似度 0.0%"，误导用户。现返回 None
     + 说明（含专业工具建议），渲染层走"未完成"路径如实展示。
 27. 【P0-2】LRU 缓存感知模型生命周期：release_idle_models() 现在
     同步清空引擎缓存——修复"断网时缓存的旧结论在模型重载后仍被
     返回"的问题。
 28. 【P0-3】中文恒定假信号治理：emotion-curve 在词表零命中（如
     中文文本）时返回中性 0.5 而非恒 1.0；emotion-curve /
     specific-details / structural-patterns 加入 _EN_ONLY_ENGINES，
     中文下降权 ×0.2。
 29. 【P0-4】改写建议语法修复：utilized→used（原 use）、
     fostered→helped（原 help）等时态/数形态丢失，通过
     _verb_map / 后缀映射表保持语法正确。
 30. 【P0-5】相似度比对支持中文：_text_ngrams 与 Jaccard 相似度对
     CJK 文本改用字符 bigram 切分——中文文档 compare_with_reference
     与段落自重叠从"恒 0"变为有效读数。
 31. 【P1-6】报告补 missing_engines 展示：to_markdown / to_html
     新增"本次未参与引擎"区块（名称 + 原因），透明度补全。
 32. 【P1-7】HTML gauge 颜色断点对齐 label 阈值 0.33/0.58——
     修复"绿区指针 + mixed 判定"的认知冲突。
 33. 【P1-8】ML 模型 ai_idx 兜底改语义推断：无 ai 关键词时先找
     human 同义词并取另一索引，避免 label 顺序颠倒时方向反转。
 34. 【P1-9】引擎计算与聚合分层缓存：引擎层按 text 缓存（冻结
     元组防篡改），normal+strict 同文档不再重复跑 ML 前向；
     聚合层每次新建，调用方拿到的结果天然独占，无需 deepcopy。
 35. 【P1-10】_hf_model_cached 支持 HF_HOME / HF_HUB_CACHE——
     自定义缓存目录的用户不再被误判"无缓存→粘性失败"。
 36. 【P1-11】中文重复粘贴校验改为 5-gram 重复率（>0.3 拒绝）：
     原二元组多样性 <0.05 阈值在中文下形同虚设。
 37. 【P2-1】删除 10 个权重 0 引擎的实现（约 200 行死代码）：
     structured-shape / structural-patterns / self-references /
     logical-connectors / hidden-characters / punctuation-patterns /
     statistical-watermark / spec-detect / dna-detect /
     vocabulary-diversity。ENGINE_SPEC 条目保留，「已弃用引擎」
     面板仍如实列出名称与弃用原因。
 38. 【P2-2】_cosine_sim 更名 _jaccard_sim（它算的是 Jaccard 不是
     余弦），并让段落自重叠复用 _text_ngrams（自动获得中文支持）。
 39. 【P2-3】_detect_with_baseline 删掉从未被消费的 flagged 死数据。
 40. 【P2-4】PPV 校准参数补注：TPR=0.90 的来源与 strict 模式下
     FPR 上界失效的局限。

v3.21（评审二轮修复）:
 41. 【P0】_detect_mlm_perplexity 的 import torch 移到模型判空
     之后：此前位于 _load_mlm_model() 之前，torch 未安装时
     ImportError 直接冒泡到 run_check，整条管线崩溃——优雅降级
     根本来不及执行。纯统计用户 / CI / slim 镜像一跑就炸。
     （v3.20 曾误标该行为「故意提前导入」，实为遗留 bug。）
 42. 【P1】_is_front_matter_line 改用 _RE_SENT_END_CJK（含
     。！？…）：硬换行、每行 <60 字、以「。」结尾的中文正文不再
     被 _extract_prose_only 当封面元数据整体清空；compare_with_reference
     对此类中文文档恢复有效相似度。
 43. 【P1】引擎层缓存快照并重放 gap 登记：命中缓存时 lmscan 未
     安装等非持久 gap 仍会出现在 missing_engines 里，透明度不再
     因缓存命中打折。

v3.22（本轮性能优化，分数与 v3.21 逐位一致）:
 44. 【性能】_detect_mlm_perplexity 批量前向默认 8 → 16：同进程 A/B
     实测 810 词文档 5.12s → 3.81s（30 → 3.40s，但峰值 logits 单块
     384MB，低配机器不友好，故取 16）。
 45. 【性能】_detect_with_ml 分块批量前向：80 词块逐条前向改为一轮
     8 块（AURORA_ML_BATCH），810 词 0.90s → 0.28s；padding 位置带
     attention_mask=0，每块分数与逐条版本一致。
 46. 【性能】_split_sentences / _word_count / _detect_language 记忆化：
     单次检测内被 17 / 20+ / 3 次调用，切句对外仍返回 list。
 47. 【清理】删除 16 处多文件拼装遗留的 `# === PART x/y ===` 标记
     （编号本身 1/6…8/8 前后不一致，属拼接残留）。
 48. 【安全】to_markdown 补回用户文本转义：高风险句原文 / 策略 /
     改写建议 / 外部匹配句与标题，与 to_html 的 P1-3 修复对齐。
 49. 【测试】契约测试改用 AST 数 return 元组项数，适配「引擎读数
     4 项 + 分层缓存 5 项（含 gap 快照）」的现状，跨行 return 也能数准。

v3.23（合并任务书 P1/P2/P3：中文长文盲区 + 中文假信号）:
 50. 【P1-A】_detect_lm_perplexity 分块重做：取样从前 1200 字符单窗口
     改为 tokenize 到 4096、按 500 token 切块、stride 450（50 重叠），
     块 CE loss 取几何平均。权重 0.22 的引擎原本看不见长文后半段。
     阈值 pivot=25 / scale=6.5 不变（单块口径与标定一致）；仅中文
     启用，英文路径不受影响（实测七篇英文文档逐位零漂移）。
 51. 【P1-B】validate_text 数句改用宽松口径：切句函数加 min_len
     参数（默认 10 = 打分口径），句数校验传 4。原先中文 7~10 字的
     完整短句被丢，正文完整却报「完整句子仅 N 句」而拒检。
     注：**没有**把打分侧阈值全局降为 6——实测那样会让 AI 中文
     样例掉 10 个百分点以上（无真人中文基线可校准，方向错误）。
 52. 【P1-F】_detect_vocabulary_richness 中文兜底：该引擎全部指标
     建立在英文词形上，中文经 _RE_WORD 恒 < 20 词 → 旧版 return 0.0
     （= 强「人写」方向性假信号，把中文综合分往下拽）。改为记 gap
     + 返回中性 0.5，报告如实显示"该引擎未参与"。
 53. 【P1-D】CLI --strict：__main__ 识别 --strict 传 run_check(mode=)，
     用法说明同步。此前严苛模式只能用 Python API 调。
 54. 【P2-C】_check_external_plagiarism_api 中文早退：Crossref 是英文
     元数据库，且下面用 len(s.split()) > 15 挑长句——中文无空格，
     整句只算 1 词永远过不了，必然走到"无足够长句"跳过。改为直接
     返回 None + 方法学说明（指引知网/万方），不伪装成"查过且干净"。
 55. 【P2-E】_word_count 缓存容量 64 → 512：key 多为短句，64 条在
     长文里反复淘汰，命中率不足。
 56. 【P2-G】_detect_human_imperfection 中文分支：另四个信号（过渡词
     密度 / 实词重复 / 模糊限定语 / 句首多样性）在中文下恒"零命中"，
     而零命中被翻译成 1.0（高 AI 嫌疑），四个假信号把中文钉在 0.48
     以上。改为把它们填中性 0.5 后按原权重合成（0.30*cv_ai + 0.35），
     保留句长信号本来的 0.30 份额——而不是让 cv 独占 1.0 权重。
 57. 【P2-H】段落阈值语言自适应（_para_min_len）：中文 30 / 英文 50。
     50 字符对中文已是一个半段落，正常中文段被整段滤掉后自重叠检测
     形同虚设。主切分与 4 句切块兜底两处统一改用该阈值。
 58. 【P3-J】删除无人引用的 _RE_SENT_END（ASCII 版）：全文仅注释
     提及，_is_front_matter_line 已改用 _RE_SENT_END_CJK。
 59. 【P3-K】_thaw_stage docstring 补副作用与参数/返回说明：该函数会
     把缓存快照里的 gap 重放进当前 ContextVar，调用方需处于
     _ENGINE_GAPS 上下文，否则 missing_engines 可能串档。
 60. 【P3-L】conflicting_engines 改用 adopted 的归一后分数：原先在
     全量 engine_scores 里取极值，会把权重 0 / 未注册的引擎当成
     冲突方；且未按 direction 归一，-1 方向的 0.1 与 +1 方向的 0.9
     本是同一结论却被判成"分歧 0.8"。

本轮合并说明：以 v3.22（批量前向 + to_markdown 转义）为基线，并入
外部实现的任务书项 A/C/G/H，但其中 B 与 G 按实测证据改写了做法
（见 51、56 的理由），并在合并时挡回三项回退：to_markdown 转义被
删、MLM 批量 16 退回 8、ML 检测器批量前向被移除。

口径备注（review 常问，先写在这里）：
 - human-imperfection 在中文下的输出区间是 [0.35, 0.65]，不是 [0.05, 0.95]。
   公式为 0.30*cv_ai + 0.70*0.5：四个英文词表信号在中文里恒零命中，
   被替换成中性 0.5，只剩句长 CV 那条 0.30 权重的支路在动。这是
   "零假信号优先"的主动取舍，不是 bug；提高中文区分度必须先有真人
   中文基线，否则只是把噪声放大。
 - _split_sentences_cached 的 lru_cache 上限从 64 提到 128：v3.23 给它
   加了 min_len 参数，key 空间由 {text} 变成 {(text, 4), (text, 10), ...}，
   同一篇文本在打分与校验两条路径下各占一个槽位。128 ≈ 单篇文档的
   2 倍句子缓存，够覆盖一次检测里所有分块文本。

已知未处理（下一轮候选）：
 - 单文件拆分 / CLI argparse（结构级重构，建议单独排期）
 - lm-perplexity 上限 4096 token（约 4000 汉字）：text[:16000] 后再按
   4096 截断，超长文档的后段仍是盲区。相对 v3.16 的 500 token 单窗口
   已是 8 倍，但上万字长文要真正覆盖需要滑窗跨批或分段续算。
 - 中文真人基线仍缺：P1-B / P2-G 的阈值只能靠 AI 样例与逻辑
   推导确定，补齐真人中文样本后可重标定
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import zlib
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from html import escape as _hesc
from typing import Dict, List, NamedTuple, Optional, Tuple

from dictionaries import (
    _AI_FORMULAIC_MARKERS,
    _TRANSITION_WORDS,
    _FILLER_PATTERNS,
    _AI_MARKERS_WEIGHTED,
    _HEDGING_PHRASES,
    _BALANCE_PATTERNS,
    _POSITIVE_WORDS,
    _NEGATIVE_WORDS,
    _ZH_TEMPLATE,
)

# ------------------------------------------------------------------
# HuggingFace 默认短超时（未联网时避免 1→2→4→8→16s 退避重试卡 2 分钟）。
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "4")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_VERBOSE = os.environ.get("AURORA_VERBOSE", "0") == "1"

__all__ = [
    "AuroraError", "ModelLoadError", "ExtractionError",
    "EngineSpec", "AIResult", "PlagiarismResult", "CitationResult",
    "CheckReport", "run_check", "run_ai_detection",
    "run_plagiarism_check", "run_citation_check",
    "compare_with_reference", "extract_text_from_file",
    "release_idle_models",
]


# ============================================================
# 分层异常：替代裸 except 静默吞错
# ============================================================
class AuroraError(Exception):
    """Aurora Check 基础异常。"""


class ExtractionError(AuroraError):
    """文件读取或文本提取失败。"""


class ModelLoadError(AuroraError):
    """模型 / 依赖加载失败。"""


# ============================================================
# 预编译正则：热路径模式只编译一次
# ============================================================
_RE_NL = re.compile(r"\n+")
_RE_ABBREV = re.compile(
    r"(?i)\b(?:e\.g\.|i\.e\.|et al\.|dr\.|mr\.|ms\.|vs\.|etc\.)")
_RE_SENTENCE = re.compile(r"[^.!?。！？]+[.!?。！？]*")
_RE_ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_RE_NON_ZH = re.compile(r"[^\u4e00-\u9fff]")
_RE_WORD = re.compile(r"[a-zA-Z]+")
_RE_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
# v3.21【P1】中文句末标点：早期的 ASCII 版正则只认 .!?，硬换行、每行
# <60 字、以「。」结尾的中文正文会被 _is_front_matter_line 误判成
# 封面元数据，_extract_prose_only 把正文整体清空（compare 路径无
# <50 兜底 → 相似度恒 0）。front-matter 判定统一用 CJK 感知版。
_RE_SENT_END_CJK = re.compile(r'[.!?。！？…]\s*$')
_RE_DOI_URL = re.compile(r"^https?://doi\.org", re.IGNORECASE)
_RE_DOI_PREFIX = re.compile(r"^doi:", re.IGNORECASE)
_RE_REF_DOI = re.compile(r"https?://doi\.org|doi:\s*10\.", re.I)
_RE_REF_APA = re.compile(r"^[A-Z][\w'\-]+,\s+[A-Za-z]\.")
_RE_REF_YEAR = re.compile(r"\(\d{4}[a-z]?\)")
_RE_REF_VOL = re.compile(r"\b\d+\s*\(\d+\)\s*,\s*\d+\s*[-–]\s*\d+")
_RE_PROPER = re.compile(r"^[A-Z][a-z]{2,}$")
_RE_QUOTED = re.compile(r"[\"“][^\"”]{8,80}[\"”]")
_RE_ZH_QUOTE = re.compile(r"[“”「」『』]")
_RE_DOIExtract = re.compile(r"10\.\d{4,}/[^\s\]\)>,;]+")
_RE_REF_HEADINGS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r'\n\s*References\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Works\s+Cited\s*\n',
        r'\n\s*Reference\s+List\s*\n',
    ))

_DOT = "\x00DOT\x00"  # 句点哨兵：自然文本中概率为 0


# ============================================================
# 数据结构
# ============================================================
class EngineSpec(NamedTuple):
    """单个引擎的注册信息。"""
    direction: int   # +1 分数高=像 AI，0 表示不参与
    weight: float    # 相对权重
    note: str = ""   # 给人看的说明，不参与计算


@dataclass
class AIResult:
    ai_probability: float
    label: str                    # human / mixed / ai
    method: str
    engine_scores: dict = field(default_factory=dict)
    flagged_sentences: list = field(default_factory=list)
    engine_disagreement: float = 0.0
    confident_engines: list = field(default_factory=list)
    conflicting_engines: list = field(default_factory=list)
    confidence: str = "low"
    adopted_engines: list = field(default_factory=list)
    rejected_engines: list = field(default_factory=list)
    rewrite_hint: str = ""
    ml_detail: dict = field(default_factory=dict)
    ml_disagreement: bool = False
    ml_note: str = ""
    stat_mean: float = 0.0
    primary_score: Optional[float] = None
    aggregation: list = field(default_factory=list)
    missing_engines: list = field(default_factory=list)
    mode: str = "normal"
    student_verdict: str = ""
    high_pressure_warning: str = ""


@dataclass
class PlagiarismResult:
    # None 有三种含义，靠 network_note 区分：未启用 / 断网 / 无足够长句。
    # 写死 0.0 会让调用方误以为「查过外部库、相似度为 0」，名实不符。
    external_similarity: Optional[float] = None
    self_overlap: float = 0.0
    matched_excerpts: list = field(default_factory=list)
    external_matches: list = field(default_factory=list)
    network_note: str = ""


@dataclass
class CitationResult:
    total_extracted: int = 0
    resolved: int = 0
    failed: list = field(default_factory=list)
    bibtex: str = ""


# 实测参数（40 篇人类文本零误判，按 rule of three 取 95% 上界）
# v3.20【P2-4】标定来源说明：
#   _EST_FPR = 0.075 —— 40 篇人类文本 0 误判，rule of three 上界
#             3/40≈0.075，保守可信。
#   _EST_TPR = 0.90 —— 经验值，样本量与构成未记录，待补标定实验。
#   局限：strict 模式故意 over-flag（不剔除离群、阈值更松），实际
#   FPR 会显著高于 0.075，但本函数不区分 mode——严格模式下的 PPV
#   被系统性低估（即对用户偏悲观，方向上是「安全侧」误差）。
_EST_TPR = 0.90
_EST_FPR = 0.075


def _positive_predictive_value(base_rate: float = 0.10) -> float:
    """判为 AI 时，实际确为 AI 的概率（阳性预测值）。

    检测器的"准确率"和"判 AI 时有多可信"是两件事。低基准率会把
    假阳性放大：AI 稿占 10% 时 PPV≈56%，占 5% 时只剩 39%。
    """
    return (base_rate * _EST_TPR) / (base_rate * _EST_TPR + (1 - base_rate) * _EST_FPR)


@dataclass
class CheckReport:
    text_length: int = 0
    note: str = ""  # 输入不合格时的说明（此时 ai 为 None）
    ai: Optional[AIResult] = None
    plagiarism: Optional[PlagiarismResult] = None
    citations: Optional[CitationResult] = None

    def to_dict(self) -> dict:
        """Serialize the report into a plain, JSON-ready dict."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines = ["# 极光查重报告\n"]
        lines.append(f"- **正文长度：** {self.text_length} 字符\n")

        # 输入不合格：只输出警告，不出具任何分数结论
        if self.note:
            lines.append(f"## ⛔ 无法判定\n\n{self.note}\n")
            lines.append("> 文本不满足最低统计要求，任何分数都不可靠，故不出具结论。\n")
            return "\n".join(lines)

        # ---- AI 检测 ----
        if self.ai:
            ai = self.ai
            lines.append("## AI 检测\n")

            if getattr(ai, "high_pressure_warning", ""):
                lines.append(f"### {ai.high_pressure_warning}\n")

            lines.append(f"- **综合判定：** {_label_to_cn(ai.label)}")
            # 「概率」是误导：它是多个信号的加权强度，不是校准过的发生概率
            lines.append(
                f"- **AI 特征强度：** {ai.ai_probability:.1%}"
                f" （多个信号的加权结果，非发生概率）")

            if getattr(ai, "student_verdict", ""):
                lines.append(f"- **给你（学生）的解读：** {ai.student_verdict}")

            lines.append(f"- **检测方法：** {ai.method}\n")

            _conf_cn = {"high": "高", "medium": "中", "low": "低"}.get(
                ai.confidence, ai.confidence)

            if ai.label == "ai":
                _ppv = _positive_predictive_value()
                lines.append(
                    f"- **若据此认定，命中率约：** {_ppv:.0%}"
                    f"（假设你的场景中 AI 稿占 10%）")
                lines.append(
                    f"  - 也就是说：被判 AI 的稿子里，约 {1 - _ppv:.0%} 其实是人写的。"
                    "这一比例随 AI 稿占比下降而急剧恶化——"
                    "若占比 5%，命中率只剩 39%。")
                lines.append("  - 分数高只说明「文本特征像 AI」，不构成抄袭认定。")

            if ai.confidence == "low":
                lines.append("  - ⚠️ 本条置信度为低，模型自身也不确定")
            lines.append("")

            if ai.rewrite_hint:
                lines.append(f"- **⚠️ 改写嫌疑：** {ai.rewrite_hint}")
                lines.append("")

            lines.append(
                f"- **置信度：** {_conf_cn}"
                f"（引擎加权分歧度 {ai.engine_disagreement:.3f}）")
            if ai.confidence == "low":
                lines.append("  - ⚠️ 引擎分歧大，该判定不可靠，建议以人工阅读为准")
            lines.append("")

            if ai.adopted_engines:
                lines.append(
                    f"- **参与加权引擎（{len(ai.adopted_engines)} 个，按影响排序）：**")
                for e in ai.adopted_engines:
                    filled = int(e["score"] * 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    lines.append(
                        f"  - {e['name']}：{e['score']:.1%} {bar} ×{e['weight']}")
                lines.append("")

            if ai.rejected_engines:
                lines.append(
                    f"<details><summary>已弃用引擎（{len(ai.rejected_engines)} 个，"
                    "不参与计算）</summary>\n")
                for e in ai.rejected_engines:
                    lines.append(f"- {e['name']}：{e['reason']}")
                lines.append("\n</details>\n")

            # v3.20【P1-6】透明度补全：missing_engines 一直被认真构造
            # 却从未渲染——用户看不到「本次哪些引擎没参与、为什么」。
            if getattr(ai, "missing_engines", None):
                lines.append(
                    f"<details><summary>本次未参与引擎（{len(ai.missing_engines)}"
                    " 个，含原因）</summary>\n")
                for me in ai.missing_engines:
                    lines.append(
                        f"- {_hesc(str(me['name']), quote=False)}："
                        f"{_hesc(str(me.get('reason', '')), quote=False)}"
                        f"{(' — ' + _hesc(str(me['note']), quote=False)) if me.get('note') else ''}")
                lines.append("\n</details>\n")

            if ai.flagged_sentences:
                _strict_mode = getattr(ai, "mode", "normal") == "strict"
                lines.append("- **高风险句（Top 5）：**")
                for s in ai.flagged_sentences[:5]:
                    icon = {"high": "🟢", "medium": "🟡"}.get(
                        s.get("confidence"), "🔴")
                    # 用户原文 / 策略 / 改写建议一律转义：to_markdown 的
                    # 输出会被嵌进 HTML（server 的报告页），不转义等于
                    # 把用户文本当 HTML 执行（本地 XSS）。与 to_html 的
                    # P1-3 修复保持一致。
                    lines.append(
                        f"  - {icon} ({s['score']:.2f}) "
                        f"{_hesc(s['text'][:80], quote=False)}...")
                    if _strict_mode and s.get("strategy"):
                        lines.append(
                            f"    ↳ 怎么改：{_hesc(s['strategy'], quote=False)}")
                    if _strict_mode and s.get("rewrite_suggestion"):
                        lines.append(
                            f"    ↳ 修改建议：<span style='color:#27ae60'>"
                            f"{_hesc(s['rewrite_suggestion'], quote=False)}"
                            "</span>")
                lines.append("")
                lines.append("  - 🟢 高确信 | 🟡 中等确信 | 🔴 低确信（引擎分歧大）")
                lines.append("")

        # ---- 相似度 ----
        if self.plagiarism:
            p = self.plagiarism
            lines.append("## 相似度\n")
            if p.external_similarity is not None:
                lines.append(
                    f"- **外部文献相似度：** {p.external_similarity:.1%}"
                    "（通过 Crossref 学术库比对）")
                for m in p.external_matches:
                    lines.append(
                        f"  - ⚠️ 疑似句：\"{_hesc(m['sentence'][:70], quote=False)}...\"")
                    lines.append(
                        f"    ↳ 可能出自《{_hesc(m['title'], quote=False)}》 ({m['doi']})")
                if not p.external_matches:
                    lines.append("  - 未发现明显匹配的文献")
            else:
                if p.network_note:
                    lines.append(
                        f"- **外部文献相似度：** 未完成 — {p.network_note}")
                else:
                    lines.append("- **外部文献相似度：** 未启用（联网查重默认关闭）")
            lines.append(f"- **文档内部段落自重叠：** {p.self_overlap:.2%}\n")

        # ---- 引用验证 ----
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
                lines.append("```\n")
                lines.append("\n</details>\n")

        lines.append("---\n_由极光查重 (Aurora Check) 生成_")
        return "\n".join(lines)

    def to_html(self, output_path: str = "") -> str:
        """Render an interactive HTML report (ECharts gauge + radar).

        输入不合格时只输出警告页，不渲染任何分数。
        """
        # 输入不合格：警告页
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
</div></body></html> """
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)
            return html

        ai = self.ai
        ai_prob = ai.ai_probability if ai else 0
        ai_pct = int(round(ai_prob * 100))

        # 与 to_markdown 共用同一套 label 文案
        _LAB = {
            "ai": ("#e74c3c", "AI 特征显著"),
            "mixed": ("#f39c12", "特征混杂"),
            "human": ("#27ae60", "未见明显 AI 特征"),
        }
        _key = ai.label if ai else "human"
        color, label = _LAB.get(_key, ("#f39c12", "混合（人机各半）"))

        student_verdict = getattr(ai, "student_verdict", "") if ai else ""
        high_pressure_warning = getattr(ai, "high_pressure_warning", "") if ai else ""
        _mode = getattr(ai, "mode", "normal") if ai else "normal"

        # 雷达图数据：得分最高的前 5 个引擎
        radar_data = []
        if ai:
            if ai.adopted_engines:
                pool = sorted(ai.adopted_engines,
                              key=lambda e: e.get("score", 0), reverse=True)
            else:
                pool = [{"name": k, "score": v, "note": ""}
                        for k, v in sorted(ai.engine_scores.items(),
                                           key=lambda kv: kv[1], reverse=True)]
            for e in pool[:5]:
                _disp = (e.get("note") or e["name"])
                if len(_disp) > 16:
                    _disp = _disp[:13] + "…"
                radar_data.append({
                    "name": _disp,
                    "value": round(float(e["score"]), 3),
                })
        radar_json = json.dumps(radar_data, ensure_ascii=False)

        # 引擎评分表 + 分歧面板
        engine_rows = ""
        divergence_panel = ""
        if ai and ai.engine_scores:
            scores = list(ai.engine_scores.values())
            avg_score = sum(scores) / len(scores) if scores else 0
            variance = (sum((s - avg_score) ** 2 for s in scores) / len(scores)
                        if scores else 0)
            std_dev = variance ** 0.5
            high_confidence = sum(1 for s in scores if s > 0.7 or s < 0.3)
            consensus = ("高共识" if std_dev < 0.2
                         else "中分歧" if std_dev < 0.35 else "高分歧")
            _adopted = ai.adopted_engines or [
                {"name": k, "score": v, "weight": 0, "note": ""}
                for k, v in ai.engine_scores.items()]

            for e in _adopted:
                name = e.get("note") or e["name"]
                score = e["score"]
                pct = int(round(score * 100))
                bar_color = ("#e74c3c" if score > 0.62
                             else "#f39c12" if score > 0.35 else "#27ae60")
                engine_rows += f"""
<tr>
  <td>{_hesc(str(name), quote=False)}<span style="color:#aaa;font-size:11px;"> ×{e.get('weight', 0)}</span></td>
  <td style="width: 220px;">
    <div style="background: #eee; height: 18px; border-radius: 9px; overflow: hidden;">
      <div style="background: {bar_color}; width: {pct}%; height: 100%;"></div>
    </div>
  </td>
  <td style="text-align: right; font-weight: bold;">{pct}%</td>
</tr>"""

            if ai.rejected_engines:
                _rej = "".join(
                    f'<span style="display:inline-block;margin:2px 6px 2px 0;'
                    f'padding:2px 8px;background:#eee;border-radius:10px;'
                    f'font-size:11px;color:#888;">{_hesc(str(x["name"]), quote=False)}</span>'
                    for x in ai.rejected_engines)
                engine_rows += (
                    f'<tr><td colspan="3" style="padding-top:16px;color:#888;'
                    f'font-size:12px;">已弃用（不参与计算）：<br>{_rej}</td></tr>')

            # v3.20【P1-6】透明度补全：本次未参与引擎及原因
            if getattr(ai, "missing_engines", None):
                _miss = "".join(
                    f'<span style="display:inline-block;margin:2px 6px 2px 0;'
                    f'padding:2px 8px;background:#fff7e6;border-radius:10px;'
                    f'font-size:11px;color:#b8860b;" title='
                    f'"{_hesc(str(x.get("reason", "")), quote=True)}">'
                    f'{_hesc(str(x["name"]), quote=False)}</span>'
                    for x in ai.missing_engines)
                engine_rows += (
                    f'<tr><td colspan="3" style="padding-top:10px;color:#888;'
                    f'font-size:12px;">本次未参与（悬停看原因）：<br>{_miss}</td></tr>')

            divergence_panel = f"""
<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
  <h3 style="font-size: 14px; color: #666; margin-bottom: 10px;">引擎分歧分析</h3>
  <div style="display: flex; gap: 20px; font-size: 13px; flex-wrap: wrap;">
    <div><span style="color: #666;">共识强度:</span>
      <span style="font-weight: bold; color: {color};">{consensus}</span></div>
    <div><span style="color: #666;">标准差:</span>
      <span style="font-weight: bold;">{std_dev:.3f}</span></div>
    <div><span style="color: #666;">高确信引擎:</span>
      <span style="font-weight: bold;">{high_confidence}/{len(scores)}</span></div>
  </div>
</div>"""

        # 高危句：红色下划线 + 悬停气泡
        # 【P1-3】用户原文 / 策略 / 改写建议一律 html 转义，杜绝本地 XSS
        flagged_html = ""
        if ai and ai.flagged_sentences:
            for s in ai.flagged_sentences[:10]:
                score_pct = int(s["score"] * 100)
                strategy = s.get("strategy", "")
                rewrite = s.get("rewrite_suggestion", "")
                esc_text = _hesc(s["text"], quote=False)
                esc_strategy = _hesc(strategy, quote=False) if strategy else ""
                esc_rewrite = _hesc(rewrite, quote=False) if rewrite else ""
                if esc_strategy or esc_rewrite:
                    _parts = []
                    if esc_strategy:
                        _parts.append(f"怎么改：{esc_strategy}")
                    if esc_rewrite:
                        _parts.append(
                            "<span style='color:#7be38a;'>修改建议："
                            f"{esc_rewrite}</span>")
                    bubble = "<br>".join(_parts)
                else:
                    bubble = f"AI 嫌疑度 {score_pct}%，建议人工复核这句"
                rewrite_html = ""
                if esc_rewrite:
                    rewrite_html = (
                        f"<div class='rewrite'>修改建议：{esc_rewrite}</div>")
                flagged_html += f"""
<div class="sentence">
  <span class="s-text"><mark>{esc_text}</mark></span>
  <span class="bubble">{bubble}</span>
  {rewrite_html}
</div>"""
        if not flagged_html:
            flagged_html = '<p style="color: #999;">没有检测到高风险句子</p>'

        _conf_map = {"high": "高", "medium": "中", "low": "低"}
        _conf = _conf_map.get(ai.confidence, "—") if ai else "—"

        _hint_html = ""
        if ai and ai.rewrite_hint:
            _hint_html = (
                '<div style="margin-top:14px;padding:12px 16px;background:#fff8e1;'
                'border-left:4px solid #f39c12;border-radius:6px;font-size:13px;'
                'color:#8a6d3b;text-align:left;">⚠️ 改写嫌疑：'
                f'{_hesc(ai.rewrite_hint, quote=False)}</div>')

        warning_html = ""
        if high_pressure_warning:
            warning_html = (f'<div class="warning-banner">'
                            f'{_hesc(high_pressure_warning, quote=False)}</div>')

        plag_html = ""
        if self.plagiarism:
            p = self.plagiarism
            if p.external_similarity is None and not p.network_note:
                ext = "未启用"
            elif p.external_similarity is not None:
                ext = f"{p.external_similarity:.1%}"
            else:
                ext = f"未完成 — {_hesc(p.network_note, quote=False)}"
            plag_html = f"""
<div class="mini-card">
  <div><span class="mini-label">外部文献相似度</span><span class="mini-val">{ext}</span></div>
  <div><span class="mini-label">内部段落自重叠</span><span class="mini-val">{p.self_overlap:.1%}</span></div>
</div>"""

        cite_html = ""
        if self.citations:
            c = self.citations
            cite_html = f"""
<div class="mini-card">
  <div><span class="mini-label">引用提取</span><span class="mini-val">{c.total_extracted} 条</span></div>
  <div><span class="mini-label">解析成功</span><span class="mini-val">{c.resolved} 条</span></div>
</div>"""

        _adopted_n = len(ai.adopted_engines) if ai else 0
        _scores_n = len(ai.engine_scores) if ai else 0

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>极光查重报告 - Aurora Check</title>
<script>__ECHARTS_INLINE__</script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f0f2f5;
padding: 24px; color: #333; }}
.container {{ max-width: 960px; margin: 0 auto; background: #fff;
border-radius: 16px; box-shadow: 0 4px 30px rgba(0,0,0,0.08); overflow: hidden; }}
.header {{ background: linear-gradient(135deg, #5b6bf5 0%, #8b5cf6 100%);
color: #fff; padding: 36px 40px; text-align: center; }}
.header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px;
letter-spacing: 1px; }}
.header p {{ opacity: 0.9; font-size: 13px; }}
.warning-banner {{ margin: 20px 40px 0; padding: 16px 20px; background: #fff1f0;
border: 1px solid #ffa39e; border-left: 5px solid #e74c3c; border-radius: 8px;
color: #cf1322; font-size: 14px; line-height: 1.7; }}
.score-section {{ padding: 32px 40px 8px; text-align: center; }}
.gauge-wrap {{ width: 100%; height: 230px; }}
.score-label {{ font-size: 20px; font-weight: 600; color: {color}; margin-top: 4px; }}
.verdict {{ display: inline-block; margin-top: 10px; padding: 6px 18px;
border-radius: 20px; background: {color}; color: #fff; font-size: 14px;
font-weight: 600; }}
.meta {{ margin-top: 12px; font-size: 12px; color: #999; }}
.section {{ padding: 26px 40px; border-top: 1px solid #f0f0f0; }}
.section h2 {{ font-size: 18px; margin-bottom: 18px; color: #222; display: flex;
align-items: center; gap: 8px; }}
.section h2::before {{ content: ''; display: inline-block; width: 4px;
height: 18px; background: linear-gradient(180deg, #5b6bf5, #8b5cf6);
border-radius: 2px; }}
.radar-wrap {{ width: 100%; height: 340px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f0f0f0;
font-size: 13px; }}
th {{ font-weight: 600; color: #888; }}
.sentence {{ position: relative; display: block; padding: 10px 12px; margin: 8px 0;
background: #fffbfb; border-left: 4px solid {color}; border-radius: 6px; }}
.s-text mark {{ background: rgba(231, 76, 60, 0.16); color: #c0392b;
padding: 0 2px; text-decoration: underline; text-decoration-color: #e74c3c;
text-decoration-thickness: 2px; text-underline-offset: 3px;
word-break: break-word; line-height: 1.8; }}
.bubble {{ display: none; position: absolute; left: 12px;
bottom: calc(100% + 8px); max-width: 420px; background: #333; color: #fff;
padding: 8px 12px; border-radius: 8px; font-size: 12px; line-height: 1.5;
z-index: 10; box-shadow: 0 4px 16px rgba(0,0,0,0.2); }}
.bubble::after {{ content: ''; position: absolute; top: 100%; left: 20px;
border: 6px solid transparent; border-top-color: #333; }}
.sentence:hover .bubble {{ display: block; }}
.rewrite {{ color: #27ae60; font-size: 12px; margin-top: 6px; line-height: 1.5; }}
.mini-card {{ display: flex; gap: 40px; flex-wrap: wrap; font-size: 13px; }}
.mini-label {{ color: #888; margin-right: 8px; }}
.mini-val {{ font-weight: 700; color: #222; }}
.footer {{ padding: 20px 40px; background: #fafafa; text-align: center;
color: #aaa; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>极光查重报告</h1>
    <p>Aurora Check · AI 文本检测{'（严苛找茬模式）' if _mode == 'strict' else ''}</p>
  </div>
  {warning_html}
  <div class="score-section">
    <div id="gauge" class="gauge-wrap"></div>
    <div class="score-label">{label}</div>
    {f'<div class="verdict">{_hesc(student_verdict, quote=False)}</div>' if student_verdict else ''}
    <div class="meta">
      特征强度非发生概率，不构成抄袭认定 · 置信度 {_conf} · 采纳引擎 {_adopted_n}/{_scores_n}
    </div>
    {_hint_html}
    <p style="margin-top: 10px; color: #bbb; font-size: 12px;">正文长度: {self.text_length} 字符</p>
  </div>
  <div class="section">
    <h2>引擎得分雷达图（前 5 项）</h2>
    <div id="radar" class="radar-wrap"></div>
  </div>
  <div class="section">
    <h2>各引擎详细评分</h2>
    {divergence_panel}
    <table>
      <thead><tr><th>引擎名称</th><th>评分</th>
      <th style="text-align: right;">百分比</th></tr></thead>
      <tbody>{engine_rows}</tbody>
    </table>
  </div>
  <div class="section">
    <h2>高风险句（悬停查看怎么改）</h2>
    {flagged_html}
  </div>
  <div class="section">
    <h2>相似度与引用</h2>
    {plag_html}{cite_html}
  </div>
  <div class="footer">
    由极光查重 (Aurora Check) 生成 · {_adopted_n} 个引擎参与加权
  </div>
</div>
<script>
(function () {{
  var gauge = echarts.init(document.getElementById('gauge'));
  gauge.setOption({{
    series: [{{
      type: 'gauge', min: 0, max: 100, startAngle: 210, endAngle: -30,
      pointer: {{ show: true, length: '58%', width: 5,
        itemStyle: {{ color: '#666' }} }},
      progress: {{ show: true, width: 20, roundCap: true,
        itemStyle: {{ color: '{color}' }} }},
      axisLine: {{ lineStyle: {{ width: 20,
        color: [[0.33, '#27ae60'], [0.58, '#f39c12'], [1, '#e74c3c']] }} }},
      axisTick: {{ show: false }}, splitLine: {{ show: false }},
      axisLabel: {{ show: false }},
      detail: {{ formatter: '{{value}}%', fontSize: 44, fontWeight: 700,
        color: '{color}', offsetCenter: [0, '12%'] }},
      data: [{{ value: {ai_pct} }}]
    }}]
  }});
  var radarData = {radar_json};
  var radar = echarts.init(document.getElementById('radar'));
  radar.setOption({{
    radar: {{
      indicator: radarData.map(function (d) {{
        return {{ name: d.name, max: 1 }};
      }}),
      shape: 'circle', radius: '68%', splitNumber: 4,
      axisName: {{ color: '#666', fontSize: 12, overflow: 'truncate', width: 80 }},
      splitArea: {{ areaStyle: {{ color: ['rgba(91,107,245,0.02)',
        'rgba(91,107,245,0.05)'] }} }},
      splitLine: {{ lineStyle: {{ color: 'rgba(0,0,0,0.07)' }} }},
      axisLine: {{ lineStyle: {{ color: 'rgba(0,0,0,0.1)' }} }}
    }},
    series: [{{
      type: 'radar',
      data: [{{
        value: radarData.map(function (d) {{ return d.value; }}),
        name: 'AI 特征强度',
        areaStyle: {{ color: 'rgba(231,76,60,0.28)' }},
        lineStyle: {{ color: '{color}', width: 2 }},
        itemStyle: {{ color: '{color}' }},
        symbol: 'circle', symbolSize: 6
      }}]
    }}]
  }});
  window.addEventListener('resize', function () {{
    gauge.resize(); radar.resize();
  }});
}})();
</script>
</body>
</html> """

        # 离线内联 ECharts；读不到本地文件时降级回 CDN
        _echarts_js = _read_echarts_inline()
        if _echarts_js:
            html = html.replace("__ECHARTS_INLINE__", _echarts_js)
        else:
            html = html.replace(
                "<script>__ECHARTS_INLINE__</script>",
                '<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0'
                '/dist/echarts.min.js"></script>')

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


_ECHARTS_INLINE_CACHE = None


def _read_echarts_inline() -> str:
    """读取本地 static/echarts.min.js 用于内联（带内存缓存）。"""
    global _ECHARTS_INLINE_CACHE
    if _ECHARTS_INLINE_CACHE:
        return _ECHARTS_INLINE_CACHE
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "static", "echarts.min.js")
    try:
        with open(_path, encoding="utf-8") as f:
            _ECHARTS_INLINE_CACHE = f.read()
    except OSError:
        _ECHARTS_INLINE_CACHE = ""
    return _ECHARTS_INLINE_CACHE


def _label_to_cn(label: str) -> str:
    return {
        "human": "未见明显 AI 特征",
        "mixed": "特征混杂",
        "ai": "AI 特征显著",
    }.get(label, label)


# ============================================================
# Strict（自检找茬）模式
# ============================================================
_STRICT_ADVANCED_WORDS = (
    "delve", "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "utilize", "leverage", "facilitate",
    "foster", "comprehensive", "crucial", "significant", "subsequently",
    "therefore", "thus", "hence", "notably", "overall", "underscore",
    "highlight", "demonstrate", "elucidate", "paramount", "pivotal",
    "robust", "multifaceted", "holistic",
)

_STRICT_TEMPLATE_STARTERS = (
    "firstly", "secondly", "thirdly", "finally", "in conclusion",
    "in summary", "to sum up", "in this essay", "this essay will",
    "it is important to note", "it is worth noting", "as mentioned above",
    "in my opinion", "from my perspective", "with the development of",
    "as we all know", "it goes without saying", "last but not least",
)

# 预编译高级词汇边界匹配（strict 句策略里逐词查）
_STRICT_ADVANCED_RES = tuple(
    re.compile(r"\b" + re.escape(w) + r"\b") for w in _STRICT_ADVANCED_WORDS)


def _student_verdict(score: float, label: str) -> str:
    """把机器分翻译成学生听得懂的五级判定（自检视角）。

    阈值刻意比 normal 三档更早拉响警报：这是"排雷工具"，
    宁可把真人误判成危险（去改），也不能漏掉 Turnitin 会抓的。
    """
    if score < 0.20:
        return "安全（你自己写的）"
    if score < 0.33:
        return "基本安全（轻微润色，别再让 AI 多改）"
    if score < 0.45:
        return "危险（会被老师怀疑用了 AI）"
    if score < 0.60:
        return "高危（重度辅助痕迹，老师大概率怀疑）"
    return "致命（准备好重写吧）"


def _strict_sentence_strategy(sentence: str, text: str,
                              engine_scores: dict) -> str:
    """给一条被标红的高风险句生成"怎么改"策略（strict 专属）。

    优先级：高级词汇 > 模板句 > 长难句 > 段落节奏。
    返回空串表示这条不需要额外策略。
    """
    low = sentence.lower()
    # 1) 高级词汇：Turnitin 高频抓的"假学术腔"
    for pat, w in zip(_STRICT_ADVANCED_RES, _STRICT_ADVANCED_WORDS):
        if pat.search(low):
            return f"含高级词汇「{w}」，这是 Turnitin 高频抓的假学术腔，请换成大白话。"
    # 2) 模板句式
    for st in _STRICT_TEMPLATE_STARTERS:
        if low.startswith(st):
            return "这句话太像四六级模板，请用大白话重写。"
    # 3) 长难句
    if _word_count(sentence) >= 22:
        return "这个长难句像机器堆砌，请故意拆成两三个短句。"
    # 4) 段落节奏
    pr = engine_scores.get("paragraph-rhythm")
    if pr is not None and pr >= 0.6:
        return "这一段长度太完美了，请故意拆成两个短句，制造人类写作的呼吸感。"
    return ""


def _verb_map(present: str, past: str = ""):
    """生成动词词形映射，让替换保持主谓一致。"""
    def _fn(m):
        suf = (m.group(1) or "").lower()
        if suf == "s":
            return present + "s"
        if suf in ("ed", "d"):
            return past or (present + "ed")
        if suf == "ing":
            return present + "ing"
        return present
    return _fn


# 顺序敏感：delve into 必须先于 delve
_REWRITE_SYNONYMS = tuple(
    (re.compile(p), repl) for p, repl in (
        (r"(?i)\bdelve into\b", "talk about"),
        (r"(?i)\bdelve(s|d)?\b", "talk"),
        # v3.20【P0-4】时态/数形态修复：utilized→used（旧版 use 是
        # 病句）、leveraged→used、facilitated→helped、fostered→helped。
        # -e 词干（utiliz/leverag/facilitat）的组值是 e/es/ed/ing，
        # _verb_map 不覆盖，用后缀映射表；foster 的组值 s/ed/ing/空
        # 正好走 _verb_map。
        (r"(?i)\butiliz(e|es|ed|ing)\b",
         lambda m: {"e": "use", "es": "uses",
                    "ed": "used", "ing": "using"}[m.group(1).lower()]),
        (r"(?i)\bleverag(e|es|ed|ing)\b",
         lambda m: {"e": "use", "es": "uses",
                    "ed": "used", "ing": "using"}[m.group(1).lower()]),
        (r"(?i)\bfacilitat(e|es|ed|ing)\b",
         lambda m: {"e": "help", "es": "helps",
                    "ed": "helped", "ing": "helping"}[m.group(1).lower()]),
        (r"(?i)\bfoster(s|ed|ing)?\b", _verb_map("help")),
        (r"(?i)\bcomprehensive\b", "full"),
        (r"(?i)\bcrucial\b", "important"),
        (r"(?i)\bsignificant\b", "important"),
        (r"(?i)\bparamount\b", "very important"),
        (r"(?i)\bpivotal\b", "key"),
        (r"(?i)\brobust\b", "strong"),
        (r"(?i)\bmultifaceted\b", "many-sided"),
        (r"(?i)\bholistic\b", "whole"),
        (r"(?i)\bsubsequently\b", "later"),
        (r"(?i)\bconsequently\b", "so"),
        (r"(?i)\btherefore\b", "so"),
        (r"(?i)\bthus\b", "so"),
        (r"(?i)\bhence\b", "so"),
        (r"(?i)\bmoreover\b", "also"),
        (r"(?i)\bfurthermore\b", "also"),
        (r"(?i)\badditionally\b", "also"),
        (r"(?i)\bnotably\b", "also"),
        (r"(?i)\bnevertheless\b", "but"),
        (r"(?i)\bnonetheless\b", "but"),
        (r"(?i)\boverall\b", "main"),
        (r"(?i)\bunderscore(s|ed|ing|d)?\b", _verb_map("show", "showed")),
        (r"(?i)\bhighlight(s|ed|ing|d)?\b", _verb_map("show", "showed")),
        (r"(?i)\bdemonstrate(s|ed|ing|d)?\b", _verb_map("show", "showed")),
        (r"(?i)\belucidate(s|ed|ing|d)?\b", _verb_map("explain", "explained")),
    ))
_RE_IN_MY_OPINION = re.compile(r"(?i)^in my opinion[,:]?\s*")
_RE_IN_MY_OPINION_HEAD = re.compile(r"(?i)^in my opinion")
_RE_SPLIT_CONJ = re.compile(r"(?i)(?<=\s)(and|but|so)(?=\s)")


def _mock_rewrite_sentence(sentence: str) -> str:
    """Mock 版「AI 降维改写」：纯规则把高危句改成大白话。

    方向永远单向：高级词 → 基础词，绝不反向升级。
    无可改之处时返回空串，调用方据此决定是否展示。
    """
    if not sentence:
        return ""
    s = sentence.strip()
    orig_first_upper = s[:1].isupper()
    changed = False

    # 1) 高级词汇降维
    for pat, repl in _REWRITE_SYNONYMS:
        if pat.search(s):
            s = pat.sub(repl, s)
            changed = True

    # 2) 模板句首
    if _RE_IN_MY_OPINION_HEAD.search(s):
        s = _RE_IN_MY_OPINION.sub("I think ", s)
        changed = True

    # 3) 超 20 词长句：在 and/but/so 或第一个逗号处硬拆
    if _word_count(s) > 20:
        m = _RE_SPLIT_CONJ.split(s, maxsplit=1)
        if len(m) == 3 and m[2].strip():
            left = m[0].rstrip()
            right = f"{m[1]} {m[2].strip()}".strip()
        elif "," in s:
            idx = s.find(",")
            left = s[:idx + 1].rstrip()
            right = s[idx + 1:].strip()
        else:
            left, right = "", ""
        if right and len(left.split()) >= 3:
            left = left.rstrip(",")
            if not left.endswith((".", "!", "?")):
                left += "."
            right = right[0].upper() + right[1:] if right else right
            s = f"{left} {right}".strip()
            changed = True

    if not changed:
        return ""
    # 替换可能把句首高级词变成小写，恢复大写
    if orig_first_upper and s[:1].islower():
        s = s[0].upper() + s[1:]
    return s


# ============================================================
# 真·AI 降维改写（本地 Qwen2.5-0.5b，可选）
# ============================================================
@lru_cache(maxsize=512)
def _llm_rewrite_cached(sentence: str) -> str:
    """实际生成逻辑（被 lru_cache 包裹，异常不缓存）。

    v3.17：推理期用 _model_in_use 守卫，防止回收线程动正在用的模型。
    """
    try:
        gen = ""
        with _model_in_use("ppl"):
            model, tok = _load_ppl_model()
            if model is None or tok is None:
                return _mock_rewrite_sentence(sentence)
            import torch
            prompt = (
                "你是一个英语 90 分的大一新生，请把下面这句高级/模板化的话，"
                "改写得更像口语，不要用高级词汇（如 delve, crucial），"
                "可以拆短句，可以加 I think。只输出改写后的句子。\n\n"
                f"原句：{sentence}\n改写："
            )
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
            ids = enc["input_ids"]
            pad = getattr(tok, "pad_token_id", None) or tok.eos_token_id
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=80, do_sample=False,
                    pad_token_id=pad, eos_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids.shape[1]:],
                             skip_special_tokens=True).strip()
            gen = re.split(r"[\n\r]+", gen)[0].strip().strip('"').strip("'").strip()

        # 防跑偏护栏：自我介绍 / 啰嗦扩写 / 空输出一律回退 Mock
        _intro_markers = (
            "好的", "下面", "以下", "改写后", "译文", "答案是", "可以的",
            "sure", "here", "okay", "ok", "certainly", "of course",
        )
        if not gen:
            return _mock_rewrite_sentence(sentence)
        if len(gen.split()) > len(sentence.split()) * 3:
            return _mock_rewrite_sentence(sentence)
        if any(gen.lower().lstrip().startswith(m) for m in _intro_markers):
            return _mock_rewrite_sentence(sentence)
        if gen != sentence and len(gen.split()) >= 3:
            return gen
    except Exception:
        pass
    return _mock_rewrite_sentence(sentence)


def _llm_rewrite_sentence(sentence: str) -> str:
    """对外入口：LLM 改写；无模型/异常回退 Mock；无需改写返回空串。"""
    if not sentence:
        return ""
    key = sentence.strip()
    if not key:
        return ""
    result = _llm_rewrite_cached(key)
    if not result or result == key:
        return ""
    return result


def _high_pressure_warning(score: float) -> str:
    """strict 模式下，综合 AI 率 > 40% 时的顶部高压预警。"""
    if score <= 0.40:
        return ""
    return (
        "⚠️ 极度危险！当前 AI 特征已进入高危区。请立刻检查："
        "① 段落开头是否镜像（每段都用同一种句式起头）；"
        "② 第二篇文献是否少了第三个点（rubric 要求 3 点却只写了 2 点）；"
        "③ 是否有高级词汇（如 delve / moreover / utilize 等假学术腔）。"
        "改完再跑一遍，直到它吼不出来为止。"
    )


# ============================================================
# 文本提取
# ============================================================
def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from a .txt / .md / .docx file.

    Raises:
        ExtractionError: 文件不存在、不可读或 docx 解析失败。
    """
    if not os.path.isfile(file_path):
        raise ExtractionError(f"文件不存在：{file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".txt", ".md"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError as e:
            raise ExtractionError(f"无法读取文件 {file_path}: {e}") from e

    if ext == ".docx":
        try:
            from docx import Document
        except ImportError as e:
            raise ExtractionError(
                "解析 .docx 需要安装 python-docx（pip install python-docx）") from e
        try:
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise ExtractionError(f"docx 解析失败 {file_path}: {e}") from e

    # 默认按文本读
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError as e:
        raise ExtractionError(f"无法读取文件 {file_path}: {e}") from e


# ============================================================
# AI 检测引擎
# ============================================================
def _detect_with_lmscan(text: str) -> Optional[float]:
    """调用 lmscan CLI，返回 AI 概率。"""
    try:
        result = subprocess.run(
            ["lmscan", "-", "--format", "json"],
            input=text, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("ai_probability", 0.5))
        _record_gap("lmscan", f"lmscan CLI 返回非零退出码（{result.returncode}）")
    except FileNotFoundError:
        _record_gap("lmscan", "未安装 lmscan CLI（可选外部检测器）")
    except Exception:
        _record_gap("lmscan", "lmscan CLI 调用失败")
    return None


def _detect_with_baseline(text: str) -> float:
    """纯 Python 统计基线：句长方差 + 词型丰富度。

    v3.20【P2-3】返回值从 (score, flagged) 精简为 score：flagged
    列表在唯一调用处被 `_, _flagged = ...` 整体丢弃，从未消费，
    属死数据。"""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.5

    sent_lengths = [int(_word_count(s)) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)

    words = text.lower().split()
    if not words:
        return 0.5
    ttr = len(set(words)) / len(words)

    # burstiness 越大越像人
    burstiness = variance / (avg_len ** 2 + 1)
    if burstiness < 0.25:
        ai_score = 0.75
    elif burstiness < 0.50:
        ai_score = 0.55
    elif burstiness < 0.75:
        ai_score = 0.35
    else:
        ai_score = 0.15

    if ttr > 0.60:
        ai_score += 0.08
    elif ttr < 0.40:
        ai_score -= 0.08

    return min(max(ai_score, 0), 1)


def _detect_perplexity_burstiness(text: str) -> Optional[float]:
    """困惑度 + 突发性近似（纯 Python）。"""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return None
    lengths = [int(_word_count(s)) for s in sentences]
    avg = sum(lengths) / len(lengths)
    std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
    burstiness = std / (avg + 1)
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
_AI_FORMULAIC_RES = tuple(
    re.compile(p, re.I | re.S) for p in _AI_FORMULAIC_MARKERS)


def _detect_formulaic_markers(text: str) -> float:
    """检测 AI 常用短语（逆向自 lynote）。"""
    text_lower = text.lower()
    marker_hits = sum(1 for pat in _AI_FORMULAIC_RES if pat.search(text_lower))
    return max(0.0, min(1.0, marker_hits / 10.0))


def _detect_transition_density(text: str) -> float:
    """检测过渡词密度（逆向自 lynote）。"""
    words = text.lower().split()
    word_count = len(words)
    if word_count < 50:
        return 0.3
    tl = text.lower()
    transition_hits = sum(tl.count(t) for t in _TRANSITION_WORDS)
    density = transition_hits / word_count * 100
    return max(0.0, min(1.0, density / 6.0))


# —— 翻译腔 / DeepL 机翻检测器 ——
_TRANSLATIONESE_PREP_CLAUSES = [
    "of which", "of what", "in which", "for which", "to which", "by which",
    "with which", "from which", "on which", "at which", "among which",
    "through which", "upon which", "in what", "at what",
]
_TRANSLATIONESE_PASSIVE = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:is|are|was|were)\s+being\s+\w+ed\b",
        r"\b(?:has|have|had)\s+been\s+\w+ed\b",
        r"\b(?:can|could|should|would|may|might|must)\s+be\s+\w+ed\b",
        r"\b(?:is|are|was|were)\s+\w+ed\s+by\b",
        r"\bit\s+(?:is|was)\s+\w+ed\s+that\b",
        r"\b(?:is|are)\s+to\s+be\s+\w+ed\b",
        r"\b(?:has|have)\s+been\s+being\s+\w+ed\b",
    ))


def _detect_translationese(text: str) -> float:
    """翻译腔 / 机翻特征检测：介词前置从句 + 冗长被动语态。

    专治「中文丢进 DeepL 直翻成英文」的洗稿手法。
    短文本（<50 词）返回 0.3（中性）。
    """
    words = text.lower().split()
    wc = len(words)
    if wc < 50:
        return 0.3
    low = text.lower()
    prep_hits = sum(low.count(p) for p in _TRANSLATIONESE_PREP_CLAUSES)
    prep_density = prep_hits / wc * 100.0
    passive_hits = sum(len(p.findall(text)) for p in _TRANSLATIONESE_PASSIVE)
    passive_density = passive_hits / wc * 100.0
    score = prep_density / 1.5 * 0.5 + passive_density / 3.0 * 0.5
    return max(0.0, min(1.0, score))


def _detect_compressibility(text: str) -> float:
    """检测压缩率：可压缩性越高 = 越重复 = 越像 AI。"""
    compressed = len(zlib.compress(text.encode("utf-8")))
    original = len(text.encode("utf-8"))
    ratio = compressed / max(1, original)
    return max(0.0, min(1.0, (0.50 - ratio) / 0.25))


_RE_LISTISH = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_RE_HEADINGISH = re.compile(
    r"(?m)^\s{0,3}#{1,3}\s+|^\s{0,3}[A-Z][A-Za-z ]{3,}:\s*$")


# ============================================================
# sloptotal 逆向集成：Formulaic 公式化表达检测
# ============================================================
_AI_OPENERS_C = tuple(re.compile(p) for p in (
    r"in today'?s (?:rapidly )?(?:evolving|changing|digital|modern)",
    r"in (?:the|a) (?:world|era|age|landscape) (?:where|of|that)",
    r"when it comes to",
    r"in the realm of",
    r"let'?s (?:dive|explore|take a (?:closer )?look|unpack|break down)",
    r"imagine (?:this|a world)",
    r"as (?:technology|the world|we|society) continues to",
    r"the (?:rise|emergence|advent|proliferation) of",
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
))
_AI_CLOSERS_C = tuple(re.compile(p) for p in (
    r"in conclusion,?",
    r"(?:in summary|to summarize|to sum up),?",
    r"(?:ultimately|at the end of the day),?",
    r"as we (?:navigate|move forward|look ahead|continue)",
    r"by (?:embracing|leveraging|harnessing|adopting)",
    r"it'?s clear that",
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
))
_FILLER_RES = tuple(re.compile(p) for p in _FILLER_PATTERNS)


def _detect_formulaic(text: str) -> float:
    """检测 AI 公式化表达（逆向自 sloptotal FormulaicEngine）。"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)

    opener_score = 0.0
    opener_text = text_lower[:300]
    for pattern in _AI_OPENERS_C:
        if pattern.search(opener_text):
            opener_score = 1.0
            break

    closer_score = 0.0
    closer_text = text_lower[-400:]
    for pattern in _AI_CLOSERS_C:
        if pattern.search(closer_text):
            closer_score = 1.0
            break

    filler_count = sum(len(p.findall(text_lower)) for p in _FILLER_RES)
    filler_density = (filler_count / word_count) * 1000
    filler_score = min(filler_density / 5.0, 1.0)

    final_score = (opener_score * 0.20 + closer_score * 0.15
                   + filler_score * 0.65)
    return round(final_score, 3)


# ============================================================
# sloptotal 逆向集成：Linguistic 语言标志检测
# ============================================================
def _detect_linguistic_markers(text: str) -> float:
    """检测 AI 语言标志（逆向自 sloptotal LinguisticEngine）。"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)
    total_weight = 0
    for marker, weight in _AI_MARKERS_WEIGHTED:
        count = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
        total_weight += count * weight
    density = (total_weight / word_count) * 1000
    return round(min(density / 15.0, 1.0), 3)


# ============================================================
# sloptotal 逆向集成：Vocabulary 词汇丰富度检测
# ============================================================
def _detect_vocabulary_richness(text: str) -> float:
    """检测词汇丰富度（逆向自 sloptotal VocabularyEngine）。

    【v3.23 P1-F】中文不再返回 0.0。本引擎全部指标（root-TTR、hapax
    比例、Yule's K、平均词长、长词/短词占比）都建立在英文词形上，
    中文经 _RE_WORD 只能切出零星英文术语，words 恒 < 20，旧实现
    return 0.0 —— 而 0.0 意味着"极不像 AI"，等于用一个没测过的引擎
    把中文综合分往下拽。改为登记 gap + 返回中性 0.5，报告里如实
    显示"该引擎未参与"，而不是伪装成一个人写证据。

    Args:
        text: 待检测文本。

    Returns:
        0~1 的 AI 嫌疑分；中文文本恒为中性 0.5（并记 gap）。
    """
    if _detect_language(text) == "zh":
        _record_gap("vocabulary-richness",
                    "仅对英文文本启用（TTR/词长指标按英文词形标定）")
        return 0.5
    words = _RE_WORD.findall(text.lower())
    if len(words) < 20:
        return 0.0
    total = len(words)
    freq = Counter(words)
    unique = len(freq)

    root_ttr = unique / math.sqrt(total)
    ttr_score = min(max(max(0, 1.0 - ((root_ttr - 3.5) / 4.0)), 0.0), 1.0)

    hapax = sum(1 for c in freq.values() if c == 1)
    hapax_ratio = hapax / unique if unique > 0 else 0
    hapax_score = min(max(max(0, 1.0 - ((hapax_ratio - 0.3) / 0.4)), 0.0), 1.0)

    freq_of_freq = Counter(freq.values())
    m1 = total
    m2 = sum(i * i * v for i, v in freq_of_freq.items())
    yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0
    k_score = min(max((yules_k - 80) / 120, 0.0), 1.0)

    avg_word_length = sum(map(len, words)) / total
    length_score = min(max((avg_word_length - 4.5) / 2.0, 0.0), 1.0)

    long_words = sum(1 for w in words if len(w) > 6)
    long_score = min(max((long_words / total - 0.10) / 0.20, 0.0), 1.0)

    short_words = sum(1 for w in words if len(w) < 3)
    short_score = min(max((0.30 - short_words / total) / 0.15, 0.0), 1.0)

    final_score = (ttr_score * 0.20 + hapax_score * 0.20 + k_score * 0.15
                   + length_score * 0.15 + long_score * 0.15
                   + short_score * 0.15)
    return round(final_score, 3)


# ============================================================
# sloptotal 逆向集成：Structural 结构特征检测
# ============================================================
_RE_EM_DASH = re.compile(r"\w—\w")
_RE_LIST_LINE = re.compile(r"\s*[-•*]\s")
_RE_NUM_LIST_LINE = re.compile(r"\s*\d+[.)]\s")
_RE_HEADING_MD = re.compile(r"^#{1,6}\s", re.MULTILINE)
_RE_COLON_DEF = re.compile(r"[A-Z][a-z]+(?:\s[A-Z]?[a-z]+){0,3}:\s[A-Z]")
_RE_PAREN = re.compile(r"[()（）]")
_RE_QUOTES = re.compile(r"[''\"\"'']")
_RE_BOLD = re.compile(r"\*\*[^*]+\*\*")
_RE_TABLE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_RE_SEMICOLON = re.compile(r";")
_RE_DASH = re.compile(r"--|—")


# ============================================================
# sloptotal 逆向集成：Sentiment & Hedging 情感与对冲检测
# ============================================================
_VAGUE_POSITIVITY = [
    "incredibly important", "extremely valuable", "truly remarkable",
    "absolutely essential", "deeply meaningful", "truly transformative",
    "game-changing", "world-class", "best-in-class", "make a difference",
    "positive impact", "meaningful change", "better future",
    "brighter future", "greater good", "incredible potential",
    "exciting opportunity", "powerful tool", "valuable resource",
    "essential guide",
    "a new era", "a new paradigm", "a game-changer", "a breakthrough",
    "a major milestone", "a significant step forward", "a key factor",
    "a crucial element", "a vital component", "a foundational pillar",
    "a cornerstone of", "a testament to", "a symbol of",
    "a representation of", "an embodiment of", "the future of",
    "the next generation", "the cutting edge", "the forefront of",
    "the leading edge",
    "a powerful force", "a driving force", "a catalyst for change",
    "a catalyst for growth", "a catalyst for innovation",
    "a beacon of hope", "a beacon of light", "a ray of hope",
    "a breath of fresh air", "a world of possibilities",
    "a world of opportunities", "a world of potential",
    "a world of difference", "a positive impact", "a lasting impact",
    "a profound impact", "a significant impact", "a meaningful impact",
    "a transformative impact",
]
_FILLER_QUALIFIER_RES = tuple(
    re.compile(r"\b" + re.escape(w) + r"\b") for w in (
        "very", "really", "extremely", "incredibly", "absolutely",
        "definitely", "certainly", "undoubtedly", "undeniably", "quite",
        "rather", "fairly", "somewhat", "relatively",
        "super", "ultra", "mega", "highly", "deeply", "profoundly",
        "tremendously", "immensely", "enormously", "massively",
        "significantly", "substantially", "considerably", "noticeably",
        "markedly",
    ))

_RE_EXPL_VERBS = re.compile(
    r"\bThis (?:means|ensures|allows|enables|helps|provides|creates|offers"
    r"|makes|includes|involves|requires)\b")
_RE_EXPL_NOUNS = re.compile(
    r"\b[Tt]his (?:approach|method|process|tool|feature|system|framework"
    r"|strategy|solution)\b")
_RE_EXPL_THESE = re.compile(
    r"\b[Tt]hese (?:tools|features|systems|methods|approaches|strategies"
    r"|solutions|techniques|elements|factors|components|resources)\b")


def _detect_sentiment_hedging(text: str) -> float:
    """检测情感与对冲模式（逆向自 sloptotal SentimentEngine）。"""
    text_lower = text.lower()
    word_count = max(int(_word_count(text)), 1)

    hedge_count = sum(text_lower.count(p) for p in _HEDGING_PHRASES)
    hedge_score = min((hedge_count / word_count) * 1000 / 4.0, 1.0)

    balance_count = sum(
        len(re.findall(p, text_lower, re.DOTALL)) for p in _BALANCE_PATTERNS)
    balance_score = min(balance_count / 3.0, 1.0)

    vague_count = sum(text_lower.count(p) for p in _VAGUE_POSITIVITY)
    vague_score = min((vague_count / word_count) * 1000 / 3.0, 1.0)

    qualifier_count = sum(len(p.findall(text_lower))
                          for p in _FILLER_QUALIFIER_RES)
    qualifier_density = (qualifier_count / word_count) * 1000
    qualifier_score = max(0, min((qualifier_density - 8) / 15.0, 1.0))

    explanatory_count = (len(_RE_EXPL_VERBS.findall(text))
                         + len(_RE_EXPL_NOUNS.findall(text))
                         + len(_RE_EXPL_THESE.findall(text)))
    explanatory_score = min(explanatory_count / 4.0, 1.0)

    final_score = (hedge_score * 0.30 + balance_score * 0.15
                   + vague_score * 0.20 + qualifier_score * 0.15
                   + explanatory_score * 0.20)
    return round(final_score, 3)


def _detect_human_imperfection(text: str) -> float:
    """人类写作痕迹检测器 v2（语篇级信号，对格式规范的真人同样有效）。

    旧版只统计打字错误，Word 排版文本恒为 0，无区分力。
    v2 用句长 CV / 过渡词密度 / 实词重复 / 模糊限定语 / 句首多样性。
    """
    sentences = [s for s in _split_sentences(text) if int(_word_count(s)) >= 3]
    if len(sentences) < 4:
        return 0.5
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)
    lens = [int(_word_count(s)) for s in sentences]

    mean_len = sum(lens) / len(lens)
    var = sum((l - mean_len) ** 2 for l in lens) / len(lens)
    cv = (var ** 0.5) / mean_len if mean_len else 0.0
    cv_ai = max(0.0, min(1.0, (0.70 - cv) / 0.45))

    # 【v3.23 P2-G】中文只保留句长 CV。本引擎另外四个信号（过渡词密度 /
    # 实词重复 / 模糊限定语 / 句首多样性）全部建立在英文词表上：中文
    # 文本里它们恒为"零命中"，而零命中在代码里被翻译成 1.0（高 AI
    # 嫌疑）——四个恒 1.0 的假信号把中文分数钉在 0.48 以上，跟文章
    # 写得怎么样无关。
    #
    # 处理方式不是"让 cv 独自占满权重"，而是把四个失效信号替换成
    # 中性 0.5 再按原权重合成：0.30*cv_ai + 0.70*0.5。这样句长信号
    # 的方向仍在，幅度回到它原本的 0.30 份额——若直接返回 cv_ai，
    # 等于把一个占 30% 的信号放大到 100%，实测会让 AI 中文样例的
    # 该引擎分从 0.696 掉到 0.25，方向是错的。
    if _detect_language(text) == "zh":
        return round(max(0.05, min(0.95, cv_ai * 0.30 + 0.5 * 0.70)), 3)

    _trans = ("furthermore", "moreover", "additionally", "consequently",
              "therefore", "however", "in addition", "overall",
              "in conclusion", "firstly", "secondly", "thirdly", "finally",
              "thus")
    tl = text.lower()
    trans = sum(tl.count(w) for w in _trans)
    trans_ai = max(0.0, min(1.0, (trans / n_words * 1000 - 2.0) / 12.0))

    content = [w for w in words if len(w) > 3]
    cnt = Counter(content)
    rep = sum(c - 1 for c in cnt.values() if c > 2) / max(len(content), 1)
    rep_ai = max(0.0, min(1.0, (0.10 - rep) / 0.10))

    _vague = (" very ", " really ", " quite ", " maybe ", " a lot ",
              " lots of ", "i think ", "i believe ", "in my opinion",
              " some ", " kind of ", " actually ", " of course ",
              " and so on ")
    padded = " " + tl + " "
    vague = sum(padded.count(w) for w in _vague)
    vague_ai = max(0.0, min(1.0, 1.0 - vague / n_words * 1000 / 8.0))

    starters = {"this", "it", "and", "but", "so", "that", "these",
                "also", "then"}
    first_words = [s.split()[0].lower().strip(",.")
                   for s in sentences if s.split()]
    lead_ratio = (sum(1 for w in first_words if w in starters)
                  / max(len(first_words), 1))
    lead_ai = max(0.0, min(1.0, (0.30 - lead_ratio) / 0.30))

    ai_prob = (cv_ai * 0.30 + trans_ai * 0.22 + rep_ai * 0.18
               + vague_ai * 0.20 + lead_ai * 0.10)
    return round(max(0.05, min(0.95, ai_prob)), 3)


# ============================================================
# 情感曲线检测器
# ============================================================
_NEGATIONS = frozenset([
    "not", "no", "never", "don't", "doesn't", "isn't", "aren't", "won't",
    "can't", "cannot", "couldn't", "wouldn't", "shouldn't", "haven't",
    "hasn't", "hadn't", "didn't", "wasn't", "weren't",
    "不", "没", "无", "非", "未", "不是", "没有", "不会", "不能", "不要",
    "并非", "毫不", "绝不", "从不",
])  # O(1) 查找替代列表扫描


def _detect_paragraph_rhythm(text: str) -> float:
    """段落节奏检测（不依赖词表，跨语言有效）。

    只统计 ≥40 词的正文段落。段落数 <5 时 CV 不稳，且注释书目等体裁
    本身就要求段长接近（体裁特征而非 AI 特征），退回中性值。
    """
    paras = [p.strip() for p in _RE_NL.split(text) if len(p.split()) >= 40]
    if len(paras) < 5:
        return 0.5
    lens = [_word_count(p) for p in paras]
    mean_len = sum(lens) / len(lens)
    if not mean_len:
        return 0.5
    cv = statistics.pstdev(lens) / mean_len
    return round(max(0.0, min(1.0, (0.55 - cv) / 0.40)), 3)


# 预编译情感词 alternation 正则：替代「逐 token × 逐词表」双层
# 子串扫描（v3.19【P2-3】）。re.search(alt, word) 与原版
# `any(w in word for w in 词表)` 布尔语义完全一致，命中即计 1。
def _compile_alternation(words) -> Optional[re.Pattern]:
    alts = [re.escape(w) for w in (words or ()) if w]
    return re.compile("|".join(alts)) if alts else None


_RE_EMOTION_POS = _compile_alternation(_POSITIVE_WORDS)
_RE_EMOTION_NEG = _compile_alternation(_NEGATIVE_WORDS)


def _detect_emotion_curve(text: str) -> float:
    """情感曲线检测器：AI 情感平，人类有起伏。

    v3.18【P2-3】：判断前词是否否定时先剥标点——
    "don't," 裸匹配不上 _NEGATIONS 里的 "don't"。
    """
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5

    emotion_scores = []
    emotion_counts = []
    for sent in sentences:
        words = sent.lower().split()
        pos_count = 0
        neg_count = 0
        for i, word in enumerate(words):
            is_negated = (i > 0 and words[i - 1].strip(".,!?;:'\"()（）")
                          in _NEGATIONS)
            # v3.19【P2-3】单条预编译正则替代逐词表子串扫描；
            # 与旧版一致：一个词最多给每个词表各贡献 1 次计数。
            if _RE_EMOTION_POS is not None and _RE_EMOTION_POS.search(word):
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            if _RE_EMOTION_NEG is not None and _RE_EMOTION_NEG.search(word):
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1
        emotion_counts.append(pos_count + neg_count)
        if pos_count + neg_count > 0:
            emotion_scores.append(
                (pos_count - neg_count) / (pos_count + neg_count))
        else:
            emotion_scores.append(0.0)

    if len(emotion_scores) < 2:
        return 0.5
    # v3.20【P0-3】词表零命中（典型：中文文本——正负情感词表是英文的），
    # 每句计数全 0 → 标准差恒 0 → 旧版恒判 1.0，中文下是系统性假 AI
    # 信号。覆盖为零说明引擎对该文本不适用，返回中性 0.5。
    if not any(c > 0 for c in emotion_counts):
        return 0.5
    std_emotion = statistics.stdev(emotion_scores)
    # 标准差越小 = 情感越平 = 越像 AI
    return round(max(0.0, min(1.0, 1.0 - (std_emotion / 0.4))), 3)


def _detect_specific_details(text: str) -> float:
    """具体细节密度检测器 v2。

    v1 用 \\b[A-Z][a-z]+ [A-Z][a-z]+\\b 抓人名，把句首语法性大写全算成
    专名，密度虚高几十倍。v2 改用三条干净线索：数字 / 非句首专名 / 引语。
    v3.18【P2-2】：删掉从未使用的 lang 形参（死参数不如没有）。
    """
    sentences = [s for s in _split_sentences(text) if s.strip()]
    n_words = max(int(_word_count(text)), 1)

    numbers = len(_RE_NUM.findall(text))

    proper = 0
    for s in sentences:
        tokens = s.split()
        for i, w in enumerate(tokens):
            if i == 0:
                continue
            core = w.strip(".,;:()\"'")
            if _RE_PROPER.match(core):
                proper += 1

    quoted = len(_RE_QUOTED.findall(text))

    density = (numbers * 3 + proper * 2 + quoted * 3) / n_words * 1000
    return round(max(0.0, min(1.0, 1.0 - density / 45.0)), 3)


_SELF_REF_STRONG = [
    "i do not understand", "i don't understand", "i did not understand",
    "i found that", "i found it", "i noticed", "i realised", "i realized",
    "i was surprised", "i was confused", "i am not sure", "i'm not sure",
    "in my own experience", "from my own experience", "i remember",
    "i have seen", "i could not", "i couldn't", "i tried",
    "i learned that", "i learnt that", "to be honest", "i am not clear",
    "i'm not clear",
]
_SELF_REF_WEAK = [
    "i think", "i believe", "i feel", "in my opinion", "in my view",
    "from my perspective", "personally", "i would say", "as far as i",
]


# ============================================================
# 缺失引擎登记（并发安全）
# ============================================================
_ENGINE_GAPS: "ContextVar[Dict[str, str]]" = ContextVar("aurora_engine_gaps")
_PERSISTENT_GAPS: Dict[str, str] = {}
_PERSISTENT_GAPS_LOCK = threading.Lock()


def _current_gaps() -> Dict[str, str]:
    """取当前上下文的缺失登记 dict；未初始化则懒建。"""
    gaps = _ENGINE_GAPS.get(None)
    if gaps is None:
        gaps = {}
        _ENGINE_GAPS.set(gaps)
    return gaps


def _reset_gaps() -> None:
    """重置当前上下文的缺失登记（主要供测试）。"""
    _ENGINE_GAPS.set({})


def _record_gap(name: str, reason: str, persistent: bool = False) -> None:
    """登记一个本次未能参与检测的引擎及原因。"""
    _current_gaps()[name] = reason
    if persistent:
        with _PERSISTENT_GAPS_LOCK:
            _PERSISTENT_GAPS[name] = reason


# ============================================================
# 模型生命周期管理：30 分钟空闲自动回收（v3.17 新增）
# ------------------------------------------------------------
# 设计要点：
#   - 每个模型资源注册 release 回调 + 最后使用时间戳；
#   - 推理前用 _model_in_use() 守卫标记「正在使用」，
#     回收线程绝不动在用的模型（杜绝推理中途模型被 del 的崩溃）；
#   - 后台 daemon 线程定期扫描（默认每 5 分钟），超过 TTL
#     （默认 30 分钟，AURORA_MODEL_TTL 可调）即释放；
#   - 释放后复位懒加载标志，下次访问自动重新加载；
#   - CPU 内存靠 del + gc.collect()；显存再补 empty_cache()。
# ============================================================
_MODEL_IDLE_TTL = float(os.environ.get("AURORA_MODEL_TTL", "1800"))       # 30 min
_REAPER_INTERVAL = float(os.environ.get("AURORA_REAPER_INTERVAL", "300"))  # 5 min

_MODEL_REGISTRY: Dict[str, dict] = {}  # name -> {"last_used","in_use","release"}
_MODEL_REGISTRY_LOCK = threading.Lock()
_REAPER_STARTED = False
_REAPER_STOP = threading.Event()


def _touch_model(name: str) -> None:
    """刷新模型最后使用时间戳（加载/推理前调用）。"""
    with _MODEL_REGISTRY_LOCK:
        entry = _MODEL_REGISTRY.get(name)
        if entry is None:
            entry = {"last_used": time.monotonic(), "in_use": 0, "release": None}
            _MODEL_REGISTRY[name] = entry
        entry["last_used"] = time.monotonic()


def _register_model(name: str, release_fn) -> None:
    """注册模型的释放回调（加载成功后调用一次）。"""
    with _MODEL_REGISTRY_LOCK:
        entry = _MODEL_REGISTRY.setdefault(
            name, {"last_used": time.monotonic(), "in_use": 0, "release": None})
        entry["release"] = release_fn
        entry["last_used"] = time.monotonic()


@contextmanager
def _model_in_use(name: str):
    """推理期守卫：in_use > 0 期间回收线程跳过该模型。

    每个调用模型的检测函数都应以 `with _model_in_use(名字):`
    包住「取模型 → 推理结束」的全程。
    """
    with _MODEL_REGISTRY_LOCK:
        entry = _MODEL_REGISTRY.setdefault(
            name, {"last_used": time.monotonic(), "in_use": 0, "release": None})
        entry["in_use"] = entry.get("in_use", 0) + 1
        entry["last_used"] = time.monotonic()
    try:
        yield
    finally:
        with _MODEL_REGISTRY_LOCK:
            entry = _MODEL_REGISTRY.get(name)
            if entry is not None:
                entry["in_use"] = max(0, entry.get("in_use", 0) - 1)
                entry["last_used"] = time.monotonic()


def _empty_cuda_cache() -> None:
    """先 gc 让张量真正析构，再清 GPU 缓存（无 GPU 时为无害 no-op）。"""
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:  # Apple Silicon（可选）
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def _release_model(name: str) -> bool:
    """释放指定模型。正在使用（in_use>0）或未注册时跳过。"""
    with _MODEL_REGISTRY_LOCK:
        entry = _MODEL_REGISTRY.get(name)
        if entry is None or entry.get("release") is None:
            return False
        if entry.get("in_use", 0) > 0:
            return False
        release_fn = entry["release"]
        entry["release"] = None
    release_fn()  # 锁外执行，避免持锁做耗时清理
    return True


def _reap_idle_models(force: bool = False) -> List[str]:
    """回收所有超过空闲 TTL（force=True 时全部未在用的）模型。"""
    now = time.monotonic()
    stale: List[str] = []
    with _MODEL_REGISTRY_LOCK:
        for name, entry in _MODEL_REGISTRY.items():
            if entry.get("release") is None:
                continue
            if entry.get("in_use", 0) > 0:
                continue
            if force or (now - entry["last_used"]) > _MODEL_IDLE_TTL:
                stale.append(name)
    # _release_model 内部会二次检查 in_use（防止收集与释放间隙的新使用者）
    return [n for n in stale if _release_model(n)]


def release_idle_models(force: bool = False) -> List[str]:
    """公开接口：手动回收闲置模型（如测试、显存紧张时主动调用）。

    force=True 额外硬重置三个懒加载标志位（_ml_load_done /
    _mlm_load_done / _PPL_TRIED）——「失败即粘性」(P1-1) 场景下
    加载失败不注册 release 回调，此前 force=True 触发不了任何重载；
    这是断网→联网后重新加载模型的唯一入口。若此刻仍断网，下次
    访问会再走一遍探测并重新粘住，行为自洽。

    注意：重置时若模型已加载且空闲，会先被 _reap_idle_models(force)
    正常释放（各 release 回调自身也会复位标志位）；若正被推理占用
    则跳过释放，待下次访问时整体重载覆盖旧引用，在途推理持有的
    旧对象随引用计数自然回收，无崩溃风险。

    Returns:
        被成功释放的模型名列表，如 ["ml-detector", "ppl", "mlm-perplexity"]。
    """
    global _ml_load_done, _mlm_load_done, _PPL_TRIED
    if force:
        # 强制重试路径：联网后可用 force=True 重新触发模型加载
        _ml_load_done = False
        _mlm_load_done = False
        _PPL_TRIED = False
    # v3.20【P0-2】缓存感知模型生命周期：模型被释放/待重载后，
    # 引擎层缓存的旧结论（可能产自"断网无模型"状态）必须作废，
    # 否则同样文本再次检测会命中旧缓存，返回旧模型的结论。
    _engine_stage_cached.cache_clear()
    return _reap_idle_models(force=force)


def _start_model_reaper(interval: float = _REAPER_INTERVAL) -> None:
    """启动后台回收线程（daemon，进程退出自动结束）。幂等。"""
    global _REAPER_STARTED
    with _MODEL_REGISTRY_LOCK:
        if _REAPER_STARTED:
            return
        _REAPER_STARTED = True

    def _loop():
        while not _REAPER_STOP.wait(interval):
            try:
                freed = _reap_idle_models()
                if freed and _VERBOSE:
                    print(f"[aurora] 已回收闲置模型（空闲超 "
                          f"{_MODEL_IDLE_TTL:.0f}s）：{', '.join(freed)}",
                          file=sys.stderr)
            except Exception:
                pass  # 回收失败不影响主流程，下轮再试

    threading.Thread(target=_loop, daemon=True,
                     name="aurora-model-reaper").start()


# ============================================================
# MLM 伪困惑度：全局状态与语料校准
# ============================================================
_mlm_model = None
_mlm_tokenizer = None
_mlm_load_done = False
_mlm_load_lock = threading.Lock()
_MLM_MODEL_NAME = os.environ.get("AURORA_PPL_MODEL",
                                 "answerdotai/ModernBERT-base")

_corpus_calibration = None
_corpus_calibration_loaded = False


def _load_corpus_calibration() -> Optional[dict]:
    """加载 corpus/calibration.json（若有），无则返回 None。"""
    global _corpus_calibration, _corpus_calibration_loaded
    if _corpus_calibration_loaded:
        return _corpus_calibration
    _corpus_calibration_loaded = True
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "corpus", "calibration.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _corpus_calibration = json.load(f)
    except Exception:
        _corpus_calibration = None
    return _corpus_calibration


# 默认链只保留 ModernBERT（旧 RoBERTa 对现代 LLM 几乎总给 0%）
_ML_MODEL_CHAIN = (os.environ.get("AURORA_AI_MODEL")
                   or "GeorgeDrayson/modernbert-ai-detection-raid-mage"
                   ).split("|")

_ml_models: Dict[str, dict] = {}
_ml_load_done = False
_ml_load_lock = threading.Lock()


def _hf_reachable(host: str = "", timeout: float = 2.5) -> bool:
    """TCP 连通性快速探测（跟随 HF_ENDPOINT 镜像设置）。"""
    if not host:
        endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
        host = endpoint.split("//", 1)[-1].split("/", 1)[0] or "huggingface.co"
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def _hf_cache_root() -> str:
    """HF 缓存根目录。v3.20【P1-10】尊重 HF_HUB_CACHE / HF_HOME——
    此前硬编码 ~/.cache/huggingface/hub，自定义缓存目录的用户明明
    有缓存却被判「无缓存 → 触发网络探测 → 失败即粘性 → 永久不加载」。
    """
    env = os.environ
    if env.get("HF_HUB_CACHE", "").strip():
        return env["HF_HUB_CACHE"]
    if env.get("HF_HOME", "").strip():
        return os.path.join(env["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache",
                        "huggingface", "hub")


def _hf_model_cached(name: str) -> bool:
    """检查模型是否已有本地 HF 缓存（可严格离线加载）。"""
    safe = name.replace("/", "--")
    d = os.path.join(_hf_cache_root(), f"models--{safe}")
    return os.path.isdir(d)


def _ensure_hf_endpoint() -> None:
    """直连不可达时自动补设 hf-mirror.com 镜像。"""
    if not _hf_reachable():
        _mirror = "https://hf-mirror.com"
        if os.environ.get("HF_ENDPOINT", "").strip().rstrip("/") != _mirror:
            os.environ["HF_ENDPOINT"] = _mirror


# ============================================================
# ML 分类器：加载 / 释放
# ============================================================
def _release_ml_models() -> None:
    """回收 ml-detector 链中全部分类模型。

    用「重新绑定新 dict」而非原地 clear：若有调用方仍持有旧 dict
    引用（在途推理），其迭代不会被打断崩溃；旧模型对象随最后一个
    引用消失而释放。全局引用换新后，下次访问自动重新懒加载。
    """
    global _ml_models, _ml_load_done
    with _ml_load_lock:
        _ml_models = {}
        _ml_load_done = False  # 复位懒加载标志
    _empty_cuda_cache()
    if _VERBOSE:
        print("[aurora] 已回收闲置模型：ml-detector", file=sys.stderr)


def _load_ml_models():
    """懒加载链中全部 ML 检测器，返回 {name: {model, tokenizer, ai_idx}}。

    v3.17：加载全程持锁（防止并发重复加载），成功后注册到空闲回收。
    v3.18【P1-1】：入口与双重检查改为「只要尝试过（_ml_load_done）
    就返回当前结果」。此前写的是 `done and models`，加载失败时
    models 为空、条件恒 False，断网环境每次 run_check 都白等
    2.5s+ 网络探测。失败即粘性，热加载请重启进程。
    """
    global _ml_load_done
    if _ml_load_done:
        if _ml_models:
            _touch_model("ml-detector")
        return _ml_models

    with _ml_load_lock:
        if _ml_load_done:
            if _ml_models:
                _touch_model("ml-detector")
            return _ml_models

        # 离线优先：已有缓存直接加载，不联网
        _any_cached = any(_hf_model_cached(n) for n in _ML_MODEL_CHAIN)
        if not _any_cached:
            if not _hf_reachable():
                _ensure_hf_endpoint()
                if not _hf_reachable():
                    _ml_load_done = True
                    _record_gap("ml-detector",
                                "无法连接 HuggingFace（及镜像 hf-mirror.com），"
                                "且本地无缓存模型", persistent=True)
                    print("[aurora] 连不上 huggingface.co 及其镜像，跳过 ML 检测器"
                          "（其余统计引擎照常工作，不影响结论方向）",
                          file=sys.stderr)
                    return _ml_models
        else:
            _ensure_hf_endpoint()

        try:
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            import torch  # noqa: F401  # noqa: F401 — 故意提前导入：torch 缺失时快速失败
        except Exception as e:
            _ml_load_done = True
            _record_gap("ml-detector", "transformers/torch 未安装或不可用",
                        persistent=True)
            if _VERBOSE:
                print(f"[aurora] transformers/torch 不可用：{e}", file=sys.stderr)
            return _ml_models

        ai_keys = ("machine", "generated", "ai", "chatgpt", "fake", "bot",
                   "llm", "synthetic", "gpt")
        for model_name in _ML_MODEL_CHAIN:
            if model_name in _ml_models:
                continue
            try:
                if _hf_model_cached(model_name):
                    try:
                        tok = AutoTokenizer.from_pretrained(
                            model_name, local_files_only=True)
                        mdl = AutoModelForSequenceClassification.from_pretrained(
                            model_name, local_files_only=True)
                    except Exception:
                        tok = AutoTokenizer.from_pretrained(model_name)
                        mdl = AutoModelForSequenceClassification.from_pretrained(
                            model_name)
                else:
                    tok = AutoTokenizer.from_pretrained(model_name)
                    mdl = AutoModelForSequenceClassification.from_pretrained(
                        model_name)
                mdl.eval()
                id2label = mdl.config.id2label
                if len(id2label) == 1:
                    ai_idx = 0
                else:
                    idx = None
                    for k, label in id2label.items():
                        low = str(label).lower()
                        if any(key in low for key in ai_keys):
                            idx = int(k)
                            break
                    if idx is None:
                        # v3.20【P1-8】无 ai 关键词时先按语义反推：找
                        # human 同义词并取「另一个」索引，而不是盲目取
                        # max(key)——label 顺序颠倒（1=human）时旧兜底
                        # 会把检测方向整体反转。
                        human_keys = ("human", "student", "real",
                                      "original", "writer")
                        hidx = None
                        for k, label in id2label.items():
                            low = str(label).lower()
                            if any(key in low for key in human_keys):
                                hidx = int(k)
                                break
                        if hidx is not None:
                            idx = (set(int(k) for k in id2label)
                                   - {hidx}).pop() if len(id2label) == 2 \
                                else max(int(k) for k in id2label
                                         if int(k) != hidx)
                        else:
                            idx = max(int(k) for k in id2label.keys())
                    ai_idx = idx
                _ml_models[model_name] = {"model": mdl, "tokenizer": tok,
                                          "ai_idx": ai_idx}
                if _VERBOSE:
                    print(f"[aurora] 已加载 ML 检测器：{model_name} "
                          f"(ai_idx={ai_idx})", file=sys.stderr)
            except Exception as e:
                if _VERBOSE:
                    print(f"[aurora] 模型 {model_name} 加载失败：{e}",
                          file=sys.stderr)
        _ml_load_done = True

    _touch_model("ml-detector")
    if _ml_models:
        _register_model("ml-detector", _release_ml_models)
        _start_model_reaper()
    return _ml_models


def _load_ml_model():
    """向后兼容：返回链中第一个成功加载的模型。"""
    models = _load_ml_models()
    if not models:
        return None, None, None
    name = next(iter(models))
    info = models[name]
    return info["model"], info["tokenizer"], info["ai_idx"]


# ============================================================
# 隐藏字符检测器
# ============================================================
_HIDDEN_CHARACTERS = [
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",   # 零宽字符
    "\u00ad",                                            # 软连字符
    "\u200e", "\u200f",                                  # 方向标记
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",   # 嵌入/覆盖
    "\u2066", "\u2067", "\u2068", "\u2069",             # 隔离
    "\u034f", "\u0483", "\u0485",                       # 组合符
]


# ============================================================
# 句首词分布检测器
# ============================================================
_AI_SENTENCE_STARTERS = [
    "first", "second", "third", "finally", "lastly", "however", "therefore",
    "moreover", "furthermore", "additionally", "in addition", "in contrast",
    "in conclusion", "in summary", "on the other hand", "on one hand",
    "as a result", "consequently", "thus", "hence", "nevertheless",
    "nonetheless", "meanwhile",
    "it is", "it's", "there are", "there is", "this is",
    "this demonstrates", "this shows", "in today's", "in the world",
    "in the realm", "when it comes to", "let's", "let us",
    "one of", "another", "some", "many", "most", "studies show",
    "research indicates", "experts believe",
]
# 按首词建索引：把 O(starters) 的逐条 startswith 降到 O(1)
_STARTER_INDEX: Dict[str, List[str]] = {}
for _s in _AI_SENTENCE_STARTERS:
    _STARTER_INDEX.setdefault(_s.split()[0], []).append(_s)


def _starter_hit(first_word: str) -> bool:
    cands = _STARTER_INDEX.get(first_word.split()[0] if first_word else "", ())
    for st in cands:
        if first_word.startswith(st) or first_word == st:
            return True
    return False


def _detect_starter_patterns(text: str) -> float:
    """开头模式检测器（合并句首词分布 + 段落开头模式）。"""
    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        starters = []
        for sent in sentences:
            words = sent.strip().split()
            if words:
                starters.append(words[0].lower().strip(".,!?;:'\"()"))
        if starters:
            ai_starter_count = sum(1 for st in starters if _starter_hit(st))
            starter_score = min(
                max((ai_starter_count / len(starters) - 0.15) / 0.30, 0.0), 1.0)
        else:
            starter_score = 0.5
    else:
        starter_score = 0.5

    paragraphs = [p.strip() for p in text.split("\n\n")
                  if len(p.strip().split()) >= 5]
    if len(paragraphs) >= 3:
        opener_count = 0
        for para in paragraphs:
            words = para.split()
            if words:
                fw = words[0].lower().strip(".,!?;:'\"()")
                if _starter_hit(fw):
                    opener_count += 1
        paragraph_score = min(
            max((opener_count / len(paragraphs) - 0.25) / 0.40, 0.0), 1.0)
    else:
        paragraph_score = 0.5

    return round(starter_score * 0.6 + paragraph_score * 0.4, 3)


def _detect_sentence_length_distribution(text: str) -> float:
    """句子长度分布检测器（合并长句/短句交替 + 滑动窗口）。"""
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return 0.5
    lengths = [int(_word_count(s)) for s in sentences]
    diffs = [abs(lengths[i] - lengths[i + 1]) for i in range(len(lengths) - 1)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    alternation_score = min(max((7.0 - avg_diff) / 5.0, 0.0), 1.0)
    mean_len = statistics.mean(lengths)
    stdev = statistics.stdev(lengths)
    cv = stdev / mean_len if mean_len > 0 else 0
    uniformity_score = max(0, 1.0 - (cv / 0.6))
    return round(alternation_score * 0.5 + uniformity_score * 0.5, 3)


def _detect_pawn_perplexity(text: str) -> float:
    """PAWN 困惑度注意力加权检测器。"""
    words = text.lower().split()
    if len(words) < 20:
        return 0.5
    word_freq = Counter(words)
    total_words = len(words)
    log_probs = []
    for word in words:
        freq = word_freq.get(word, 0)
        prob = (freq + 0.5) / (total_words + 0.5 * len(word_freq))
        log_probs.append(math.log(prob))
    if not log_probs:
        return 0.5
    avg_log_prob = sum(log_probs) / len(log_probs)
    weights = [max(0, lp - avg_log_prob) for lp in log_probs]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5
    weighted_avg = (sum(lp * w for lp, w in zip(log_probs, weights))
                    / total_weight)
    return round(max(0.0, min(1.0, (-2.0 - weighted_avg) / 3.0)), 3)


def _detect_adversarial_perturbation(text: str) -> float:
    """对抗扰动程度检测器。"""
    words = text.lower().split()
    if len(words) < 20:
        return 0.5
    word_freq = Counter(words)
    unique_ratio = len(word_freq) / len(words)

    sentences = _split_sentences(text)
    if len(sentences) >= 5:
        sent_lengths = [int(_word_count(s)) for s in sentences]
        mean_len = sum(sent_lengths) / len(sent_lengths)
        variance = (sum((l - mean_len) ** 2 for l in sent_lengths)
                    / len(sent_lengths))
        cv = math.sqrt(variance) / mean_len if mean_len > 0 else 0
    else:
        cv = 0.5

    punct_types = set()
    for char in text:
        if char in ".,!?;:()[]{}\"'":
            punct_types.add(char)
    punct_diversity = len(punct_types) / 10.0

    perturbation = unique_ratio * 0.4 + cv * 0.3 + punct_diversity * 0.3
    return round(max(0.0, min(1.0, (0.5 - perturbation) / 0.3)), 3)


# ============================================================
# 语言检测 & 中文检测器
# ============================================================
@lru_cache(maxsize=64)
def _detect_language(text: str) -> str:
    """检测文本语言（中英文）。

    v3.18【P2-4】：原版拿「汉字数」对比「英文单词数」，一个英文
    单词平均 4~5 个字母，等于把英文打了 4~5 折，中英混排偏袒英文。
    现改为拉丁字母数 vs 汉字数，字符口径公平对比。
    注意：混排文本的 zh/en 判定可能与旧版不同（更准），下游
    语言自适应权重会随之切换，属预期行为变化。
    """
    han = len(_RE_ZH_CHAR.findall(text))
    latin = len(_RE_LATIN.findall(text))
    return "zh" if han > latin else "en"


_ZH_JARGON = [
    "赋能", "抓手", "闭环", "生态", "维度", "层面", "机制", "体系", "格局",
    "路径", "举措", "支撑", "驱动", "融合", "协同", "优化", "提升", "构建",
    "助力", "纵深", "引领", "夯实", "着力", "系统性", "整体性", "协同性",
    "显著提升", "大幅改善", "有效优化", "深远影响", "有力支撑",
    "重要保障", "卓越成效", "全面提升",
]
_ZH_BOILERPLATE = [
    "随着", "在当今", "近年来", "众所周知", "综上所述", "总而言之",
    "总的来说", "一言以蔽之", "由此可见", "需要指出的是",
]
_ZH_COLLOQUIAL = [
    "我觉得", "我感觉", "其实", "挺", "特别", "真的", "咱们", "咋", "吧",
    "呢", "啊", "嘛", "搞", "啥", "说白了", "反正", "估计", "好像", "有点",
]

_PPL_MODEL = None
_PPL_TOKENIZER = None
_PPL_TRIED = False
_ppl_load_lock = threading.Lock()


def _find_local_model():
    """按优先级找本地模型目录。"""
    cands = [
        os.environ.get("AURORA_PPL_MODEL", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "qwen2.5-0.5b"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "config.json")):
            return c
    return ""


def _release_ppl_model() -> None:
    """回收困惑度模型（Qwen2.5-0.5b）。置 None 后调用方持有的旧引用
    仍安全（Python 引用计数保证用完才释放）。"""
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_TRIED
    _PPL_MODEL = None
    _PPL_TOKENIZER = None
    _PPL_TRIED = False  # 复位：下次访问自动重载
    _empty_cuda_cache()
    if _VERBOSE:
        print("[aurora] 已回收闲置模型：ppl（Qwen）", file=sys.stderr)


def _load_ppl_model():
    """懒加载困惑度模型（v3.17：加锁防并发重载 + 空闲回收注册）。

    v3.18【P1-1】：失败即粘性（_PPL_TRIED 置位后不再重试）。
    """
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_TRIED
    if _PPL_TRIED:
        if _PPL_MODEL is not None:
            _touch_model("ppl")
        return _PPL_MODEL, _PPL_TOKENIZER

    with _ppl_load_lock:
        if _PPL_TRIED:
            if _PPL_MODEL is not None:
                _touch_model("ppl")
            return _PPL_MODEL, _PPL_TOKENIZER
        _PPL_TRIED = True
        path = _find_local_model()
        if not path:
            return None, None
        try:
            import warnings
            import logging
            warnings.filterwarnings("ignore")
            logging.disable(logging.WARNING)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            _PPL_TOKENIZER = AutoTokenizer.from_pretrained(path)
            _PPL_MODEL = AutoModelForCausalLM.from_pretrained(
                path, low_cpu_mem_usage=True)
            _PPL_MODEL.eval()
            if _VERBOSE:
                print(f"[aurora] 困惑度模型已加载：{path}", file=sys.stderr)
        except Exception as e:
            if _VERBOSE:
                print(f"[aurora] 模型加载失败：{e}", file=sys.stderr)
            _PPL_MODEL = None
            _PPL_TOKENIZER = None
            return None, None

    _touch_model("ppl")
    _register_model("ppl", _release_ppl_model)
    _start_model_reaper()
    return _PPL_MODEL, _PPL_TOKENIZER


_PPL_CHUNK = 500      # 单块 token 数（与 pivot=25 / scale=6.5 的标定口径一致）
_PPL_STRIDE = 450     # 块间步长，留 50 token 重叠
_PPL_MIN_CHUNK = 30   # 少于这么多 token 的块 CE loss 方差过大，不采信


def _ppl_chunk_spans(total: int) -> List[Tuple[int, int]]:
    """把 total 个 token 切成重叠块，返回按起点升序的 [(start, end), ...]。

    【v3.23 P1-A】上次 review 提过"尾块 30~499 token 会不会被丢掉"。
    不会，理由是重叠：stride(450) < chunk(500)，任何被 `end - start < 30`
    拦下的块，其起点 start 都满足 start > total - 30，而上一块
    [start - 450, start + 50) 的终点已经 ≥ total——尾段在那块里算过。
    唯一整段被跳过的情况是 total < 30，调用方在此之前已 return None。

    单独抽成函数是为了能用一个纯循环的测试把"区间并集覆盖 [0, total)"
    钉住（见 tests/test_contracts.py::test_ppl_chunks_cover_all_tokens），
    不必加载模型。
    """
    spans: List[Tuple[int, int]] = []
    for start in range(0, total, _PPL_STRIDE):
        end = min(start + _PPL_CHUNK, total)
        if end - start < _PPL_MIN_CHUNK:   # 尾块太短：已被上一块重叠覆盖
            break
        spans.append((start, end))
        if end >= total:
            break
    return spans


def _detect_lm_perplexity(text: str, lang: str = ""):
    """语言模型困惑度 → AI 概率（对"改写"最鲁棒的信号，单边证据）。

    只有"太顺"才构成证据：困惑度高不反证是人写（人写得好一样很顺），
    超过阈值直接不参与。v3.17：推理期受空闲回收守卫保护。

    【v3.23 P1-A】取样范围从「前 1200 字符单窗口」改为「4096 token
    分块扫描」：500 token 一块、stride 450、块 loss 取几何平均。
    长文后段的盲区消除；单块口径与既有 25 / 6.5 阈值保持一致。

    Args:
        text: 待检测文本（仅中文启用，英文直接返回 None 并记 gap）。
        lang: 已判定的语言，省略时内部再判一次。

    Returns:
        0~1 的 AI 嫌疑分；不适用（非中文 / 无模型 / ppl 越界）时 None。
    """
    if (lang or _detect_language(text)) != "zh":
        _record_gap("lm-perplexity",
                    "仅对中文文本启用（当前挂载中文模型，对英文困惑度不可靠）")
        return None
    with _model_in_use("ppl"):
        model, tok = _load_ppl_model()
        if model is None:
            _record_gap("lm-perplexity", "无本地语言模型（需安装 Qwen2.5 等中文模型）")
            return None
        try:
            import torch
            # 【v3.23 P1-A】分块覆盖全文。旧实现只喂前 1200 字符（单窗口
            # 500 token），长中文文档的后半部分是完全盲区，而这个引擎在
            # 中文权重表里占 0.22——一篇文章前半段像人、后半段像 AI，
            # 旧实现会判成"很干净"。
            #
            # 现在的做法：tokenize 到 4096（约 4000 汉字），按 _PPL_CHUNK
            # 切块、_PPL_STRIDE 步进（50 token 重叠，避免块边界处因缺少
            # 上下文而虚高 loss），每块算 CE loss，最后 exp(mean(loss)) 即
            # 几何平均伪困惑度。阈值 pivot=25 / scale=6.5 不动——它们是按
            # 单块 500 token 标定的，分块后每块的统计口径与之一致。
            ids = tok(text[:16000], return_tensors="pt",
                      truncation=True, max_length=4096)["input_ids"]
            total = ids.shape[1]
            if total < 30:
                return None
            losses = []
            with torch.inference_mode():
                for start, end in _ppl_chunk_spans(total):
                    chunk = ids[:, start:end]
                    losses.append(
                        float(model(input_ids=chunk, labels=chunk).loss))
            if not losses:
                return None
            ppl = float(math.exp(sum(losses) / len(losses)))
            if ppl <= 0 or ppl > 25.0:
                return None
            score = (math.log(25) - math.log(ppl)) / (math.log(25) - math.log(6.5))
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return None


def _detect_zh_char_entropy(text: str, lang: str = "") -> float:
    """中文字符级 3-gram 熵（对改写最鲁棒的中文信号）。"""
    if (lang or _detect_language(text)) != "zh":
        return 0.5
    han = _RE_NON_ZH.sub("", text)
    if len(han) < 300:
        return 0.5
    n = 3
    grams = Counter(han[i:i + n] for i in range(len(han) - n + 1))
    total = sum(grams.values())
    entropy = -sum((v / total) * math.log2(v / total)
                   for v in grams.values())
    return round(max(0.0, min(1.0, (11.5 - entropy) / 3.5)), 3)


def _detect_chinese_ai(text: str, lang: str = "") -> float:
    """中文 AI 文本检测（六组中文特有信号，权重按实测区分力分配）。"""
    if (lang or _detect_language(text)) != "zh":
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
    s_boiler = min(density(_ZH_BOILERPLATE) / 6.0, 1.0)
    s_colloq = max(0.0, 1.0 - density(_ZH_COLLOQUIAL) / 4.0)
    s_number = max(0.0, 1.0 - (len(re.findall(r"\d", text)) / n * 1000) / 8.0)
    s_cv = max(0.0, min(1.0, (0.62 - cv) / 0.35))
    s_quote = max(0.0, 1.0 - (len(_RE_ZH_QUOTE.findall(text)) / n * 1000) / 6.0)

    # 权重按"可伪造性"分配：句长节奏最难伪装占大头
    score = (s_tpl * 0.12 + s_jargon * 0.14 + s_boiler * 0.10
             + s_colloq * 0.12 + s_number * 0.10 + s_cv * 0.30
             + s_quote * 0.12)
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
# 统计水印 / SpecDetect / DNA-DetectLLM / 词汇多样性
# ============================================================
def _detect_with_ml(text: str) -> Optional[Dict[str, float]]:
    """用链中全部 ML 检测器各自打分，返回 {模型名: AI 概率}。

    v3.17：推理期受空闲回收守卫保护；改用 inference_mode 微降开销。
    """
    with _model_in_use("ml-detector"):
        models = _load_ml_models()
        if not models:
            return None
        import torch

        # 分块处理（避免超长文本被 512 token 截断）。
        # v3.19【P1】中文没有空格，text.split() 会把全文当成一个
        # 「词」，结果整段塞进单块再被截断，长中文文档只检测了
        # 开头。现按语言自适应：英文按 80 词分块；中文按 160 字符
        # 分块（ModernBERT 512 token ≈ 中文 ~500 字，留安全余量）。
        if _detect_language(text) == "zh":
            chunks = [text[i:i + 160]
                      for i in range(0, len(text), 160)]
            chunks = [c for c in chunks if len(c.strip()) > 10]
        else:
            words = text.split()
            chunks = []
            for i in range(0, len(words), 80):
                chunk = " ".join(words[i:i + 80])
                if len(chunk.strip()) > 10:
                    chunks.append(chunk)
        if not chunks:
            return None

        out: Dict[str, float] = {}
        # 【v3.22 性能】分块前向：块 padding 后形状一致，可拼成 (b, L)
        # 一批算完。逐块前向实测 0.90s（810 词 / 11 块），每批 8 块后
        # 降至 0.28s。分类头取 [CLS]（行首）位置，padding 位置
        # attention_mask=0 不参与注意力，故每块分数与逐条版本一致
        # （实测 7 篇文档最终分逐位不变）。
        ml_batch = max(1, int(os.environ.get("AURORA_ML_BATCH", "8")))
        for name, info in models.items():
            tok_ = info["tokenizer"]
            model_ = info["model"]
            ai_idx = info["ai_idx"]
            scores = []
            with torch.inference_mode():
                for i in range(0, len(chunks), ml_batch):
                    enc = tok_(chunks[i:i + ml_batch], return_tensors="pt",
                               padding=True, truncation=True, max_length=512)
                    logits = model_(**enc).logits
                    probs = torch.softmax(logits, dim=-1)[:, ai_idx]
                    scores.extend(probs.tolist())
            if scores:
                out[name] = sum(scores) / len(scores)
        return out or None


def _release_mlm_model() -> None:
    """回收伪困惑度 MLM 模型（ModernBERT-base）。"""
    global _mlm_model, _mlm_tokenizer, _mlm_load_done
    _mlm_model = None
    _mlm_tokenizer = None
    _mlm_load_done = False  # 复位：下次访问自动重载
    _empty_cuda_cache()
    if _VERBOSE:
        print("[aurora] 已回收闲置模型：mlm-perplexity（ModernBERT-base）",
              file=sys.stderr)


def _load_mlm_model():
    """懒加载 MLM 模型（v3.17：加锁防并发重载 + 空闲回收注册）。

    v3.18【P1-1】：失败即粘性（_mlm_load_done 置位后不再重试）。
    """
    global _mlm_model, _mlm_tokenizer, _mlm_load_done
    if _mlm_load_done:
        if _mlm_model is not None:
            _touch_model("mlm-perplexity")
        return _mlm_model, _mlm_tokenizer

    with _mlm_load_lock:
        if _mlm_load_done:
            if _mlm_model is not None:
                _touch_model("mlm-perplexity")
            return _mlm_model, _mlm_tokenizer
        _mlm_load_done = True
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch  # noqa: F401  # noqa: F401 — 故意提前导入：torch 缺失时快速失败
        except Exception as e:
            _record_gap("mlm-perplexity", "transformers/torch 未安装或不可用",
                        persistent=True)
            if _VERBOSE:
                print(f"[aurora] transformers/torch 不可用，跳过伪困惑度：{e}",
                      file=sys.stderr)
            return None, None
        try:
            if _hf_model_cached(_MLM_MODEL_NAME):
                try:
                    _mlm_tokenizer = AutoTokenizer.from_pretrained(
                        _MLM_MODEL_NAME, local_files_only=True)
                    _mlm_model = AutoModelForMaskedLM.from_pretrained(
                        _MLM_MODEL_NAME, local_files_only=True)
                except Exception:
                    _mlm_tokenizer = AutoTokenizer.from_pretrained(
                        _MLM_MODEL_NAME)
                    _mlm_model = AutoModelForMaskedLM.from_pretrained(
                        _MLM_MODEL_NAME)
            else:
                if not _hf_reachable():
                    _ensure_hf_endpoint()
                _mlm_tokenizer = AutoTokenizer.from_pretrained(_MLM_MODEL_NAME)
                _mlm_model = AutoModelForMaskedLM.from_pretrained(
                    _MLM_MODEL_NAME)
            _mlm_model.eval()
            if _VERBOSE:
                print(f"[aurora] 已加载伪困惑度模型：{_MLM_MODEL_NAME}",
                      file=sys.stderr)
        except Exception as e:
            _record_gap("mlm-perplexity",
                        f"伪困惑度模型加载失败（{_MLM_MODEL_NAME}）",
                        persistent=True)
            if _VERBOSE:
                print(f"[aurora] 伪困惑度模型 {_MLM_MODEL_NAME} 加载失败：{e}",
                      file=sys.stderr)
            _mlm_model = None
            return None, None

    _touch_model("mlm-perplexity")
    _register_model("mlm-perplexity", _release_mlm_model)
    _start_model_reaper()
    return _mlm_model, _mlm_tokenizer


def _detect_mlm_perplexity(text: str) -> Optional[Tuple[float, float]]:
    """用 MLM 头算伪困惑度，返回 (AI 概率, 原始伪困惑度)。

    越低困惑度（token 越可预测）越像 AI。对「表演式不完美」拟人 AI
    尤其有效——表面有错字，但底层分布仍比模仿的真人更平滑。

    v3.17 批量推理：max_masks 个 mask 窗口合并为 (B, win) 张量按块
    一次前向，前向次数从 max_masks 次降到 ceil(max_masks / batch) 次。

    v3.18 修复记录：
      - 【P0】批量窗口构造必须把目标 token 替换成 [MASK]。
        v3.17 漏了这一步：模型看到完整原文等于"背诵"，CE≈0、
        pseudo_pp≈1、引擎恒定输出 ≈0.3 中性噪声，彻底失效。
      - 【P1-2】snippet 改为按字符截断再 tokenize。中文没有空格，
        原先 text.split() 后 words[:400] 对中文要么是全文、要么
        覆盖极小，len(words)<20 门槛对中文也过严；现在中英文统一
        走字符门槛 + token 层自检。
      - del 扩展到 target_logits / nll，(b, win, vocab) 大张量当块即释。
    """
    # v3.21【P0】import torch 移到模型判空之后：此前这行在
    # _load_mlm_model() 之前，torch 未安装时 ImportError 直接冒泡
    # 到 _engine_stage_cached → run_check，整条管线崩溃——
    # _load_mlm_model 里的优雅降级（返回 (None, None)）根本来不及
    # 执行。纯统计用户 / CI / Docker slim 镜像一跑就炸。
    with _model_in_use("mlm-perplexity"):
        model, tok = _load_mlm_model()
        if model is None or tok is None:
            return None
        import torch  # 模型已就绪，torch 必然可用
        if tok.mask_token_id is None:  # MLM 模型必有的防御性检查
            return None

        # 【P1-2】先按字符截断，再 tokenize——逻辑直接对 token 层，
        # 不依赖空格分词。8K 字符足够 512 token（英文 ~4 字/token，
        # 中文 ~1 字/token）。
        if len(text) < 200:
            return None
        enc = tok(text[:8000], return_tensors="pt",
                  truncation=True, max_length=512)
        ids = enc.input_ids[0]
        special = set(tok.all_special_ids)

        win = 64
        max_masks = int(os.environ.get("AURORA_PPL_MAX_MASKS", "30"))
        # 默认 16：实测（810 词文档、30 个 mask 窗口，同机同进程对比）
        #   batch=1 10.11s | 2 → 6.90s | 4 → 5.81s | 8 → 5.12s
        #   16 → 3.81s | 30 → 3.40s
        # 再往上收益趋平，而峰值 logits 按 (b, win, vocab) 线性上涨
        # （b=30 单块 384MB），故 16 兼顾速度与低配机器。
        batch_size = max(1, int(os.environ.get("AURORA_PPL_BATCH", "16")))

        n = len(ids)
        if n < 30:
            return None
        positions = [i for i in range(n) if int(ids[i]) not in special]
        if len(positions) < 8:
            return None
        # 均匀采样 max_masks 个位置，覆盖全文
        # （只取开头会撞上标题/固定句式等最可预测位置，严重失真）
        if len(positions) > max_masks:
            gap = len(positions) / max_masks
            sampled = [positions[min(int(i * gap), len(positions) - 1)]
                       for i in range(max_masks)]
        else:
            sampled = positions

        # ===== 一次性构造全部 (B, win) mask 窗口与注意力掩码 =====
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        B = len(sampled)
        batch = torch.full((B, win), pad_id, dtype=torch.long)
        attn = torch.zeros((B, win), dtype=torch.long)
        j_idx = torch.zeros(B, dtype=torch.long)   # mask 在窗口内的位置
        t_idx = torch.zeros(B, dtype=torch.long)   # 被遮住的原 token id
        for row, pos in enumerate(sampled):
            start = max(0, pos - win // 2)
            window = ids[start:start + win].clone()   # clone：别改到 ids
            j = pos - start
            t_idx[row] = int(window[j])               # 记下原 token 作为 CE 目标
            window[j] = tok.mask_token_id             # ★ P0 关键：换成 [MASK]
            L = window.shape[0]
            batch[row, :L] = window
            attn[row, :L] = 1                         # 右侧 padding 全部屏蔽
            j_idx[row] = j

        # ===== 分块前向：每块一次前向替代逐 mask 的 B 次前向 =====
        nlls: List[float] = []
        with torch.inference_mode():
            for i in range(0, B, batch_size):
                b_in = batch[i:i + batch_size]
                b_attn = attn[i:i + batch_size]
                b_j = j_idx[i:i + batch_size]
                b_t = t_idx[i:i + batch_size]
                logits = model(input_ids=b_in, attention_mask=b_attn).logits
                rows = torch.arange(b_in.shape[0])
                target_logits = logits[rows, b_j]          # (b, vocab)
                # cross_entropy 等价于 -log_softmax(...)[target]，一行算完
                nll = torch.nn.functional.cross_entropy(
                    target_logits, b_t, reduction="none")  # (b,)
                nlls.extend(nll.tolist())
                # 尽早释放 (b, win, vocab) ~百 MB 张量
                del logits, target_logits, nll

        if not nlls:
            return None
        pseudo_pp = math.exp(sum(nlls) / len(nlls))

        # 标定参数优先级：环境变量 > corpus/calibration.json > 默认值
        _calib = _load_corpus_calibration() or {}
        _pp_calib = _calib.get("pseudo_perplexity", {}) or {}
        PIVOT = float(os.environ.get(
            "AURORA_PPL_PIVOT", str(_pp_calib.get("pivot", 1650.0))))
        SCALE = float(os.environ.get(
            "AURORA_PPL_SCALE", str(_pp_calib.get("scale", 1950.0))))
        ai_score = 1.0 / (1.0 + math.exp((pseudo_pp - PIVOT) / SCALE))
        return round(float(ai_score), 3), round(float(pseudo_pp), 2)


# ============================================================
# 本地文档相似度比对
# ============================================================
def _text_ngrams(t: str, n: int = 3) -> set:
    """取 n-gram 集合（供 Jaccard 相似度使用）。

    v3.20【P0-5】中文没有空白分词：旧版 text.lower().split() 把整段
    中文当成一个 token，n-gram 集合极小甚至为空，相似度恒 0——
    compare_with_reference 与段落自重叠对中文是装饰性的。CJK 为主时
    改用字符 bigram（n 参数仅对英文路径生效），中文获得有效读数。
    """
    t = t.lower()
    han_chars = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if han_chars >= max(len(t) * 0.25, 8):
        return {t[i:i + 2] for i in range(len(t) - 1)
                if not t[i].isspace() and not t[i + 1].isspace()}
    words = t.split()
    return {" ".join(words[i:i + n])
            for i in range(len(words) - n + 1)}


def _compute_text_similarity(text1: str, text2: str) -> float:
    """计算两段文本的相似度（基于 3-gram 重叠的 Jaccard）。"""
    ngrams1 = _text_ngrams(text1)
    ngrams2 = _text_ngrams(text2)
    if not ngrams1 or not ngrams2:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


def compare_with_reference(doc_path: str, ref_path: str) -> dict:
    """比对当前文档和参考文档的相似度。"""
    doc_text = extract_text_from_file(doc_path)
    ref_text = extract_text_from_file(ref_path)

    doc_prose, _ = _extract_prose_only(doc_text)
    ref_prose, _ = _extract_prose_only(ref_text)

    overall_similarity = _compute_text_similarity(doc_prose, ref_prose)

    doc_sentences = _split_sentences(doc_prose)
    ref_sentences = _split_sentences(ref_prose)
    # v3.19【P2-2】参考句 3-gram 集合只建一次（原先 O(n×m) 次重复
    # 分词建集），长文档比对提速约 2 倍；结果与旧版完全一致。
    ref_ngram_sets = [_text_ngrams(s) for s in ref_sentences]
    overlapping_sentences = []
    for doc_sent in doc_sentences:
        doc_ngrams = _text_ngrams(doc_sent)
        if not doc_ngrams:
            continue
        for ref_sent, ref_ngrams in zip(ref_sentences, ref_ngram_sets):
            if not ref_ngrams:
                continue
            sim = len(doc_ngrams & ref_ngrams) / len(doc_ngrams | ref_ngrams)
            if sim > 0.5:
                overlapping_sentences.append({
                    "doc_sentence": doc_sent,
                    "ref_sentence": ref_sent,
                    "similarity": sim,
                })
                break
    return {
        "overall_similarity": overall_similarity,
        "overlapping_count": len(overlapping_sentences),
        "overlapping_sentences": overlapping_sentences[:10],
    }


# ============================================================
# 基础文本工具
# ============================================================
@lru_cache(maxsize=512)
def _word_count(text: str) -> float:
    """估算词数，兼容中英文（中文按 1.6 字/词折算）。

    【v3.22 记忆化】单次检测里被 20+ 个引擎各调一次，同一文本重复
    扫两遍全文字符类；lru_cache 命中即 0 扫描（key 就是原文，
    str 自带哈希缓存）。
    """
    en = len(_RE_EN_WORD.findall(text))
    zh = len(_RE_ZH_CHAR.findall(text))
    return en + zh / 1.6


# size 128（原 64）：v3.23 加了 min_len 参数，key 从 text 变成
# (text, min_len)，打分路径（10）与校验路径（4）各占一个槽位。
@lru_cache(maxsize=128)
def _split_sentences_cached(text: str, min_len: int = 10) -> tuple:
    """句子切分核心实现（记忆化，返回 tuple 保证缓存安全）。

    保留句末标点；保护 e.g. / et al. 等常见缩写。

    Args:
        text: 待切分文本。
        min_len: 最短句字符数，短于等于此值的片段直接丢弃。
            默认 10 是**打分口径**——所有引擎（句长 CV、burstiness、
            句首模式、滑窗）都建立在它上面，动它等于改分数。
            validate_text 这类"数够不够句"的完整性检查传 4，
            见下方说明。

    【v3.23 P1-B】原先这里写死 >10，看起来只是"过滤碎片"，实际
    造成两处中文误伤：
      1) 打分侧：10 个中文字符 ≈ 6 个词，7~10 字的中文短句被整句
         丢弃，句长 CV 因此系统性偏小 → 中文被判更像 AI（假阳性）。
      2) 校验侧：validate_text 用"切完还剩几句"判是否够 5 句，
         短句被丢后正文其实完整却报「完整句子仅 N 句」而拒检。
    第 2 条是真 bug，第 1 条则**没有真人中文基线可校准**——实测
    把阈值降到 6 会让 AI 中文样例掉 12 个百分点（方向错误）。
    所以这里保留 10 作为打分口径（分数零漂移），把宽松口径
    下放到 validate_text 的句数校验里（min_len=4）。
    """
    text = _RE_NL.sub(" ", text)
    protected = _RE_ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), text)
    parts = _RE_SENTENCE.findall(protected)
    return tuple(p.replace(_DOT, ".").strip()
                 for p in parts if len(p.strip()) > min_len)


def _split_sentences(text: str) -> List[str]:
    """句子切分，返回可变 list。

    【v3.22 记忆化】单次检测里切句被 17+ 次调用（句长分布 /
    burstiness / 句首模式 / 段落节奏…），每次都是全文两轮正则替换
    + 一次 findall。拆成 cache(tuple) + 薄包装(list)：命中只需把
    tuple 转成 list，调用方 sort/append 的行为不变。
    """
    return list(_split_sentences_cached(text))


def _looks_like_reference_line(line: str) -> bool:
    """判断一段是否像参考文献条目（不依赖 "References" 标题）。"""
    if _RE_REF_DOI.search(line):
        return True
    if _RE_REF_APA.match(line) and _RE_REF_YEAR.search(line):
        return True
    if _RE_REF_VOL.search(line):
        return True
    return False


def _is_front_matter_line(line: str) -> bool:
    """判断是否为封面/标题短行（非正文）：短且无句末标点。

    v3.21【P1】改用 _RE_SENT_END_CJK：中文正文行以「。」收尾同样
    视为句子，不再被当元数据吞掉。"""
    if len(line) >= 60:
        return False
    if _RE_SENT_END_CJK.search(line):
        return False
    return True


def _extract_prose_only(text: str) -> Tuple[str, str]:
    """自动排除引用/参考文献与封面元数据，只返回正文。"""
    ref_start = len(text)
    for pat in _RE_REF_HEADINGS:
        m = pat.search(text)
        if m:
            ref_start = min(ref_start, m.start())
    prose = text[:ref_start]
    references = text[ref_start:]

    prose_lines = []
    in_front_matter = True
    for line in prose.split('\n'):
        line = line.strip()
        if not line:
            continue
        if _RE_DOI_URL.match(line):
            continue
        if _RE_DOI_PREFIX.match(line):
            continue
        if len(line) < 60 and line.isupper():
            continue
        if _looks_like_reference_line(line):
            continue
        if in_front_matter and _is_front_matter_line(line):
            continue
        in_front_matter = False
        prose_lines.append(line)
    return '\n'.join(prose_lines), references


def _compute_perplexity_ngram(text: str, n: int = 3) -> float:
    """基于 n-gram 的简化困惑度计算（不需要下载大模型）。"""
    words = text.lower().split()
    if len(words) < n + 5:
        return 0.5
    ngrams: Counter = Counter()
    unigrams: Counter = Counter()
    for w in words:
        unigrams[w] += 1
    for i in range(len(words) - n + 1):
        ngrams[tuple(words[i:i + n])] += 1

    log_prob_sum = 0.0
    count = 0
    total_words = len(words)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        prev = words[i]
        ngram_count = ngrams.get(ngram, 0)
        unigram_count = unigrams.get(prev, 0)
        if unigram_count > 0:
            prob = (ngram_count + 0.5) / (unigram_count + 0.5 * len(unigrams))
        else:
            prob = 1.0 / total_words
        log_prob_sum -= math.log(prob)
        count += 1
    if count == 0:
        return 0.5
    avg_log_prob = log_prob_sum / count
    ai_prob = max(0.1, min(0.9, (4.5 - avg_log_prob) / 2.0 + 0.3))
    return round(ai_prob, 3)


def _sliding_window_detection(sentences: list, window_size: int = 7) -> list:
    """段落级滑动窗口检测：带上下文的逐句分数。"""
    if len(sentences) < window_size:
        return [0.5] * len(sentences)
    window_scores = []
    for i in range(len(sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        lengths = [int(_word_count(s)) for s in sentences[start:end]]
        avg = sum(lengths) / len(lengths)
        std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
        burstiness = std / (avg + 1)
        uniformity = 1 - (std / avg) if avg > 0 else 0
        if burstiness < 0.35 and uniformity > 0.65:
            window_ai = 0.75
        elif burstiness < 0.55:
            window_ai = 0.55
        else:
            window_ai = 0.25
        window_scores.append(window_ai)
    return window_scores


# ============================================================
# 输入质量校验
# ============================================================
def validate_text(text: str) -> Tuple[bool, str]:
    """输入质量校验。

    统计型检测器对过短、重复、非自然语言的文本会给出看似确定的分数，
    这在学术诚信场景里比"报错"更危险——不合格时直接拒绝判定。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, "文本为空"

    chars = len(stripped)
    wc = _word_count(stripped)
    if chars < 200:
        return False, (f"文本过短（{chars} 字符）；"
                       "至少需要 200 字符，约 3~4 个段落")
    if wc < 50:
        return False, f"篇幅过短（约 {int(wc)} 词）；至少需要 50 词"

    # 【v3.23 P1-B】数句用 min_len=4 的宽松口径：打分侧的 10 字符
    # 阈值是为了滤掉碎片，不该拿来判"这篇文章是不是写完整了"。
    # 中文 10 字符 ≈ 6 词，正文完整却因短句被丢而报「句子不足」。
    sentences = [s for s in _split_sentences_cached(stripped, 4)
                 if int(_word_count(s)) >= 2]
    if len(sentences) < 5:
        return False, (f"完整句子仅 {len(sentences)} 句；"
                       "至少需要 5 句才能估计句长分布")

    letters = sum(map(str.isalpha, stripped))
    if letters / chars < 0.40:
        return False, "字母比例过低，疑似非自然语言文本（表格、代码或符号串）"

    # 中英文分别统计词汇多样性（中文按字符二元组近似）
    # 【v3.16 bug 修复】原版 diversity<0.15 检查在 if/else 之外重复执行，
    # 导致中文分支被英文阈值二次套用。现已收敛进各自分支。
    if _detect_language(stripped) == "zh":
        # v3.20【P1-11】旧阈值 diversity（唯一二元组占比）< 0.05 在
        # 中文下形同虚设：常用汉字 ~3000，二元组空间 ~900 万，正常
        # 文本 diversity 几乎总 > 0.3，同一句反复粘贴都难触发。
        # 改用 5-gram 重复率：重复占比 > 0.3 才判「反复粘贴」。
        chars_only = _RE_NON_ZH.sub("", stripped)
        grams = [chars_only[i:i + 5] for i in range(len(chars_only) - 4)]
        if grams:
            dup_ratio = (len(grams) - len(set(grams))) / len(grams)
            if dup_ratio > 0.3:
                return False, (f"内容重复率 {dup_ratio:.0%} 异常"
                               "（5-gram 层面）；疑似同一句反复粘贴，"
                               "不能作为检测对象")
    else:
        words = stripped.split()
        uniq = len({w.lower().strip(".,!?;:") for w in words})
        diversity = uniq / max(len(words), 1)
        if diversity < 0.15:
            return False, (f"词汇多样性仅 {diversity:.0%}，重复度异常；"
                           "疑似同一句反复粘贴，不能作为检测对象")

    return True, ""


# ============================================================
# 引擎注册表（模块级：不再每次检测重建）
# ============================================================
ENGINE_SPEC: Dict[str, EngineSpec] = {
    # —— 实测区分力强 ——
    "ml-detector": EngineSpec(+1, 0.25, "ML 检测器（ModernBERT 主力，需联网下载）"),
    "mlm-perplexity": EngineSpec(+1, 0.15, "MLM 伪困惑度（专攻表演式不完美拟人 AI，需本地 ModernBERT-base）"),
    "lmscan": EngineSpec(+1, 0.15, "外部 CLI"),
    "linguistic-markers": EngineSpec(+1, 0.05, "AI 高频词表（3.13 校准：学术词汇与 AI 词表重叠，降权）"),
    "formulaic-patterns": EngineSpec(+1, 0.12, "公式化表达"),
    "emotion-curve": EngineSpec(+1, 0.10, "情感曲线平坦度"),
    "transition-density": EngineSpec(+1, 0.08, "过渡词密度"),
    "sentiment-hedging": EngineSpec(+1, 0.08, "对冲语密度"),
    # —— 中等 ——
    "sentence-length": EngineSpec(+1, 0.07, "句长分布"),
    "burstiness-stat": EngineSpec(+1, 0.05, "突发度（3.13 校准：真人 p90=0.55，微降）"),
    "statistical-baseline": EngineSpec(+1, 0.03, "统计基线（3.13 校准：近 0.5 无区分力，降权）"),
    "sliding-window": EngineSpec(+1, 0.06, "段落滑动窗口"),
    "human-imperfection": EngineSpec(+1, 0.06, "人味特征（v2 重写）"),
    "paragraph-rhythm": EngineSpec(+1, 0.07, "段落长度均匀度（跨语言）"),
    "zh-char-entropy": EngineSpec(+0, 0.0, "仅中文启用，见语言自适应"),
    "lm-perplexity": EngineSpec(+1, 0.22, "语言模型困惑度（需本地模型）"),
    "formulaic-markers": EngineSpec(+1, 0.05, "套话标记"),
    "vocabulary-richness": EngineSpec(+1, 0.03, "词汇丰富度（3.13 校准：真人均值 0.497，降权）"),
    # —— 弱 ——
    "starter-patterns": EngineSpec(+1, 0.04, "句首模式"),
    "specific-details": EngineSpec(+1, 0.02, "具体细节密度（LLM 已普遍生成数值，弱信号）"),
    "adversarial-perturbation": EngineSpec(+1, 0.04, "对抗扰动"),
    "translationese": EngineSpec(+1, 0.06, "翻译腔检测（介词前置从句 + 冗长被动，专治 DeepL 洗稿）"),
    "compressibility": EngineSpec(+1, 0.0, "压缩率（3.13 校准：真人学术文献均值 0.74，方向失效，归零）"),
    "pawn-perplexity": EngineSpec(+1, 0.03, "PAWN"),
    "ngram-perplexity": EngineSpec(+1, 0.0, "对短文本不可靠（3.13 校准：恒定输出 0.1，无区分力，归零）"),
    # —— 已实测失效，权重置 0（仍显示，便于对照）——
    "self-references": EngineSpec(0, 0.0, "AI 同样高频使用 I think，无区分力"),
    "logical-connectors": EngineSpec(0, 0.0, "几乎恒为 1.0"),
    "vocabulary-diversity": EngineSpec(0, 0.0, "方向相反"),
    "structural-patterns": EngineSpec(0, 0.0, "区分力弱"),
    "structured-shape": EngineSpec(0, 0.0, "恒为 0"),
    "hidden-characters": EngineSpec(0, 0.0, "恒为 0"),
    "punctuation-patterns": EngineSpec(0, 0.0, "恒为 1.0"),
    "statistical-watermark": EngineSpec(0, 0.0, "方向相反"),
    "spec-detect": EngineSpec(0, 0.0, "近乎恒定"),
    "dna-detect": EngineSpec(0, 0.0, "恒为 1.0，实现有误"),
    "chinese-ai": EngineSpec(0, 0.0, "仅中文文本生效"),
}

# 中文文本下失效的英文词表类引擎
_EN_ONLY_ENGINES = (
    "linguistic-markers", "formulaic-markers", "formulaic-patterns",
    "sentiment-hedging", "transition-density", "starter-patterns",
    "self-references", "logical-connectors", "ngram-perplexity",
    "pawn-perplexity", "adversarial-perturbation", "translationese",
    # v3.20【P0-3】英文词表/拉丁字形依赖，中文下要么恒 1.0 要么恒 0，
    # 降权 ×0.2 抑制假信号
    "emotion-curve", "specific-details", "structural-patterns",
)

# 中文自适应变体缓存（懒建，只建一次）
_ENGINE_SPEC_ZH: Optional[Dict[str, EngineSpec]] = None


def _engine_spec_for(lang: str) -> Dict[str, EngineSpec]:
    """按语言返回引擎注册表；中文变体只构建一次并缓存。"""
    global _ENGINE_SPEC_ZH
    if lang != "zh":
        return ENGINE_SPEC
    if _ENGINE_SPEC_ZH is None:
        spec = dict(ENGINE_SPEC)
        for k in _EN_ONLY_ENGINES:
            if k in spec:
                d, w, n = spec[k]
                spec[k] = EngineSpec(d, round(w * 0.2, 4),
                                     n + "（对中文无效，降权）")
        spec["chinese-ai"] = EngineSpec(+1, 0.24, "中文 AI 特征")
        spec["zh-char-entropy"] = EngineSpec(+1, 0.20, "中文字符熵（抗改写）")
        _ENGINE_SPEC_ZH = spec
    return _ENGINE_SPEC_ZH


def _compute_engine_scores(prose_text: str,
                           lang: str = "") -> Tuple[dict, list, list, dict]:
    """跑全部「有权重」的检测引擎，只产出原始读数，不做任何判断。

    v3.18【P2-1】：注册表里权重 0 的引擎直接跳过计算——此前它们
    照样跑满 regex/zlib，结果却只进「已弃用」面板，纯属浪费。
    跳过名单按语言自适应后的注册表现算（中文下 chinese-ai /
    zh-char-entropy 会被激活，不能写死成静态集合）。

    Returns:
        (engine_scores, sentences, window_scores, ml_detail)
    """
    _lang = lang or _detect_language(prose_text)
    spec_table = _engine_spec_for(_lang)
    _skip = frozenset(k for k, v in spec_table.items() if v[1] <= 0)

    engine_scores: Dict[str, float] = {}

    lmscan_score = _detect_with_lmscan(prose_text)
    if lmscan_score is not None:
        engine_scores["lmscan"] = lmscan_score

    sentences = _split_sentences(prose_text)
    window_scores = _sliding_window_detection(sentences, window_size=5)
    engine_scores["sliding-window"] = (
        sum(window_scores) / len(window_scores) if window_scores else 0.5)

    pp_score = _detect_perplexity_burstiness(prose_text)
    if pp_score is not None:
        engine_scores["burstiness-stat"] = pp_score

    baseline_score = _detect_with_baseline(prose_text)
    engine_scores["statistical-baseline"] = baseline_score

    # —— 纯文本统计类引擎（统一走权重闸门）——
    _text_engines = (
        ("formulaic-markers", _detect_formulaic_markers),
        ("transition-density", _detect_transition_density),
        ("compressibility", _detect_compressibility),
        ("translationese", _detect_translationese),
        ("formulaic-patterns", _detect_formulaic),
        ("linguistic-markers", _detect_linguistic_markers),
        ("vocabulary-richness", _detect_vocabulary_richness),
        ("sentiment-hedging", _detect_sentiment_hedging),
        ("human-imperfection", _detect_human_imperfection),
        ("paragraph-rhythm", _detect_paragraph_rhythm),
        ("emotion-curve", _detect_emotion_curve),
        ("specific-details", _detect_specific_details),  # P2-2：签名已去 lang
        ("starter-patterns", _detect_starter_patterns),
        ("sentence-length", _detect_sentence_length_distribution),
        ("pawn-perplexity", _detect_pawn_perplexity),
        ("adversarial-perturbation", _detect_adversarial_perturbation),
        ("ngram-perplexity", lambda t: _compute_perplexity_ngram(t, n=3)),
    )
    for name, fn in _text_engines:
        if name in _skip:  # P2-1：权重 0 直接跳过，不白算
            continue
        engine_scores[name] = fn(prose_text)

    # —— 中文专属（英文下权重 0，自动跳过）——
    for name, fn in (("zh-char-entropy", _detect_zh_char_entropy),
                     ("chinese-ai", _detect_chinese_ai)):
        if name in _skip:
            continue
        engine_scores[name] = fn(prose_text, lang=_lang)

    # —— ML 检测器（链中全部模型各自打分）——
    ml_detail: dict = {"scores": {}, "mean": None, "spread": 0.0, "n": 0}
    ml_result = None if "ml-detector" in _skip else _detect_with_ml(prose_text)
    if ml_result:
        ml_detail["scores"] = ml_result
        vals = list(ml_result.values())
        ml_detail["n"] = len(vals)
        ml_detail["mean"] = sum(vals) / len(vals)
        ml_detail["spread"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        engine_scores["ml-detector"] = ml_detail["mean"]

    # —— MLM 伪困惑度 ——
    if "mlm-perplexity" not in _skip:
        pp_res = _detect_mlm_perplexity(prose_text)
        if pp_res is not None:
            _pp_score, _pp_val = pp_res
            engine_scores["mlm-perplexity"] = _pp_score
            ml_detail["pseudo_perplexity"] = _pp_val

    # —— 语言模型困惑度（本地 Qwen，仅中文）——
    if "lm-perplexity" not in _skip:
        ppl_score = _detect_lm_perplexity(prose_text, lang=_lang)
        if ppl_score is not None:
            engine_scores["lm-perplexity"] = ppl_score

    return engine_scores, sentences, window_scores, ml_detail


# ============================================================
# 引擎计算层缓存（v3.20【P1-9】引擎/聚合分层）
# ============================================================
@lru_cache(maxsize=16)
def _engine_stage_cached(prose_text: str):
    """按 text 缓存昂贵的引擎计算结果（ML 前向、统计引擎全套）。

    v3.20【P1-9】：此前整个 AIResult 按 (text, mode) 缓存，
    normal+strict 同文档会把所有引擎跑两遍（ML 前向成本最高）。
    现在引擎层只按 text 缓存一份，聚合层各自消费。
    返回冻结元组，防止下游改动污染缓存对象。
    """
    engine_scores, sentences, window_scores, ml_detail = \
        _compute_engine_scores(prose_text)
    ml_frozen = (
        tuple(ml_detail.get("scores", {}).items()),
        ml_detail.get("mean"),
        ml_detail.get("spread"),
        ml_detail.get("n"),
        ml_detail.get("pseudo_perplexity"),
    )
    # v3.21【P1】把本次运行登记的非持久 gap（如 lmscan 未安装）一并
    # 冻结进缓存：命中时引擎不重跑，没有快照的话这些登记不会重现，
    # missing_engines 展示会缺项（透明度打折）。
    run_gaps = tuple(sorted(_current_gaps().items()))
    return (tuple(engine_scores.items()), tuple(sentences),
            tuple(window_scores), ml_frozen, run_gaps)


def _thaw_stage(stage) -> Tuple[Dict[str, float], List[str],
                                List[float], Dict]:
    """把冻结的引擎层缓存还原成普通可读结构。

    Side effects:
        会把缓存里记录的「引擎缺失」重新登记到当前检测上下文
        （_record_gap）。normal / strict 复用同一份引擎缓存时，
        报告的 missing_engines 面板因此不会丢；已登记的不重复写。

    Args:
        stage: _engine_stage_cached 返回的冻结 5 元组
            (engine_scores, sentences, window_scores, ml_detail, gaps)。

    Returns:
        (engine_scores, sentences, window_scores, ml_detail) 四元组。
        前三项都是可变副本（dict / list），调用方可原地调权。
    """
    engine_scores = dict(stage[0])
    ml_detail = {"scores": dict(stage[3][0]), "mean": stage[3][1],
                 "spread": stage[3][2], "n": stage[3][3]}
    if stage[3][4] is not None:
        ml_detail["pseudo_perplexity"] = stage[3][4]
    # v3.21【P1】重放缓存的 gap 登记（持久 gap 已在上下文种子里的
    # 跳过，避免重复写锁）。
    for _name, _reason in stage[4]:
        if _name not in _current_gaps():
            _record_gap(_name, _reason)
    return engine_scores, list(stage[1]), list(stage[2]), ml_detail


# ============================================================
# AI 检测主流程（引擎层缓存 + 聚合层实时计算）
# ============================================================
def _run_ai_detection_cached(text: str,
                             mode: str = "normal") -> AIResult:
    """AI 检测内部实现。

    注意：
      - v3.20【P1-9】本函数不再缓存 AIResult：昂贵引擎层由
        _engine_stage_cached 按 text 缓存，聚合层每次新建返回——
        调用方拿到的对象天然独占，可放心原地修改。
      - mode 不再进缓存 key：normal/strict 共享引擎计算，
        仅聚合策略不同。

    mode:
        "normal"：稳健聚合（0.5×统计 + 0.5×主力，阈值 0.58/0.33）。
        "strict"：自检找茬模式，取消离群剔除，均值 >0.30 即标红。
    """
    strict = (mode == "strict")

    # 每次跑新文档前重置当前上下文的缺失记录（并发隔离）
    with _PERSISTENT_GAPS_LOCK:
        _ENGINE_GAPS.set(dict(_PERSISTENT_GAPS))

    # 步骤 0: 自动排除引用/参考文献部分
    prose_text, _references = _extract_prose_only(text)
    if len(prose_text.strip()) < 50:
        prose_text = text

    _lang = _detect_language(prose_text)
    spec_table = _engine_spec_for(_lang)

    engine_scores, sentences, window_scores, ml_detail = \
        _thaw_stage(_engine_stage_cached(prose_text))

    # ---- 加权聚合 ----
    adopted: Dict[str, Tuple[float, float]] = {}
    rejected: Dict[str, str] = {}
    # 【P2-1 配套】权重 0 的引擎已在 _compute_engine_scores 跳过计算，
    # 这里补登记进 rejected，「已弃用引擎」报告面板不失明。
    for _name, _spec in spec_table.items():
        if _spec[1] <= 0:
            rejected[_name] = "实测无效（权重 0，已跳过计算）"
    for name, score in engine_scores.items():
        spec = spec_table.get(name)
        if spec is None:
            rejected[name] = "未注册"
            continue
        direction, w, _note = spec
        if direction == 0 or w <= 0:
            rejected[name] = "实测无效"
            continue
        s = score if direction > 0 else 1.0 - score
        # 只把"精确 0.5"视为引擎未触发（0/1 是合法饱和信号）
        if abs(s - 0.5) < 1e-9 and not strict:
            w *= 0.15
        adopted[name] = (s, w)

    if adopted:
        tw = sum(w for _, w in adopted.values())
        ws = sum(s * w for s, w in adopted.values())
        mean1 = ws / tw if tw else 0.5

        if strict:
            # 排雷视角不剔除离群：高离群往往正是"像 AI"的强信号
            filt = adopted
        else:
            filt = {k: v for k, v in adopted.items()
                    if abs(v[0] - mean1) <= 0.45}
            if not filt:
                filt = adopted

        tw = sum(w for _, w in filt.values())
        ws = sum(s * w for s, w in filt.values())
        stat_mean = ws / tw if tw else 0.5

        wvar = (sum(w * (s - stat_mean) ** 2 for s, w in filt.values()) / tw
                if tw else 0)
        wstd = wvar ** 0.5
        confidence = ("high" if wstd < 0.15
                      else "medium" if wstd < 0.25 else "low")
    else:
        stat_mean, wstd, confidence = 0.5, 0.0, "low"

    # ---- v3.3：ML 锚定 + 困惑度反向锚定 ----
    _scores = ml_detail.get("scores", {})
    ml_n = ml_detail.get("n", 0)
    _primary_name = (_ML_MODEL_CHAIN[0]
                     if _ML_MODEL_CHAIN[0] in _scores
                     else (next(iter(_scores)) if _scores else None))
    primary = _scores.get(_primary_name) if _primary_name else None
    ml_disagreement = False
    ml_note = ""
    aggregation: List[str] = []

    if primary is not None:
        if strict:
            final_score = max(stat_mean, primary)
            aggregation.append(
                f"① 统计引擎加权均值 = **{stat_mean:.2f}**"
                f"（{len(adopted)} 个引擎，严格模式不剔除离群）")
            aggregation.append(
                f"② 主力检测器 {(_primary_name or '').split('/')[-1] or 'ML'}"
                f" = **{primary:.2f}**")
            aggregation.append(
                f"③ 严格模式：终分 = max(统计, 主力) = "
                f"max({stat_mean:.2f}, {primary:.2f}) = **{final_score:.3f}**"
                "（不取平均，谁凶听谁的）")
        else:
            final_score = 0.5 * stat_mean + 0.5 * primary
            aggregation.append(
                f"① 统计引擎加权均值 = **{stat_mean:.2f}**"
                f"（{len(adopted)} 个引擎）")
            aggregation.append(
                f"② 主力检测器 {(_primary_name or '').split('/')[-1] or 'ML'}"
                f" = **{primary:.2f}**")
            aggregation.append(
                f"③ ML 锚定：终分 = 0.5×统计 + 0.5×主力 = "
                f"0.5×{stat_mean:.2f} + 0.5×{primary:.2f} = "
                f"**{final_score:.3f}**")
    else:
        final_score = stat_mean

    # 主力分类器高于"人"基线时，绝不自信判 human
    if primary is not None and primary > 0.33 and final_score < 0.33:
        aggregation.append(
            f"④ 安全下限：主力 {primary:.2f} 高于「纯人」基线 0.33，"
            "终分不得低于 mixed 带 → 抬至 0.34")
        final_score = 0.34

    # 困惑度反向锚定：对抗统计引擎"集体盲区"
    pp_score = engine_scores.get("mlm-perplexity")
    _pp_thresh = 0.50 if strict else 0.60
    if pp_score is not None and pp_score >= _pp_thresh:
        if strict:
            if pp_score > final_score:
                aggregation.append(
                    f"⑤ 严格模式困惑度反向锚定：伪困惑度引擎给出 {pp_score:.2f}"
                    f"（>=0.50，指 AI），max 覆盖上一步 → {pp_score:.3f}")
                final_score = pp_score
        else:
            pp_anchored = 0.5 * stat_mean + 0.5 * pp_score
            if pp_anchored > final_score:
                aggregation.append(
                    f"⑤ 困惑度反向锚定：伪困惑度引擎给出 {pp_score:.2f}"
                    f"（>=0.60，强烈指 AI），取 0.5×{stat_mean:.2f}"
                    f"+0.5×{pp_score:.2f}={pp_anchored:.3f} 覆盖上一步的 "
                    f"{final_score:.3f}")
                final_score = pp_anchored

    # ---- 本次缺失/未参与的引擎（透明度）----
    missing_engines = []
    for _name, _spec in spec_table.items():
        _d, _w, _note = _spec
        if _d == 0 or _w <= 0:
            continue
        if _name not in engine_scores:
            _reason = _current_gaps().get(
                _name, "本次未产出分数（可能因文本不满足该引擎最小长度要求）")
            missing_engines.append(
                {"name": _name, "note": _note, "reason": _reason})
    missing_engines.sort(key=lambda x: -spec_table[x["name"]][1])

    # ---- 判定 ----
    if strict:
        if final_score > 0.30:
            label = "ai"
        elif final_score < 0.15:
            label = "human"
        else:
            label = "mixed"
    else:
        # 阈值在 41 篇真实语料上重标定：
        # 人类上界 0.385 / AI 下界 0.488，中间 mixed 带供人工复核
        if final_score > 0.58:
            label = "ai"
        elif final_score < 0.33:
            label = "human"
        else:
            label = "mixed"

    confident_engines = sorted(
        ({"name": name, "score": round(score, 3)}
         for name, score in engine_scores.items() if score > 0.7),
        key=lambda x: x["score"], reverse=True)[:5]

    # v3.23 P3-L：用 adopted（已按方向归一、已剔除零权重/无效引擎）
    # 而不是 engine_scores 原始读数。否则一个权重 0、只因标定失效而
    # 取值极端的引擎，会让报告在"引擎分歧"面板里点名一个根本没参与
    # 加权计算的信号。
    _adopted_scores = {k: v[0] for k, v in adopted.items()}
    sorted_engines = sorted(_adopted_scores.items(), key=lambda x: x[1])
    conflicting_engines = []
    if len(sorted_engines) >= 2:
        lowest, highest = sorted_engines[0], sorted_engines[-1]
        if highest[1] - lowest[1] > 0.5:
            conflicting_engines = [
                {"lowest": {"name": lowest[0], "score": round(lowest[1], 3)}},
                {"highest": {"name": highest[0], "score": round(highest[1], 3)}},
            ]

    # ---- 高风险句（滑动窗口逐句分数）----
    flagged_sentences = []
    _flag_thresh = 0.35 if strict else 0.5
    for i, s in enumerate(sentences):
        if i < len(window_scores) and window_scores[i] > _flag_thresh:
            _wc = int(_word_count(s))
            _low = s.lower()
            # strict：过滤无害短句，只标长难句/模板句/高级词汇句
            if strict and _wc < 12:
                _has_signal = (
                    any(p.search(_low) for p in _STRICT_ADVANCED_RES)
                    or any(_low.startswith(st)
                           for st in _STRICT_TEMPLATE_STARTERS))
                if not _has_signal:
                    continue

            window_start = max(0, i - 2)
            window_end = min(len(window_scores), i + 3)
            local_window = window_scores[window_start:window_end]
            if len(local_window) > 1:
                local_mean = sum(local_window) / len(local_window)
                local_var = (sum((x - local_mean) ** 2 for x in local_window)
                             / len(local_window))
                uncertainty = local_var ** 0.5
            else:
                uncertainty = 0.0

            _entry = {
                "text": s[:100],
                "score": round(window_scores[i], 3),
                "uncertainty": round(uncertainty, 3),
                "confidence": ("high" if uncertainty < 0.1
                               else "medium" if uncertainty < 0.2 else "low"),
            }
            if strict:
                _strategy = _strict_sentence_strategy(
                    s, prose_text, engine_scores)
                if _strategy:
                    _entry["strategy"] = _strategy
                _rewrite = _mock_rewrite_sentence(s)
                if _rewrite:
                    _entry["rewrite_suggestion"] = _rewrite
            flagged_sentences.append(_entry)

    flagged_sentences.sort(key=lambda x: x["score"], reverse=True)

    # ---- 改写嫌疑 / 翻译腔提示 ----
    rewrite_hint = ""
    _lmk = engine_scores.get("linguistic-markers")
    _fmp = engine_scores.get("formulaic-patterns")
    # 判据基于英文词表，中文下会稳定误报，仅英文语境启用
    if (_lang == "en" and _lmk is not None and _lmk > 0.8
            and _fmp is not None and _fmp < 0.25):
        rewrite_hint = (
            f"词汇层仍有 AI 特征（linguistic-markers {_lmk:.0%}），"
            f"但句式与模板层已被打散（formulaic-patterns {_fmp:.0%}）。"
            "典型形态为「AI 生成后人工改写」或改写工具处理，"
            "此类文本的总分通常会被压低，需单独判断。")

    _tse = engine_scores.get("translationese")
    if _lang == "en" and _tse is not None and _tse >= 0.5:
        _tse_hint = (
            f"检测到明显的机器翻译特征（translationese {_tse:.0%}："
            "长被动句 / of which、in which 等介词前置从句堆叠）。"
            "建议用中文思维重新组织语言，不要直翻——先想清楚这句话中文想说什么，"
            "再用你自己的英文表达出来，而不是逐字对照翻译。")
        rewrite_hint = ((rewrite_hint + " " + _tse_hint).strip()
                        if rewrite_hint else _tse_hint)

    student_verdict = _student_verdict(final_score, label) if strict else ""
    high_pressure_warning = (
        _high_pressure_warning(final_score) if strict else "")

    return AIResult(
        ai_probability=round(final_score, 4),
        label=label,
        method=(f"ensemble v3.3[{mode}] "
                f"({len(adopted)}/{len(engine_scores)} engines adopted, "
                f"ml_models={ml_n}, ml_disagreement={ml_disagreement}, "
                f"confidence={confidence}, references excluded)"),
        engine_scores=engine_scores,
        flagged_sentences=flagged_sentences[:10],
        engine_disagreement=round(wstd, 4),
        confident_engines=confident_engines,
        conflicting_engines=conflicting_engines,
        confidence=confidence,
        adopted_engines=sorted(
            [{"name": k, "score": round(v[0], 3), "weight": round(v[1], 3),
              "note": spec_table.get(k, EngineSpec(0, 0, ""))[2]}
             for k, v in adopted.items()],
            key=lambda x: x["weight"] * abs(x["score"] - final_score),
            reverse=True),
        rejected_engines=[{"name": k, "reason": v}
                          for k, v in rejected.items()],
        rewrite_hint=(rewrite_hint
                      + (" " + ml_note if ml_note else "")).strip(),
        ml_detail=ml_detail,
        ml_disagreement=ml_disagreement,
        ml_note=ml_note,
        stat_mean=round(stat_mean, 4),
        primary_score=(round(primary, 4) if primary is not None else None),
        aggregation=aggregation,
        missing_engines=missing_engines,
        mode=mode,
        student_verdict=student_verdict,
        high_pressure_warning=high_pressure_warning,
    )


def run_ai_detection(text: str, mode: str = "normal") -> AIResult:
    """跑所有可用的 AI 检测引擎，返回综合结果。

    v3.20【P1-9】引擎层缓存由 _engine_stage_cached 提供（cache_info/
    cache_clear 透传）。本函数每次返回新建对象，调用方可原地修改。
    """
    return _run_ai_detection_cached(text, mode=mode)


# 缓存管理接口（引擎层）
run_ai_detection.cache_info = _engine_stage_cached.cache_info
run_ai_detection.cache_clear = _engine_stage_cached.cache_clear


# ============================================================
# 相似度检测（Crossref 联网查重）
# ============================================================
_CROSSREF_USER_AGENT = "AuroraCheck/3.7 (mailto:aurora-check@example.com)"
_CROSSREF_WORKS_URL = "https://api.crossref.org/works"

_EXTERNAL_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "by", "at", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "their", "its", "it", "this", "that", "these", "those",
    "which", "who", "whom", "whose", "about", "into", "over", "under",
    "between", "through", "during", "more", "most", "other", "some",
    "such", "only", "also", "not", "no", "than", "then", "can", "may",
    "will", "would", "should", "could", "but", "if", "when", "where",
    "how", "why", "what", "do", "does", "did", "has", "have", "had",
    "per", "via", "vs", "using", "based", "among", "within", "without",
    "both", "each", "few", "many", "much", "same", "new", "use", "used",
})

_CAPTION_RE = re.compile(r"^(figure|table|fig\.|tab\.)", re.IGNORECASE)


def _title_word_overlap(sentence: str, title: str) -> Tuple[float, int]:
    """返回 (coverage, hit_count)：标题实词在句子里的命中比例与个数。"""
    def _words(s: str) -> set:
        return set(re.findall(r"[a-z]{3,}", s.lower()))

    tw = _words(title) - _EXTERNAL_STOP_WORDS
    if not tw:
        return 0.0, 0
    sw = _words(sentence)
    hits = sum(1 for w in tw if w in sw)
    return hits / len(tw), hits


def _fetch_crossref_match(sentence: str) -> Optional[dict]:
    """对单个句子查 Crossref，返回最匹配条目，失败返回 None。"""
    url = (_CROSSREF_WORKS_URL + "?query.bibliographic="
           + urllib.parse.quote(sentence) + "&rows=1")
    req = urllib.request.Request(
        url, headers={"User-Agent": _CROSSREF_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    items = (data.get("message") or {}).get("items") or []
    if not items:
        return None
    it = items[0]
    title = (it.get("title") or [""])[0]
    return {"title": title, "doi": it.get("DOI", ""),
            "score": it.get("score", 0.0)}


def _check_external_plagiarism_api(text: str, max_queries: int = 5) -> dict:
    """用 Crossref 做外部查重（关键词级近似，非逐字比对）。

    Returns:
        {"overall_similarity": 0~1 或 None, "matched": [...], "note": str}
        None = 「没查成」（断网 / 无足够长句），note 说明原因。
    """
    # 【v3.23 P2-C】中文早退。Crossref 是英文文献元数据库，对中文期刊
    # 覆盖极低；更糟的是下面用 len(s.split()) > 15 挑长句——中文没有
    # 空格，整句只算 1 个"词"，永远过不了 15，于是必然走到"无足够长句
    # 跳过"，白跑一次网络却没给出真正的原因。直接return None + 说清
    # 方法学边界，比伪装成"查过且干净"诚实。
    if _detect_language(text) == "zh":
        return {"overall_similarity": None, "matched": [],
                "note": ("Crossref 是英文文献元数据库，对中文文献覆盖有限，"
                         "已跳过外部查重；中文查重请使用知网 / 万方等工具")}
    sentences = _split_sentences(text)
    long_sents = [s for s in sentences if len(s.split()) > 15][:max_queries]
    if not long_sents:
        return {"overall_similarity": None, "matched": [],
                "note": "无足够长的句子，跳过外部查重"}

    # 并发请求：把 5×5s 串行压到 ~1-2s
    results = []
    with ThreadPoolExecutor(max_workers=min(len(long_sents), 5)) as ex:
        futures = {ex.submit(_fetch_crossref_match, s): s for s in long_sents}
        for fut in futures:
            sent = futures[fut]
            try:
                res = fut.result(timeout=8)
            except Exception:
                res = None
            if res and res.get("title"):
                results.append((sent, res))

    if not results:
        return {"overall_similarity": None, "matched": [],
                "note": "网络不可用，跳过外部查重"}

    matched = []
    for sent, res in results:
        title = (res.get("title") or "").strip()
        # 图表题注不是可引用文献
        if _CAPTION_RE.match(title):
            continue
        cov, hits = _title_word_overlap(sent, title)
        # 双门槛防假阳性：命中 ≥6 且覆盖率 ≥50%
        if hits >= 6 and cov >= 0.5:
            matched.append({
                "sentence": sent[:140],
                "title": title[:160],
                "doi": res.get("doi", ""),
                "coverage": round(cov, 3),
                "hit_words": hits,
                "score": round(res.get("score", 0.0), 1),
            })

    if not matched:
        # v3.20【P0-1】Crossref query.bibliographic 是元数据级模糊
        # 匹配，不是全文索引——「元数据没命中」≠「内容没抄袭」。
        # 此前返回 0.0 会被渲染成「外部相似度 0.0%」，让用户误以为
        # 查过且干净；比不查更危险。现在诚实返回 None + 方法学说明。
        return {"overall_similarity": None, "matched": [],
                "note": ("已联网查询 Crossref（元数据级模糊匹配，非全文"
                         "比对），未发现元数据命中；「未命中」不代表"
                         "「无抄袭」，内容级查重请使用 Turnitin/"
                         "iThenticate 等专业工具")}
    overall = round(min(1.0, len(matched) / len(long_sents)), 4)
    return {"overall_similarity": overall, "matched": matched, "note": ""}


def _para_min_len(text: str) -> int:
    """自重叠比对的段落最小长度阈值（字符数），按语言自适应。

    【v3.23 P2-H】旧版一刀切 >50。50 个字符对英文是"两三个短句"，
    对中文已是一个半完整段落——正常的中文段（30~50 字）被整段滤掉，
    段落数不足 2 时又要走 4 句切块兜底，中文自重叠检测形同虚设。
    中文放宽到 30，英文维持 50（英文段落路径逐字节不变）。

    Args:
        text: 整篇文本，仅用于判语言。

    Returns:
        该语言下的段落最小字符数。
    """
    return 30 if _detect_language(text) == "zh" else 50


def run_plagiarism_check(text: str,
                         enable_network: bool = True) -> PlagiarismResult:
    """相似度检测：内部段落自重叠 +（可选）Crossref 外部比对。

    enable_network=True 时可用环境变量 AURORA_NETWORK=0 全局关闭联网。
    """
    # 1. 内部自重叠（词袋 Jaccard）
    # 【v3.23 P2-H】段落过滤阈值按语言自适应（中文 30 / 英文 50）
    _para_min = _para_min_len(text)
    paragraphs = [p.strip() for p in text.split("\n\n")
                  if len(p.strip()) > _para_min]
    # 兜底：单回车/全连成一段时按 4 句切块
    if len(paragraphs) < 2:
        _buf: List[str] = []
        _chunks: List[str] = []
        for _s in _split_sentences(text):
            _buf.append(_s)
            if len(_buf) >= 4:
                _chunks.append(" ".join(_buf))
                _buf = []
        if _buf:
            _chunks.append(" ".join(_buf))
        paragraphs = [p for p in _chunks if len(p.strip()) > _para_min]

    max_overlap = 0.0
    matched: List[dict] = []
    if len(paragraphs) >= 2:
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                sim = _jaccard_sim(paragraphs[i], paragraphs[j])
                if sim > max_overlap:
                    max_overlap = sim
                    matched = [
                        {"para1": paragraphs[i][:80] + "...",
                         "sim": round(sim, 3)},
                        {"para2": paragraphs[j][:80] + "..."},
                    ]

    # 2. 外部文献比对（可选联网）
    ext_sim: Optional[float] = None
    ext_matches: List[dict] = []
    net_note = ""
    if enable_network and os.environ.get("AURORA_NETWORK", "1") != "0":
        ext = _check_external_plagiarism_api(text)
        ext_sim = ext["overall_similarity"]
        ext_matches = ext["matched"]
        net_note = ext.get("note", "")

    return PlagiarismResult(
        external_similarity=ext_sim,
        self_overlap=round(max_overlap, 4),
        matched_excerpts=matched[:5],
        external_matches=ext_matches,
        network_note=net_note,
    )


def _jaccard_sim(text1: str, text2: str) -> float:
    """Jaccard 相似度（英文词袋 / 中文字符 bigram，效果近似余弦）。

    v3.20【P2-2】由 _cosine_sim 更名：它算的是 Jaccard 而非余弦，
    旧名误导读者。分词改走 _text_ngrams，自动获得中文支持。
    """
    ng1 = _text_ngrams(text1)
    ng2 = _text_ngrams(text2)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


# ============================================================
# 引用验证
# ============================================================
def _doi_to_bibtex(doi: str) -> str:
    """用 doi.org 内容协商取 BibTeX，不依赖外部命令。

    原实现只调 onecite CLI，未安装时把所有有效 DOI 记成"解析失败"，
    把"工具没装"伪装成"文献有问题"。
    """
    url = "https://doi.org/" + urllib.parse.quote(doi)
    req = urllib.request.Request(
        url, headers={"Accept": "application/x-bibtex"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", "replace")
        return body.strip() if "@" in body else ""
    except Exception:
        return ""


def _resolve_one_doi(doi: str, has_onecite: bool) -> str:
    """单个 DOI → BibTeX：优先 onecite CLI，缺失时回落 doi.org。"""
    if has_onecite:
        try:
            result = subprocess.run(
                ["onecite", "process", doi, "--quiet"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return _doi_to_bibtex(doi)


def run_citation_check(text: str) -> CitationResult:
    """提取引用并逐一验证：优先 onecite，缺失时回落到 doi.org。

    v3.19【P2-1】并行解析（网络 IO 密集，5 并发）：20 个 DOI 串行
    最坏 20×(15s+12s)≈9 分钟，压到约 1/5 墙钟；解析顺序与输出
    格式保持不变。
    """
    import shutil

    dois = _RE_DOIExtract.findall(text)
    dois = list(dict.fromkeys(dois))  # 去重保序
    if not dois:
        return CitationResult(total_extracted=0, resolved=0)

    has_onecite = shutil.which("onecite") is not None
    selected = dois[:20]
    with ThreadPoolExecutor(max_workers=5) as ex:
        entries = list(ex.map(lambda d: _resolve_one_doi(d, has_onecite),
                              selected))

    resolved = 0
    failed = []
    bibtex_entries = []
    for doi, entry in zip(selected, entries):
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
def run_check(text: str = "", file_path: str = "",
              mode: str = "normal") -> CheckReport:
    """Run the full detection pipeline in one call.

    Args:
        text: The text to analyse directly (mutually exclusive with file_path).
        file_path: Read the text from a file (.txt/.md/.docx) instead.
        mode: "normal" (default, balanced) or "strict"
            (paranoid pre-submission screening that deliberately over-flags).

    Returns:
        CheckReport: call .to_markdown() or .to_html() to render.

    Raises:
        ExtractionError: file_path 给定但文件读取/解析失败。
    """
    if file_path:
        text = extract_text_from_file(file_path)

    ok, reason = validate_text(text)
    if not ok:
        return CheckReport(text_length=len((text or "").strip()), note=reason)

    ai_result = run_ai_detection(text, mode=mode)
    plagiarism_result = run_plagiarism_check(text)
    citation_result = run_citation_check(text)

    return CheckReport(
        text_length=len(text),
        ai=ai_result,
        plagiarism=plagiarism_result,
        citations=citation_result,
    )


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    _flags = {a for a in sys.argv[1:] if a.startswith("--")}
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.argv = [sys.argv[0]] + _args

    if len(sys.argv) < 2:
        print("用法:")
        print("  AI 检测:  python aurora_check.py <文件路径> [--strict] [--json] [--no-html]")
        print("  文档比对: python aurora_check.py compare <当前文档> <参考文档>")
        print("  环境变量: AURORA_VERBOSE=1 显示模型加载细节")
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
            print("高重叠句子:")
            for i, s in enumerate(result["overlapping_sentences"][:10]):
                print(f"  {i + 1}. 相似度 {s['similarity']:.1%}")
                print(f"     当前文档: {s['doc_sentence'][:80]}...")
                print(f"     参考文档: {s['ref_sentence'][:80]}...")
            print()
        sys.exit(0)

    # 普通 AI 检测模式（--strict 走严苛模式，与 Web 端"严苛检测"同源）
    report = run_check(file_path=sys.argv[1],
                       mode="strict" if "--strict" in _flags else "normal")

    if "--json" in _flags:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        sys.exit(0)

    print(report.to_markdown())

    if "--no-html" not in _flags:
        base_name = os.path.splitext(sys.argv[1])[0]
        html_path = base_name + "_aurora_report.html"
        report.to_html(output_path=html_path)
        print(f"\nHTML 报告已生成: {html_path}")
