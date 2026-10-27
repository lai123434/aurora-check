# -*- coding: utf-8 -*-
"""文档取证：统计层之外的三类信号。

统计引擎（困惑度、突发度、词频）只看文字本身。但评阅人先看的是另外三样：
文件是谁、什么时候、改了多久；引用和文献表对不对得上；段落是不是一个模子。

这里把这三样做成机器可查的检查项：
  1. 元数据 —— docx 的 author / created / TotalTime / Pages / revision
  2. 引用一致性 —— 正文 in-text citation 与参考文献表是否一一对应
  3. 结构同构 —— 段首句式的重复度、注释段长度的均匀度

它们测不出"论证断裂"这种真需要人读的东西，但能把九成的机械痕迹挡在门外。
"""
from __future__ import annotations

import re
import statistics
import zipfile

import docx

AUTH = r"[A-Z][A-Za-z'\-]+"
# 文献条目：Surname, A. / Surname Second, A. / Surname, A., & Other, B.
ENTRY_RE = re.compile(rf"^({AUTH}(?:\s+{AUTH})?),\s+[A-Z]\.")
YEAR_PAREN = re.compile(r"\((\d{4})[a-z]?\)")
# 正文引用：Lin, Gu and Yang (2021) / Yu and Lu (2022) / Wut et al. (2025)
CITE_RUN = re.compile(
    rf"({AUTH}(?:\s*(?:,|and|&)\s*{AUTH})*(?:\s+et\s+al\.?)?)\s*\((\d{{4}})[a-z]?\)")
# 括号式：(Lin et al., 2021) / (Yu & Lu, 2022)
CITE_PAREN = re.compile(
    rf"\(({AUTH}(?:\s*(?:,|&|and)\s*{AUTH})*(?:\s+et\s+al\.?)?),\s*(\d{{4}})[a-z]?\)")


# ---------------------------------------------------------------- 元数据

def _app_props(path: str) -> dict:
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("docProps/app.xml").decode("utf-8", "replace")
    except Exception:
        return out
    for tag in ("TotalTime", "Pages", "Words", "Characters", "Paragraphs", "Application"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S)
        if m:
            out[tag] = m.group(1).strip()
    return out


def check_metadata(path: str) -> dict:
    d = docx.Document(path)
    cp = d.core_properties
    app = _app_props(path)
    body_words = sum(len(p.text.split()) for p in d.paragraphs if p.text.strip())

    def _int(v):
        try:
            return int(str(v).strip())
        except Exception:
            return 0

    total_time, pages = _int(app.get("TotalTime")), _int(app.get("Pages"))
    words, rev = _int(app.get("Words")), cp.revision or 0
    ok, warn, bad = [], [], []

    author = (cp.author or "").strip()
    if author.lower() in ("python-docx", "python_docx"):
        bad.append(f"作者栏是 {author} —— 这份文件由脚本生成，不是文字处理软件保存的")
    elif author:
        ok.append(f"作者：{author}")
    else:
        ok.append("作者栏为空（WPS / 未登录账户常见，正常）")

    created, modified = cp.created, cp.modified
    if created and created.year < 2020:
        bad.append(f"创建时间是 {created:%Y-%m-%d} —— 这是生成库自带模板的时间戳")
    elif created and modified:
        hours = (modified - created).total_seconds() / 3600
        if hours < 0.5:
            bad.append(f"创建到修改只隔 {hours * 60:.0f} 分钟 —— 一次性生成的特征")
        elif hours < 24:
            warn.append(f"创建到修改不足 1 天（{hours:.1f} 小时）")
        else:
            ok.append(f"创建 {created:%m-%d} → 修改 {modified:%m-%d}，跨 {hours / 24:.0f} 天")
    else:
        warn.append("缺少创建或修改时间")

    if total_time == 0:
        bad.append("编辑时长为 0 分钟 —— 文档从未被真正打开编辑过")
    elif total_time < 15 and body_words > 300:
        warn.append(f"编辑时长仅 {total_time} 分钟，正文 {body_words} 词，偏短")
    else:
        ok.append(f"编辑 {total_time} 分钟，正文 {body_words} 词（{body_words / total_time:.1f} 词/分钟）")

    if pages and body_words > 500 and pages <= 1:
        bad.append(f"属性记 {pages} 页，实际正文 {body_words} 词 —— 页数与内容不符")
    if words and body_words and abs(words - body_words) > max(120, body_words * 0.25):
        warn.append(f"属性记 {words} 词，正文实测 {body_words} 词")

    if rev == 0:
        warn.append("修订次数为 0 —— 通常是导出而非逐次编辑")
    elif rev >= 3:
        ok.append(f"修订 {rev} 次")

    appname = (app.get("Application") or "").strip()
    if appname and "Macintosh" in appname:
        warn.append(f"生成程序标记为 {appname}（Mac 版 Word），与常见提交环境不符")

    return {"ok": ok, "warn": warn, "bad": bad,
            "raw": {"author": author, "created": str(created), "modified": str(modified),
                    "revision": rev, "total_time": total_time, "pages": pages, "words": words}}


# ---------------------------------------------------------------- 结构拆分

def _is_entry(p: str) -> bool:
    return bool(ENTRY_RE.match(p.strip())) and bool(YEAR_PAREN.search(p))


def split_blocks(paragraphs: list) -> tuple:
    """把文档切成「文献条目 + 紧随其后的注释段」，返回 (blocks, loose)。

    annotated bibliography 里结论段不属于任何一条文献，单独归到 loose，
    否则会被误判成"这一篇没读懂"。
    """
    paras = [p.strip() for p in paragraphs]
    idx = [i for i, p in enumerate(paras) if p and _is_entry(p)]
    blocks, used = [], set()
    for k, i in enumerate(idx):
        end = idx[k + 1] if k + 1 < len(idx) else len(paras)
        ann = ""
        for j in range(i + 1, end):
            if len(paras[j].split()) >= 25:
                ann = paras[j]
                used.add(j)
                break
        blocks.append((paras[i], ann))
        used.add(i)
    loose = [p for j, p in enumerate(paras) if p and j not in used and len(p.split()) >= 40]
    return blocks, loose


# ---------------------------------------------------------------- 引用一致性

def check_citations(blocks: list, loose: list) -> dict:
    entries = []
    for entry, _ in blocks:
        surname = ENTRY_RE.match(entry).group(1)
        year = YEAR_PAREN.search(entry).group(1)
        entries.append((surname, year))

    prose_all = " ".join([ann for _, ann in blocks] + loose)

    def _has_cite(text: str, surname: str, year: str) -> bool:
        """正文是否以 APA 形式引过这一条。

        两种情况都算：叙述式 "Surname et al. (2020)"，括号式 "(Surname, 2020)"。
        括号里可能并列多条，所以要在括号内容里找姓氏与年份，而不是要求
        括号紧跟姓氏——"（A et al., 2020; B & C, 2020）" 这种写法很常见。
        """
        if re.search(re.escape(surname) + r"[^()]{0,90}?\(" + year, text):
            return True
        for m in re.finditer(r"\(([^()]{0,260})\)", text):
            inner = m.group(1)
            if year in inner and surname in inner:
                return True
        return False

    missing_in_text = [f"{s} ({y})" for s, y in entries if not _has_cite(prose_all, s, y)]

    # 注释段内部没标注年份的——只靠文末列表或结论段并列，Referencing 会打折
    block_missing = []
    for i, ((_, ann), (s, y)) in enumerate(zip(blocks, entries), 1):
        if ann and not _has_cite(ann, s, y):
            block_missing.append(f"第 {i} 段（{s} {y}）")

    # 正文出现、文献表没有的作者
    found = set()
    for m in re.finditer(rf"({AUTH}(?:\s+{AUTH})?(?:\s*(?:,|and|&)\s*"
                         rf"(?:some\s+other\s+authors|{AUTH}))*(?:\s+et\s+al\.?)?)"
                         rf"\s*\((\d{{4}})[a-z]?\)", prose_all):
        first = re.split(r"[\s,&]+", m.group(1).strip())[0]
        found.add((first, m.group(2)))
    for m in CITE_PAREN.finditer(prose_all):
        first = re.split(r"[\s,&]+", m.group(1).strip())[0]
        found.add((first, m.group(2)))
    # 双词姓会被截成首词（Portocarrero Ramos → Portocarrero），
    # 只要首词能对上某条文献且年份一致，就不算"文献表没有"
    def _known(s: str, y: str) -> bool:
        return any(s == es.split()[0] and y == ey for es, ey in entries)

    missing_entry = [f"{s} ({y})" for s, y in sorted(found - set(entries)) if not _known(s, y)]

    years = {}
    for s, y in found:
        years.setdefault(s, set()).add(y)
    mismatch = [f"{s}：文献表 {y}，正文 {', '.join(sorted(years.get(s) or []))}"
                for s, y in entries if years.get(s) and y not in years[s]]

    return {
        "entries": len(entries),
        "in_text_count": len(entries) - len(missing_in_text),
        "loose_author_phrase": bool(re.search(
            r"and\s+some\s+other\s+authors|and\s+other\s+authors", prose_all, re.I)),
        "missing_in_text": missing_in_text,
        "missing_entry": missing_entry,
        "block_missing": block_missing,
        "year_mismatch": mismatch,
    }


# ---------------------------------------------------------------- 结构同构

def check_structure(blocks: list) -> dict:
    anns = [a for _, a in blocks if a]
    if len(anns) < 2:
        return {"similarity": 0.0, "pairs": [], "lengths": [], "cv": 0.0, "uniform": False}

    heads = [" ".join(p.split()[:6]).lower().strip(".,;:") for p in anns]
    worst, pair = 0.0, None
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            a, b = set(heads[i].split()), set(heads[j].split())
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac > worst:
                worst, pair = jac, (i + 1, j + 1, heads[i], heads[j])

    lens = [len(p.split()) for p in anns]
    mean = sum(lens) / len(lens)
    cv = statistics.pstdev(lens) / mean if mean else 0.0
    return {"similarity": round(worst, 3), "pairs": [pair] if pair else [],
            "lengths": lens, "cv": round(cv, 3), "uniform": cv < 0.15}


# ---------------------------------------------------------------- 人工审查提示

POINT_RE = re.compile(r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?|finally)\b", re.I)


def review_hints(blocks: list, citations: dict, structure: dict) -> list:
    hints = []
    for i, (_, ann) in enumerate(blocks, 1):
        if not ann:
            hints.append(f"第 {i} 条文献后面没有注释段")
            continue
        n = len(set(m.group(1).lower() for m in POINT_RE.finditer(ann)))
        if n and n < 3:
            hints.append(f"第 {i} 段只标出 {n} 个要点——若 rubric 要求每篇 3 点，这里会扣分")
        if len(ann.split()) < 120:
            hints.append(f"第 {i} 段仅 {len(ann.split())} 词，明显短于常规注释段——"
                         "若不是刻意为之，说明这一篇没读懂")

    if structure.get("uniform"):
        hints.append(f"注释段长度过于均匀（变异系数 {structure['cv']:.2f}）——评阅人常看作模板痕迹")
    if structure.get("similarity", 0) >= 0.5 and structure.get("pairs"):
        a, b, ha, hb = structure["pairs"][0]
        hints.append(f"第 {a} 段与第 {b} 段开头六个词重合 {structure['similarity']:.0%}"
                     f"（“{ha}” / “{hb}”）")

    if citations.get("block_missing"):
        hints.append("这些注释段正文没标年份（只靠文末列表或结论并列）："
                     + "、".join(citations["block_missing"][:3]))
    if citations.get("missing_in_text"):
        hints.append("文献表里这些条目正文从未提及：" + "、".join(citations["missing_in_text"][:4]))
    if citations.get("missing_entry"):
        hints.append("正文引用了但文献表没有：" + "、".join(citations["missing_entry"][:4]))
    if citations.get("year_mismatch"):
        hints.append("年份不一致：" + "；".join(citations["year_mismatch"][:2]))
    if citations.get("in_text_count") == 0 and citations.get("entries"):
        hints.append("全文没有 in-text citation，只有文末列表——Referencing 一项会扣分")
    if citations.get("loose_author_phrase"):
        hints.append("正文用 “and some other authors” 代替 et al.——APA 7 要求三位以上作者用 et al.")

    return hints


def run_forensic(path: str, text: str, paragraphs: list) -> dict:
    blocks, loose = split_blocks(paragraphs)
    cites = check_citations(blocks, loose)
    struct = check_structure(blocks)
    return {
        "metadata": check_metadata(path),
        "citations": cites,
        "structure": struct,
        "blocks": len(blocks),
        "hints": review_hints(blocks, cites, struct),
    }
