# -*- coding: utf-8 -*-
"""
Aurora Check 统一入口

用法：
    python check.py paper 论文.docx         完整核查（提取 DOI + 验真 + 期刊 + AI 痕迹）
    python check.py doi 10.3389/frai.2025.1629320
    python check.py text 文稿.txt          只看 AI 痕迹
    python check.py batch 目录/            批量核查所有文件

输出：人读的 Markdown（默认）或 --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aurora_check as ac
import forensic as fr
import verify as vf
import teacher as tc

DOI_IN_TEXT = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")


def extract_text(path: str) -> str:
    if path.lower().endswith(".docx"):
        return ac.extract_text_from_file(path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def find_dois(text: str) -> list:
    """从正文里找 DOI，去重并去掉尾部标点。"""
    seen, out = set(), []
    for m in DOI_IN_TEXT.findall(text):
        d = m.rstrip(".,;)】」")
        if d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out


def _tier_cn(t: str) -> str:
    return {"未见异常": "未见异常",
            "需注意": "⚠️ 需注意",
            "审稿强度存疑": "⚠️ 审稿强度存疑"}.get(t, "未知")


def render_paper(path: str, do_check_ai: bool = True, do_forensic: bool = True) -> dict:
    text = extract_text(path)
    name = os.path.basename(path)

    result = {
        "file": name,
        "chars": len(text),
        "dois": [],
        "ai": None,
        "forensic": None,
        "warnings": [],
    }

    # ---- 1. DOI ----
    dois = find_dois(text)
    if not dois:
        result["warnings"].append("正文中未找到 DOI。若这是学术论文，缺少 DOI 本身就是疑点。")

    for d in dois[:8]:                      # 一篇论文通常不会超过 8 个自有 DOI
        r = vf.verify_paper(d)
        result["dois"].append(r)

    # ---- 2. 期刊一致性：正文声称的期刊名 vs DOI 指向的期刊 ----
    for r in result["dois"]:
        jn = r["journal"].get("name") or ""
        if jn and jn.lower() not in text.lower():
            # 期刊名可能在页眉被省略，只作提示不作结论
            result["warnings"].append(
                f"DOI 指向《{jn}》，但正文中未出现该刊名——"
                "若论文标注了期刊，请核对是否一致"
            )

    # ---- 3. AI 痕迹 ----
    if do_check_ai:
        rep = ac.run_check(text=text)
        if rep.ai:
            result["ai"] = {
                "score": rep.ai.ai_probability,
                "label": rep.ai.label,
                "confidence": rep.ai.confidence,
                "rewrite_hint": rep.ai.rewrite_hint,
                "adopted": len(rep.ai.adopted_engines),
            }
        else:
            result["ai"] = {"error": rep.note}

    # ---- 4. 文档取证：元数据 / 引用一致性 / 结构同构 ----
    if do_forensic and path.lower().endswith(".docx"):
        try:
            import docx as _dx
            paras = [p.text for p in _dx.Document(path).paragraphs]
            result["forensic"] = fr.run_forensic(path, text, paras)
        except Exception as e:                      # 取证失败不能拖垮主流程
            result["warnings"].append(f"取证未完成：{e}")

    return result


def render_markdown(rows: list) -> str:
    L = []
    for r in rows:
        L.append(f"\n## {r['file']}")
        # doi 模式没有正文，显示"0 字符"是噪声
        if r.get("chars"):
            L.append(f"`{r['chars']} 字符`\n")

        # DOI
        if r["dois"]:
            L.append("### DOI 核验")
            for d in r["dois"]:
                doi, j = d["doi"], d["journal"]
                # 「格式不合法」与「查过但不存在」是两件事，不能共用"未注册"，
                # 否则用户会以为系统真的去 doi.org 查过。
                if doi.get("error") and "格式" in doi["error"]:
                    mark = "❌ 格式不合法，不是有效 DOI"
                else:
                    mark = {True: "✅ 已注册", False: "❌ 未注册（该编号不存在）",
                            None: "⚠️ 无法确认（网络问题）"}[doi["registered"]]
                L.append(f"- **{doi['normalized']}** — {mark}")
                if doi["title"]:
                    L.append(f"  - 标题：{doi['title'][:90]}")
                if doi["journal"]:
                    cite = f"  - 期刊：{doi['journal']}　{doi['year']}"
                    if doi["volume"]:
                        cite += f"　{doi['volume']}"
                        if doi["issue"]:
                            cite += f"({doi['issue']})"
                    if doi["pages"]:
                        cite += f"　{doi['pages']}"
                    L.append(cite)
                if doi["authors"]:
                    a = doi["authors"]
                    L.append(f"  - 作者：{', '.join(a[:3])}{' 等' if len(a) > 3 else ''}")
                # 撤稿状态排在最前：这是唯一能让"一切指标正常"的论文作废的信号
                if doi.get("retracted") is True:
                    L.append("  - ⛔ **已撤稿 — 不应引用**")
                _pct = doi.get("cited_percentile")
                if _pct and _pct > 0:
                    L.append(f"  - 被引百分位：同年同领域前 {max(1, 100 - _pct + 1)}% 以内"
                             f"（{doi.get('topic') or '领域未知'}）")
                elif _pct == -1:
                    L.append("  - 被引百分位：发表不足 2 年，暂不评价")
                if j.get("name"):
                    L.append(f"  - 期刊档案：**{_tier_cn(j['tier'])}**")
                    if j.get("tier_reason"):
                        L.append(f"    - 依据：{j['tier_reason']}")
                    if j.get("yearly"):
                        curve = " → ".join(f"{y}:{c}" for y, c in reversed(j["yearly"][:5]))
                        L.append(f"    - 年发文：{curve}")
                    bits = []
                    if j.get("h_index") is not None:
                        bits.append(f"h-index {j['h_index']}")
                    if j.get("mean_cited_2y") is not None:
                        bits.append(f"2 年均被引 {j['mean_cited_2y']:.1f}")
                    if bits:
                        L.append(f"    - {' · '.join(bits)}")
                    for f in (j.get("risk_flags") or []):
                        # 依据行已经写过同样内容，不重复
                        if j.get("tier_reason") and f in j["tier_reason"]:
                            continue
                        L.append(f"    - {f}")
        else:
            L.append("### DOI 核验\n- 未找到 DOI")

        # AI
        if r["ai"] and "score" in r["ai"]:
            a = r["ai"]
            L.append("\n### AI 痕迹")
            L.append(f"- 特征强度 **{a['score']:.0%}**（{a['label']}，置信度 {a['confidence']}）")
            L.append(f"- 采纳引擎 {a['adopted']} 个")
            if a.get("rewrite_hint"):
                L.append(f"- ⚠️ 疑似改写：{a['rewrite_hint'][:80]}")
            if a["label"] == "ai":
                ppv = ac._positive_predictive_value()
                L.append(f"- 若据此认定，命中率约 {ppv:.0%}"
                         f"（假设场景中 AI 稿占 10%）")
        elif r["ai"]:
            L.append(f"\n### AI 痕迹\n- 无法判定：{r['ai'].get('error')}")

        fo = r.get("forensic")
        if fo:
            meta = fo["metadata"]
            L.append("\n### 文档取证")
            for x in meta["bad"]:
                L.append(f"- ❌ {x}")
            for x in meta["warn"]:
                L.append(f"- ⚠️ {x}")
            for x in meta["ok"]:
                L.append(f"- ✅ {x}")

            ci = fo["citations"]
            L.append("\n### 引用一致性")
            L.append(f"- 文献表 {ci['entries']} 条，正文标注 {ci['in_text_count']} 条")
            for k, label in (("missing_in_text", "文献表有但正文未引"),
                             ("missing_entry", "正文引了但文献表没有"),
                             ("block_missing", "注释段内未标年份"),
                             ("year_mismatch", "年份不一致")):
                if ci.get(k):
                    L.append(f"- ⚠️ {label}：" + "、".join(ci[k][:4]))

            st = fo["structure"]
            if st["lengths"]:
                L.append("\n### 结构")
                L.append(f"- 注释段词数：{' / '.join(str(x) for x in st['lengths'])}"
                         f"（变异系数 {st['cv']:.2f}）")
                L.append(f"- 段首六个词最高重合：{st['similarity']:.0%}")

            if fo["hints"]:
                L.append("\n### 评阅人可能会问")
                for h in fo["hints"]:
                    L.append(f"- ⚠️ {h}")

        if r["warnings"]:
            L.append("\n### 提示")
            for w in r["warnings"]:
                L.append(f"- {w}")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["paper", "doi", "text", "batch"])
    ap.add_argument("target")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-ai", action="store_true", help="只核 DOI，不跑 AI 检测")
    ap.add_argument("--no-forensic", action="store_true",
                    help="跳过文档取证（元数据 / 引用一致性 / 结构同构）")
    ap.add_argument("--teacher", action="store_true",
                    help="在报告末尾追加 rubric 评分表（教师人工复核要点），"
                         "供 academic-english-teacher 人设做最终判断")
    a = ap.parse_args()

    if a.mode == "doi":
        r = vf.verify_paper(a.target)
        print(json.dumps(r, ensure_ascii=False, indent=2) if a.json
              else render_markdown([{"file": a.target, "chars": 0,
                                     "dois": [r], "ai": None, "warnings": []}]))
        return

    if a.mode == "text":
        rep = ac.run_check(text=extract_text(a.target))
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2) if a.json
              else rep.to_markdown())
        return

    files = ([a.target] if a.mode == "paper"
             else sorted(glob.glob(os.path.join(a.target, "*.*"))))
    files = [f for f in files if f.lower().endswith((".txt", ".md", ".docx"))]
    if not files:
        print("没有找到可处理的文件")
        sys.exit(1)

    rows = [render_paper(f, do_check_ai=not a.no_ai, do_forensic=not a.no_forensic)
            for f in files]
    md = render_markdown(rows)
    if a.teacher and not a.json:
        md += tc.render_teacher_section(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2) if a.json
          else "# Aurora Check · 论文核查\n" + md)


if __name__ == "__main__":
    main()
