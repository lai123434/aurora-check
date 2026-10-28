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
import persona as ps

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


def render_paper(path: str, do_check_ai: bool = True, do_forensic: bool = True,
                 mode: str = "normal") -> dict:
    text = extract_text(path)
    name = os.path.basename(path)

    result = {
        "file": name,
        "chars": len(text),
        "dois": [],
        "ai": None,
        "plagiarism": None,
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
        rep = ac.run_check(text=text, mode=mode)
        if rep.ai:
            # 「为什么是这个分」的证据链：不能只给一个数字。
            # 取贡献最大的若干引擎及其人类可读释义（ENGINE_SPEC 的 note），
            # 让用户能复核「这条凭什么说像 AI」。
            final = rep.ai.ai_probability
            top = []
            for e in rep.ai.adopted_engines:
                name = e["name"]
                spec = getattr(ac, "ENGINE_SPEC", {}).get(name)
                note = spec[2] if spec else ""
                top.append({
                    "name": name,
                    "score": e["score"],
                    "weight": e["weight"],
                    "lift": e["weight"] * (e["score"] - final),
                    "note": e.get("note") or (spec[2] if spec else ""),
                })
            top.sort(key=lambda x: abs(x["lift"]), reverse=True)
            result["ai"] = {
                "score": rep.ai.ai_probability,
                "label": rep.ai.label,
                "confidence": rep.ai.confidence,
                "mode": rep.ai.mode,
                "student_verdict": rep.ai.student_verdict,
                "high_pressure_warning": rep.ai.high_pressure_warning,
                "rewrite_hint": rep.ai.rewrite_hint,
                "adopted": len(rep.ai.adopted_engines),
                "method": rep.ai.method,
                "top": top,
                "ml_detail": rep.ai.ml_detail,
                "ml_note": rep.ai.ml_note,
                "rejected": rep.ai.rejected_engines,
                "conflicting": rep.ai.conflicting_engines,
                "flagged": rep.ai.flagged_sentences,
                # 可解释性：终分的推导步骤 + 主导信号
                "aggregation": rep.ai.aggregation,
                "stat_mean": rep.ai.stat_mean,
                "primary_score": rep.ai.primary_score,
                # 透明度：本次未参与检测的引擎及原因
                "missing": rep.ai.missing_engines,
            }
        else:
            result["ai"] = {"error": rep.note}

        # ---- 3.5 相似度：内部自重叠 + Crossref 外部查重 ----
        if rep.plagiarism:
            p = rep.plagiarism
            result["plagiarism"] = {
                "external_similarity": p.external_similarity,
                "self_overlap": p.self_overlap,
                "external_matches": p.external_matches,
                "network_note": p.network_note,
            }

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

        # 相似度（内部自重叠 + Crossref 外部查重）
        pl = r.get("plagiarism")
        if pl:
            L.append("\n### 相似度")
            if pl.get("external_similarity") is not None:
                L.append(f"- 外部文献相似度 **{pl['external_similarity']:.1%}**"
                         f"（通过 Crossref 学术库比对）")
                for m in (pl.get("external_matches") or []):
                    L.append(f"  - ⚠️ 疑似句：\"{m['sentence'][:70]}...\"")
                    L.append(f"    ↳ 可能出自《{m['title']}》 ({m['doi']})")
                if not pl.get("external_matches"):
                    L.append("  - 未发现明显匹配的文献")
            else:
                if pl.get("network_note"):
                    L.append(f"- 外部文献相似度：未完成 — {pl['network_note']}")
                else:
                    L.append("- 外部文献相似度：未启用")
            L.append(f"- 文档内部段落自重叠：{pl.get('self_overlap', 0):.2%}")

        # AI
        if r["ai"] and "score" in r["ai"]:
            a = r["ai"]
            L.append("\n### AI 痕迹")
            # —— 自检找茬（strict）模式：报告顶部先吼高压预警 ——
            if a.get("high_pressure_warning"):
                L.append("")
                L.append(f"> {a['high_pressure_warning']}")
            _mode_tag = "　🧨 严苛模式" if a.get("mode") == "strict" else ""
            L.append(f"- 特征强度 **{a['score']:.0%}**（{a['label']}，置信度 {a['confidence']}，"
                     f"采纳引擎 {a['adopted']} 个）{_mode_tag}")
            if a.get("student_verdict"):
                L.append(f"- **给你（学生）的解读：** {a['student_verdict']}")

            # —— 终分是怎么来的（可解释性：先给算式，再给证据）——
            if a.get("aggregation"):
                L.append("\n**这个分数是怎么算出来的：**")
                L.append("")
                for step in a["aggregation"]:
                    L.append(f"- {step}")
                if a.get("mode") == "strict":
                    L.append("- 判定带（严苛）：`ai` >0.30 · `mixed` 0.15–0.30 · `human` <0.15")
                else:
                    L.append("- 判定带：`ai` >0.58 · `mixed` 0.33–0.58 · `human` <0.33")

            # —— 为什么是这个分：列出把分数推向「AI」的主要证据 ——
            pushers = [t for t in a.get("top", []) if t["lift"] > 0.004]
            pullers = [t for t in a.get("top", []) if t["lift"] < -0.004]
            if pushers:
                L.append("\n**为什么会被判「有 AI 痕迹」——主要证据：**")
                L.append("")
                L.append("| 引擎 | 它测什么 | 得分 | 权重 | 把总分推向AI |")
                L.append("|---|---|---|---|---|")
                for t in pushers[:8]:
                    L.append(f"| `{t['name']}` | {t['note'] or '—'} | {t['score']:.2f} | "
                             f"{t['weight']:.3f} | +{t['lift'] * 100:.1f} 点 |")
            if pullers:
                L.append("\n**反方向（把分数压向「像人」）的证据：**")
                L.append("")
                L.append("| 引擎 | 它测什么 | 得分 | 权重 | 把总分压向人 |")
                L.append("|---|---|---|---|---|")
                for t in pullers[:5]:
                    L.append(f"| `{t['name']}` | {t['note'] or '—'} | {t['score']:.2f} | "
                             f"{t['weight']:.3f} | {t['lift'] * 100:.1f} 点 |")

            mld = a.get("ml_detail") or {}
            if mld.get("scores"):
                L.append("\n**深度学习检测器分项（交叉验证）：**")
                for name, v in mld["scores"].items():
                    L.append(f"- `{name.split('/')[-1]}`：**{v:.0%}**")
                if a.get("ml_note"):
                    L.append(f"- ⚠️ {a['ml_note']}")
                if mld.get("pseudo_perplexity") is not None:
                    L.append(f"- 伪困惑度原始值：**{mld['pseudo_perplexity']}**"
                             f"（越低越可预测＝越像 AI；本域实测 AI 约 150–450，真人约 9000+）")

            if a.get("flagged"):
                L.append("\n**被标为可疑的句子/段落：**")
                for s in a["flagged"][:6]:
                    if isinstance(s, dict):
                        conf = {"high": "🟢", "medium": "🟡"}.get(
                            s.get("confidence"), "🔴")
                        L.append(f"- {conf} ({s.get('score', 0):.2f}) "
                                 f"{s.get('text', '')[:90]}")
                        if a.get("mode") == "strict" and s.get("strategy"):
                            L.append(f"  ↳ 怎么改：{s['strategy']}")

            if a.get("rejected"):
                L.append("\n<details><summary>未参与投票的引擎（及其原因）</summary>\n")
                for x in a["rejected"][:12]:
                    L.append(f"- `{x['name']}`：{x['reason']}")
                L.append("\n</details>")

            # —— 透明度：本次没跑起来的引擎，必须明确告诉用户（而非静默跳过）——
            if a.get("missing"):
                L.append("\n**本次未参与检测的引擎（缺失或不可用）：**")
                for m in a["missing"]:
                    L.append(f"- `{m['name']}`（{m['note']}）→ {m['reason']}")

            if a.get("rewrite_hint"):
                L.append(f"\n- ⚠️ 疑似改写：{a['rewrite_hint'][:120]}")
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


def render_friend_sections(rows: list) -> str:
    """渲染『第三人格：同学群像』段，供 --friend 时追加到报告末尾。

    委托 persona.render_friend_sections：按 FRIEND_ROLES 顺序取
    result["friend_reviews"] 里的点评渲染；缺某角色则输出其已填充提示词。
    """
    parts = ["\n---\n"]
    for r in rows:
        parts.append(ps.render_friend_sections(r))
    return "\n".join(parts)


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
    ap.add_argument("--friend", action="store_true",
                    help="在报告末尾追加『第三人格：同学/好朋友』点评。"
                         "若结果已注入 friend 字段（score + review）则渲染正文，"
                         "否则输出已填充好的提示词，交给 LLM/agent 回填")
    ap.add_argument("--strict", action="store_true",
                    help="自检找茬（严苛）模式：取消宽容逻辑，均值>30% 直接标红，"
                         "并附逐句修改策略与高压预警。定位是学生的『排雷工具』。")
    a = ap.parse_args()

    if a.mode == "doi":
        r = vf.verify_paper(a.target)
        print(json.dumps(r, ensure_ascii=False, indent=2) if a.json
              else render_markdown([{"file": a.target, "chars": 0,
                                     "dois": [r], "ai": None, "warnings": []}]))
        return

    if a.mode == "text":
        rep = ac.run_check(text=extract_text(a.target),
                           mode=("strict" if a.strict else "normal"))
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2) if a.json
              else rep.to_markdown())
        return

    files = ([a.target] if a.mode == "paper"
             else sorted(glob.glob(os.path.join(a.target, "*.*"))))
    files = [f for f in files if f.lower().endswith((".txt", ".md", ".docx"))]
    if not files:
        print("没有找到可处理的文件")
        sys.exit(1)

    rows = [render_paper(f, do_check_ai=not a.no_ai, do_forensic=not a.no_forensic,
                         mode=("strict" if a.strict else "normal"))
            for f in files]
    md = render_markdown(rows)
    if a.teacher and not a.json:
        md += tc.render_teacher_section(rows)
    if a.friend and not a.json:
        md += render_friend_sections(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2) if a.json
          else "# Aurora Check · 论文核查\n" + md)


if __name__ == "__main__":
    main()
