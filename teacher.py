# teacher.py — 引擎的「人工判断」层桥接模块
#
# 作用：把 aurora_check（AI 概率）、forensic（元数据/引用/结构）、verify（DOI/期刊）
# 三处的*机械证据*，预填进 BUS100 A1 的 rubric 评分表，生成「教师复核要点」，
# 交给 academic-english-teacher 人设（Skill）做最终人工判断与反馈。
#
# 机械检查是证据不是结论；本模块只负责「把证据摆到 rubric 对应的格子里」。

from __future__ import annotations

# BUS100 Assessment 1 · Annotated Bibliography 官方评分标准
RUBRIC = [
    ("Critical analysis",        30, "穿透文献：评方法局限/样本·语境限制，并说明对'我论文'论点的含义"),
    ("Synthesis of article",     30, "三篇之间构成关系（递进/互补），每篇在'我论证链'的哪一环"),
    ("Article quality",          15, "主动评判来源层级：peer-reviewed / specialist / commercial interest / currency"),
    ("Referencing and citation", 15, "文末条目 ↔ 正文 in-text 呼应；APA 7（et al.、年份、页码）"),
    ("Formatting and mechanics", 10, "语言地道、衔接自然、无语法/拼写错、段落清晰"),
]

# 让 agent 在加载 academic-english-teacher Skill 后，用这张表做判断
TEACHER_SYSTEM_HINT = (
    "你现在是 academic-english-teacher（正常学术英语老师）。"
    "下面是根据引擎机械检查预填的 rubric 评分表，请基于它 + 学生原文，"
    "逐维给出等级（HD/D/Cr/P/F）与具体、指名到句子的反馈，并标出 AI 痕迹疑点。"
    "机械结果是线索不是结论——最终判断靠你读。"
)


def _citations_obs(fo: dict) -> list:
    """引用维度：从 forensic 直接取机械发现。"""
    out = []
    if not fo:
        return out
    ci = fo.get("citations", {})
    out.append(f"文献表 {ci.get('entries', '?')} 条，正文标注 {ci.get('in_text_count', '?')} 条")
    for k, label in (("missing_in_text", "文献表有但正文未引"),
                     ("missing_entry", "正文引了但文献表没有"),
                     ("block_missing", "注释段内未标年份（只靠文末/结论并列）"),
                     ("year_mismatch", "年份不一致")):
        if ci.get(k):
            out.append(f"⚠️ {label}：{'、'.join(ci[k][:4])}")
    if ci.get("loose_author_phrase"):
        out.append("⚠️ 正文用 'and some other authors' 代替 et al.（APA 7 要求三位以上用 et al.）")
    return out


def _article_quality_obs(dois: list) -> list:
    """文献质量维度：从 verify 的期刊/撤稿结果取机械发现。"""
    out = []
    for d in (dois or []):
        j = d.get("journal", {}) or {}
        tier = j.get("tier")
        name = j.get("name") or d.get("doi", {}).get("normalized", "")
        if d.get("doi", {}).get("retracted"):
            out.append(f"⛔ {name} — 已撤稿，不应引用")
        elif tier:
            out.append(f"{name} — 期刊档案：{tier}（{j.get('tier_reason','')[:40]}）")
        elif d.get("doi", {}).get("error"):
            out.append(f"{name} — DOI {d['doi']['error']}")
    return out


def _formatting_obs(fo: dict) -> list:
    """格式维度：从 forensic 元数据取机械发现。"""
    out = []
    if not fo:
        return out
    meta = fo.get("metadata", {})
    for x in meta.get("bad", []):
        out.append(f"❌ {x}")
    for x in meta.get("warn", []):
        out.append(f"⚠️ {x}")
    return out


def build_sheet(result: dict) -> list:
    """把一份 render_paper 的结果，转成 [(维度, 权重, 说明, [机械发现], [老师需人工判断的提示])]。"""
    fo = result.get("forensic")
    ai = result.get("ai") or {}
    dois = result.get("dois") or []

    sheet = []
    # Critical analysis 30% — 纯人工
    ca_clues, ca_prompt = [], []
    if fo:
        st = fo.get("structure", {})
        if st.get("lengths"):
            ca_clues.append(f"注释段长度变异系数 {st.get('cv'):.2f}，段首六词最高重合 {st.get('similarity'):.0%}")
        for h in fo.get("hints", []):
            if "要点" in h or "et al" in h:
                ca_clues.append(f"评阅提示：{h}")
    ca_prompt = [
        "逐篇判断：是否点出 ≥2 个具体局限（方法/样本/因果/普适性），并说明该局限如何限制其在'我论文'中的使用？",
        "是否只做摘要、缺批判评价？（只摘要→此维掉档）",
    ]
    sheet.append(("Critical analysis", 30, "穿透文献、评局限、关联自身论点", ca_clues, ca_prompt))

    # Synthesis 30% — 纯人工
    syn_prompt = [
        "三篇之间是否构成递进/互补结构？每篇被指派到'我论证链'的哪一环？",
        "还是三篇平行罗列、互不相关？（平行→此维掉档）",
    ]
    sheet.append(("Synthesis of article", 30, "三篇的关系与论证分工", [], syn_prompt))

    # Article quality 15%
    aq_obs = _article_quality_obs(dois)
    if not aq_obs:
        aq_obs = ["（无 DOI / 未核验期刊）"]
    sheet.append(("Article quality", 15, "来源层级与可信度评判", aq_obs,
                  ["学生稿中是否主动评判每篇的来源性质（peer-reviewed? specialist? purpose? currency?）？",
                   "三篇评价是否千篇一律（都'very reliable'）？"]))

    # Referencing 15%
    ref_obs = _citations_obs(fo)
    if not ref_obs:
        ref_obs = ["（无引用数据）"]
    sheet.append(("Referencing and citation", 15, "文末↔正文呼应、APA 7", ref_obs,
                  ["in-text 与文末条目是否真呼应（而非仅形式正确）？", "et al. / 年份 / 双词姓处理是否正确？"]))

    # Formatting 10%
    fmt_obs = _formatting_obs(fo)
    ai_note = ""
    if "score" in ai:
        ai_note = f"AI 特征强度 {ai['score']:.0%}（{ai['label']}）— 作线索"
        fmt_obs = [ai_note] + fmt_obs
    sheet.append(("Formatting and mechanics", 10, "语言、衔接、语法、段落", fmt_obs,
                  ["语言是否地道、衔接自然、有无语法/拼写错？",
                   "若 AI 概率高：是否还有真实 ownership（'my argument / my essay' 式措辞）？"]))
    return sheet


def render_teacher_section(rows: list) -> str:
    """生成 markdown 的『教师人工复核』段落，供 check.py 追加到 paper 输出。"""
    L = ["\n### 教师人工复核（rubric 评分表 · 待学术英语老师判定）"]
    L.append(f"> {TEACHER_SYSTEM_HINT}")
    for r in rows:
        L.append(f"\n**{r['file']}**")
        for name, weight, desc, obs, prompts in build_sheet(r):
            L.append(f"\n- **{name}（{weight}%）** — {desc}")
            if obs:
                for o in obs:
                    L.append(f"  - 机械证据：{o}")
            for p in prompts:
                L.append(f"  - ❓ 人工判断：{p}")
    return "\n".join(L)


if __name__ == "__main__":
    # 自测：用一份已跑过引擎的结果验证 build_sheet 不报错
    import json, sys
    if len(sys.argv) > 1:
        data = json.load(open(sys.argv[1]))
        rows = data if isinstance(data, list) else [data]
        print(render_teacher_section(rows))
