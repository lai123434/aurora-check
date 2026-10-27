# -*- coding: utf-8 -*-
"""
学术核查模块：DOI 真伪 + 期刊档案 + 分级

数据源（全部公开、免 key）：
    https://doi.org/api/handles/{doi}      DOI 是否注册
    https://api.crossref.org/works/{doi}   论文元数据
    https://api.openalex.org/sources/issn:{issn}  期刊档案
    https://doaj.org/api/search/journals/issn:{issn}  开放获取白名单

设计原则：
    只报告可核验的事实（是否注册、发文量、被引、是否 DOAJ），
    涉及"质量"的判断给出依据和不确定度，不替代同行评议。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional

UA = {"User-Agent": "AuroraCheck/2.7 (academic-verification; mailto:noreply@example.com)"}
TIMEOUT = 15


def _get(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except Exception:
        return None


def _head_ok(url: str) -> Optional[bool]:
    """只判断可达性：200=存在，404=不存在，None=网络问题（不作结论）。"""
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return None
    except Exception:
        return None


DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s\"<>]+$")


@dataclass
class DoiCheck:
    doi: str = ""
    normalized: str = ""
    registered: Optional[bool] = None     # None = 无法判断
    title: str = ""
    authors: list = field(default_factory=list)
    journal: str = ""
    issn: list = field(default_factory=list)
    year: str = ""
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    type: str = ""
    url: str = ""
    # 论文级风险（来自 OpenAlex works）
    retracted: Optional[bool] = None        # 是否被撤稿
    cited_percentile: Optional[int] = None  # 同年同领域被引百分位（100 = 前 1%）
    topic: str = ""
    error: str = ""


@dataclass
class JournalCheck:
    issn: str = ""
    name: str = ""
    publisher: str = ""
    h_index: Optional[int] = None
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None
    is_oa: Optional[bool] = None
    in_doaj: Optional[bool] = None
    apc: str = ""
    mean_cited_2y: Optional[float] = None
    homepage: str = ""
    yearly: list = field(default_factory=list)   # [(year, works)] 近 6 年
    growth: float = 0.0                          # 近三年相对三年前的倍数
    tier: str = "未知"                            # 未见异常 / 需注意 / 审稿强度存疑 / 未知
    tier_reason: str = ""
    impact_note: str = ""                        # 影响力参考，非质量结论
    risk_flags: list = field(default_factory=list)
    error: str = ""


def normalize_doi(raw: str) -> str:
    if not raw:
        return ""
    d = raw.strip()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    d = d.strip().strip(".,;)")
    return d


def check_doi(raw: str) -> DoiCheck:
    """核验 DOI 是否真实存在，并取回元数据。"""
    out = DoiCheck(doi=raw)
    d = normalize_doi(raw)
    out.normalized = d

    if not d:
        out.error = "未提供 DOI"
        return out
    if not DOI_RE.match(d):
        out.error = "格式不合法（DOI 形如 10.xxxx/xxxxx）"
        out.registered = False
        return out

    exists = _head_ok("https://doi.org/api/handles/" + urllib.parse.quote(d))
    out.registered = exists

    meta = _get("https://api.crossref.org/works/" + urllib.parse.quote(d))
    if meta and meta.get("status") == "ok":
        m = meta["message"]
        out.title = (m.get("title") or [""])[0]
        out.journal = (m.get("container-title") or [""])[0]
        out.publisher = m.get("publisher", "")
        out.issn = m.get("ISSN") or []
        out.volume = m.get("volume", "")
        out.issue = m.get("issue", "")
        out.pages = m.get("page", "")
        out.type = m.get("type", "")
        out.url = m.get("URL", "")
        issued = m.get("issued", {}).get("date-parts", [[""]])
        out.year = str(issued[0][0]) if issued and issued[0] else ""
        for a in (m.get("author") or [])[:12]:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x)
            if name:
                out.authors.append(name)
    elif exists:
        out.error = "DOI 已注册，但 Crossref 无元数据（可能来自其他注册机构）"

    # 论文级风险：撤稿状态与领域百分位（Crossref 不提供，需 OpenAlex）
    if exists:
        w = _get("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(d))
        if w and w.get("id"):
            out.retracted = bool(w.get("is_retracted"))
            pct = w.get("cited_by_percentile_year") or {}
            if isinstance(pct, dict) and pct.get("min") is not None:
                # 只有发表满 2 年的文章，百分位才有统计意义。
                # 新文章样本少，OpenAlex 常直接给到 100，看起来像"前 1%"，
                # 实际上只是"还没被引够久"。
                import datetime as _dt
                try:
                    _yr = int(out.year or 0)
                except ValueError:
                    _yr = 0
                if _yr and _dt.date.today().year - _yr >= 2:
                    out.cited_percentile = int(pct.get("max") or pct.get("min"))
                elif _yr:
                    out.cited_percentile = -1     # 标记：太新，暂不评价
            out.topic = ((w.get("primary_topic") or {}).get("display_name") or "")
            # Crossref 标题常带 "RETRACTED:" 前缀，作为二次确认
            if not out.retracted and out.title.upper().startswith("RETRACTED"):
                out.retracted = True

    return out


def check_journal(issn: str) -> JournalCheck:
    """按 ISSN 取期刊档案，给出影响力档位与风险标记。"""
    out = JournalCheck(issn=issn)
    if not issn:
        out.error = "未提供 ISSN"
        return out

    # OpenAlex 要求 ISSN 带连字符，且 /sources/issn:xx 返回单个对象
    key = issn.strip()
    src = _get(f"https://api.openalex.org/sources/issn:{key}")

    # 兼容两种返回形态：单个对象 / {results:[...]}
    obj = None
    if src and src.get("id"):
        obj = src
    elif src and src.get("results"):
        obj = src["results"][0]

    if obj:
        s = obj
        out.name = s.get("display_name", "")
        out.publisher = (s.get("host_organization_name") or "")
        ss = s.get("summary_stats") or {}
        out.h_index = ss.get("h_index")
        out.mean_cited_2y = ss.get("2yr_mean_citedness")
        out.works_count = s.get("works_count")
        out.cited_by_count = s.get("cited_by_count")
        out.is_oa = s.get("is_oa")
        out.in_doaj = s.get("is_in_doaj")
        out.homepage = s.get("homepage_url", "")
        apc = s.get("apc_prices") or []
        if apc and isinstance(apc, list) and apc[0]:
            out.apc = f"{apc[0].get('price')} {apc[0].get('currency')}"
        # 年发文量曲线：判别"灌水刊"最有效的信号。
        # h-index 只会随发文量单调上涨，永远区分不了"办好刊"和"多收稿"；
        # 增长曲线能看出一个刊是稳定办刊还是突然放量。
        # 当年数据尚未统计完整（例如 10 月只录入了半年），
        # 若混入增长率的分子会让"暴跌"假象丛生，一律剔除。
        import datetime as _dt
        this_year = _dt.date.today().year
        out.yearly = [(x.get("year"), x.get("works_count"))
                      for x in (s.get("counts_by_year") or [])
                      if x.get("year") and x.get("year") < this_year][:6]
    else:
        dj = _get(f"https://doaj.org/api/search/journals/issn:{key}")
        if dj and dj.get("total", 0) > 0:
            j = dj["results"][0]["bibjson"]
            out.name = j.get("title", "")
            out.publisher = j.get("publisher", {}).get("name", "")
            out.in_doaj = True
            out.is_oa = True
            out.homepage = (j.get("ref") or {}).get("journal", "")
        else:
            out.error = "未在 OpenAlex / DOAJ 中查到该 ISSN"
            return out

    # ---- 影响力：只作参考，不作质量结论 ----
    h = out.h_index or 0
    if h >= 100:
        out.impact_note = f"h-index {h}，国际影响力较高"
    elif h >= 45:
        out.impact_note = f"h-index {h}，领域内有一定影响力"
    elif h >= 12:
        out.impact_note = f"h-index {h}，影响力有限"
    elif out.h_index is not None:
        out.impact_note = f"h-index {h}，偏低"
    else:
        out.impact_note = "无 h-index 数据"

    # ---- 发文量趋势：判别灌水刊的核心信号 ----
    if len(out.yearly) >= 4:
        recent = [c for y, c in out.yearly[:3] if c]
        older = [c for y, c in out.yearly[3:6] if c]
        if recent and older:
            r = sum(recent) / len(recent)
            o = sum(older) / len(older)
            if o > 0:
                out.growth = round(r / o, 2)

    danger, caution = [], []

    if out.works_count is not None and out.works_count < 30:
        caution.append(f"累计发文仅 {out.works_count} 篇，可能是新刊或小型刊")

    latest = out.yearly[0][1] if out.yearly else None

    # 放量：年发文量过大或短期内暴涨
    if latest and latest > 8000:
        danger.append(f"年发文 {latest} 篇，属巨型 OA，单篇审稿强度难以保证")
    elif latest and latest > 3000 and out.growth >= 2.0:
        danger.append(f"年发文比三年前增长 {out.growth:.1f} 倍，现为 {latest} 篇")
    elif out.growth >= 3.0:
        # 订阅制期刊扩刊多是专题组稿、页码调整，作者不付费，不构成敛财信号；
        # 只有 OA / DOAJ 刊的扩刊才和"多收快发"有关。分开展示，不升风险等级。
        if out.is_oa or out.in_doaj:
            caution.append(f"近三年发文量增长 {out.growth:.1f} 倍，扩张较快")
        else:
            out.risk_flags.append(f"近三年发文量增长 {out.growth:.1f} 倍（订阅制，扩刊不涉及版面费）")
    elif out.growth >= 1.8 and latest and latest > 1000:
        caution.append(f"近三年发文量增长 {out.growth:.1f} 倍")

    # 缩量：发文量骤降往往意味着被数据库剔除、停止收稿或整改
    if out.growth and out.growth <= 0.4 and latest and latest < 2000:
        caution.append(f"年发文从三年前的水平骤降至 {latest} 篇，"
                       "可能被数据库剔除或停止收稿，需核实当前收录状态")

    # 见顶回落：先暴涨后骤降，是期刊被 on hold / 整改 / 停收特刊的典型曲线。
    # 单看增长率会漏掉（首尾可能持平），必须看峰值出现在哪里。
    if len(out.yearly) >= 4:
        counts = [c for y, c in out.yearly if c]
        peak = max(counts)
        peak_idx = counts.index(peak)
        if peak_idx <= 2 and latest and peak > 0 and latest / peak <= 0.45 and peak >= 3000:
            danger.append(f"发文量曾达 {peak} 篇后骤降至 {latest} 篇（回落 "
                          f"{100 - int(latest / peak * 100)}%），"
                          "符合被数据库暂缓收录或整改的形态")

    if out.is_oa and not out.in_doaj:
        caution.append("宣称开放获取但不在 DOAJ 白名单")

    # APC 单独不构成风险：Nature / Cell / Lancet 的 OA 版面费都在 1 万美元以上。
    # 只有"高版面费 + 放量收稿"同时出现才说明问题。
    if out.apc and out.apc.split()[0].replace(".", "").isdigit():
        _apc = float(out.apc.split()[0])
        # 订阅制期刊（is_oa=False）的作者不付版面费，OpenAlex 里的 APC 只是
        # "可选金色 OA"的标价。Business and Professional Communication Quarterly
        # （SAGE，Scopus Q1）就因此被误判成"需核实办刊动机"。
        # 只有作者真的要付钱（OA / DOAJ）时，费用与扩刊叠加才构成信号。
        _pay_to_publish = bool(out.is_oa or out.in_doaj)
        if _apc > 2000 and _pay_to_publish and ((latest and latest > 5000) or out.growth >= 2.0):
            caution.append(f"版面费 {out.apc} 且年发文 {latest} 篇，需核实办刊动机")
        elif _apc > 2000:
            out.risk_flags.append(f"版面费 {out.apc}（可选 OA 标价，订阅制投稿不收取）")
    if out.in_doaj:
        out.risk_flags.append("在 DOAJ 开放获取白名单内")

    out.risk_flags = out.risk_flags + danger + caution

    if danger:
        out.tier = "审稿强度存疑"
        out.tier_reason = "；".join(danger)
    elif caution:
        out.tier = "需注意"
        out.tier_reason = "；".join(caution)
    else:
        out.tier = "未见异常"
        if out.growth >= 2.0 and out.yearly:
            _y, _c = out.yearly[0]
            out.tier_reason = (f"{_y} 年发文 {_c} 篇，较三年前扩刊；"
                               "订阅制期刊，作者不付版面费")
        elif out.yearly:
            _y, _c = out.yearly[0]
            out.tier_reason = f"{_y} 年发文 {_c} 篇，与近年持平"
        else:
            out.tier_reason = "无发文量数据"

    return out


def verify_paper(raw_doi: str) -> dict:
    """一键核查：DOI 真伪 → 期刊档案 → 结论。"""
    d = check_doi(raw_doi)
    j = JournalCheck()
    # 论文可能同时有印刷版/电子版 ISSN，逐个试到能查到为止
    for _issn in (d.issn or []):
        _t = check_journal(_issn)
        if not _t.error:
            j = _t
            break
    else:
        if d.issn:
            j = check_journal(d.issn[0])

    verdict = []
    if d.registered is False:
        verdict.append("DOI 未注册——这个编号不存在，可能是编造的")
    elif d.registered is None:
        verdict.append("DOI 状态无法确认（网络问题），请稍后重试")
    else:
        verdict.append("DOI 已注册")

    if d.retracted is True:
        verdict.append("⛔ 该论文已被撤稿——不应引用")
    elif d.retracted is False:
        verdict.append("未见撤稿记录")

    if d.title and not d.journal:
        verdict.append("Crossref 无期刊信息，可能不是期刊论文")

    return {
        "doi": asdict(d),
        "journal": asdict(j),
        "verdict": verdict,
        "confidence": "high" if d.registered is not None else "low",
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "10.3389/frai.2025.1629320"
    r = verify_paper(target)
    print(json.dumps(r, ensure_ascii=False, indent=2))
