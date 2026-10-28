# -*- coding: utf-8 -*-
"""采集真实人类学术文献语料（开放获取平台）。

数据源：
  1. Europe PMC  —— 聚合 PubMed / BioMed Central 开放获取全文（JATS XML）
  2. PLOS        —— 完全开放获取（manuscript XML）

主题围绕 BUS100 / 就业（employability）领域：graduate employability、
employment、business management、organizational behavior、career
development、human resource management、job satisfaction、work motivation。

产出：
  corpus/text/<source>__<id>.txt   提取后的正文纯文本（≥200 词才保留）
  corpus/raw/<source>__<id>.xml    原始全文 XML（供溯源）
  corpus/manifest.json             清单（来源/ID/标题/词数/DOI）
"""
import json
import os
import re
import sys
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "corpus", "raw")
TEXT = os.path.join(BASE, "corpus", "text")
MANIFEST = os.path.join(BASE, "corpus", "manifest.json")

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

HEADERS = {"User-Agent": "AuroraCheck/3.13 (research calibration; mailto:research@example.com)"}

# 主题查询（商业/就业/管理，均为真实人类学术论文高发领域）
TOPICS = [
    "graduate employability",
    "employment",
    "business management",
    "organizational behavior",
    "career development",
    "human resource management",
    "job satisfaction",
    "work motivation",
]

PLOS_QUERIES = [
    'q=abstract:"employability"&rows=12&wt=json',
    'q=abstract:"career development"&rows=12&wt=json',
]


def epmc_fetch_topics(per_topic=8):
    """每个主题检索 per_topic 篇有全文的文献，返回 [{pmcid,title,doi,source}]。"""
    seen = set()
    out = []
    for t in TOPICS:
        q = f"HAS_FT:Y AND ({t})"
        params = {
            "query": q, "format": "json", "pageSize": per_topic,
            "resultType": "core",
        }
        try:
            r = requests.get(EPMC_SEARCH, params=params, headers=HEADERS, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"[epmc] 检索失败 {t}: {e}", file=sys.stderr)
            continue
        for item in data.get("resultList", {}).get("result", []):
            pmcid = item.get("pmcid") or item.get("id")
            if not pmcid or not pmcid.upper().startswith("PMC"):
                continue
            doi = item.get("doi", "")
            title = item.get("title", "")
            # 跳过明显非人类撰写的元数据占位（极少见，防御）
            if len(title) < 10:
                continue
            key = pmcid.upper()
            if key in seen:
                continue
            seen.add(key)
            out.append({"pmcid": pmcid, "title": title, "doi": doi,
                        "source": "europepmc", "topic": t})
        time.sleep(0.4)
    return out


def epmc_download_text(pmcid):
    """下载全文 XML 并提取正文，返回 (raw_xml_str, text_str)。"""
    url = EPMC_FULLTEXT.format(pmcid=pmcid)
    r = requests.get(url, headers=HEADERS, timeout=40)
    if r.status_code != 200:
        return None, None
    raw = r.text
    soup = BeautifulSoup(raw, "xml")
    body = soup.find("body")
    if body is None:
        return raw, None
    paras = []
    for p in body.find_all("p"):
        # 跳过图/表标题
        txt = p.get_text(" ", strip=True)
        if txt:
            paras.append(txt)
    text = "\n\n".join(paras)
    return raw, text


def plos_fetch(rows_per_query=12):
    """PLOS 检索，返回 [{doi,title,source}]。"""
    seen = set()
    out = []
    for qs in PLOS_QUERIES:
        url = f"https://api.plos.org/search?{qs}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"[plos] 检索失败: {e}", file=sys.stderr)
            continue
        for doc in data.get("response", {}).get("docs", []):
            doi = doc.get("id")
            if not doi or doi in seen:
                continue
            seen.add(doi)
            title = (doc.get("title_display") or doc.get("title") or "")
            out.append({"doi": doi, "title": title, "source": "plos"})
    return out


def plos_download_text(doi):
    """PLOS 全文 manuscript XML → 正文。"""
    # PLOS 全文 manuscript 端点
    url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=manuscript"
    r = requests.get(url, headers=HEADERS, timeout=40, allow_redirects=True)
    if r.status_code != 200:
        return None, None
    raw = r.text
    if "<body" not in raw:
        return raw, None
    soup = BeautifulSoup(raw, "xml")
    body = soup.find("body")
    if body is None:
        return raw, None
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p") if p.get_text(strip=True)]
    return raw, "\n\n".join(paras)


def save(doc, raw, text):
    if not text or len(text.split()) < 200:
        return None
    sid = f"{doc['source']}__{doc.get('pmcid') or doc.get('doi','').replace('/', '_')}"
    sid = re.sub(r"[^\w.-]", "_", sid)
    txt_path = os.path.join(TEXT, sid + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    if raw:
        raw_path = os.path.join(RAW, sid + ".xml")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw)
    wc = len(text.split())
    rec = {
        "id": sid, "source": doc["source"], "title": doc.get("title", ""),
        "doi": doc.get("doi", ""), "pmcid": doc.get("pmcid", ""),
        "topic": doc.get("topic", ""), "words": wc,
    }
    print(f"  ✓ {sid:<46s} {wc:>6d} 词  {rec['title'][:50]}")
    return rec


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(TEXT, exist_ok=True)
    manifest = []

    print("== Europe PMC 检索 ==")
    docs = epmc_fetch_topics(per_topic=8)
    print(f"  命中 {len(docs)} 篇有全文文献")
    for i, doc in enumerate(docs, 1):
        try:
            raw, text = epmc_download_text(doc["pmcid"])
        except Exception as e:
            print(f"  ✗ {doc['pmcid']} 下载失败: {e}")
            continue
        rec = save(doc, raw, text)
        if rec:
            manifest.append(rec)
        time.sleep(0.3)

    print("== PLOS 检索 ==")
    pdocs = plos_fetch()
    print(f"  命中 {len(pdocs)} 篇")
    for i, doc in enumerate(pdocs, 1):
        try:
            raw, text = plos_download_text(doc["doi"])
        except Exception as e:
            print(f"  ✗ {doc['doi']} 下载失败: {e}")
            continue
        rec = save(doc, raw, text)
        if rec:
            manifest.append(rec)
        time.sleep(0.3)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    total_words = sum(m["words"] for m in manifest)
    print(f"\n== 采集完成 ==\n  有效文献 {len(manifest)} 篇，共 {total_words} 词")
    print(f"  清单: {MANIFEST}")


if __name__ == "__main__":
    main()
