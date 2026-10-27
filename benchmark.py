# -*- coding: utf-8 -*-
"""
Aurora Check 基准测试

跑法：
    python benchmark.py            # 标准输出表格
    python benchmark.py --json     # 机器可读

数据集：examples/benchmark/*.txt
    文件名前缀即标签 —— human_* / ai_* / paraphrased_*
    paraphrased 视为"AI 生成后改写"，单独统计，不计入主指标
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aurora_check as core

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "examples", "benchmark")


def label_of(path: str) -> str:
    name = os.path.basename(path).lower()
    if name.startswith("human"):
        return "human"
    if name.startswith("paraphrased"):
        return "paraphrased"
    return "ai"


def auc(pos, neg):
    """Mann-Whitney U 形式，无需 sklearn。"""
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA, "*.txt")))
    if not files:
        print(f"未找到测试集：{DATA}")
        sys.exit(1)

    rows = []
    import time as _t
    for f in files:
        text = core.extract_text_from_file(f)
        _s = _t.perf_counter()
        r = core.run_check(text)
        _elapsed = _t.perf_counter() - _s
        if not r.ai:
            rows.append({
                "file": os.path.basename(f), "label": label_of(f),
                "score": None, "verdict": "rejected", "note": r.note,
            })
            continue
        rows.append({
            "file": os.path.basename(f),
            "label": label_of(f),
            "score": r.ai.ai_probability,
            "verdict": r.ai.label,
            "confidence": r.ai.confidence,
            "rewrite": bool(r.ai.rewrite_hint),
            "words": len(text.split()),
            "ms": round(_elapsed * 1000),
        })

    human = [r["score"] for r in rows if r["label"] == "human" and r["score"] is not None]
    ai = [r["score"] for r in rows if r["label"] == "ai" and r["score"] is not None]
    para = [r["score"] for r in rows if r["label"] == "paraphrased" and r["score"] is not None]

    if args.json:
        print(json.dumps({
            "rows": rows,
            "auc": round(auc(ai, human), 3) if human and ai else None,
            "gap": round(min(ai) - max(human), 3) if human and ai else None,
        }, ensure_ascii=False, indent=2))
        return

    ms = [r["ms"] for r in rows if "ms" in r]
    if ms:
        ms_sorted = sorted(ms)
        print(f"\n  单篇耗时    中位 {ms_sorted[len(ms)//2]}ms"
              f"   最长 {max(ms)}ms   （中文含模型推理约 1.3s，英文纯统计约 20ms）")

    print("\n" + "=" * 78)
    print("  Aurora Check · 基准测试")
    print("=" * 78)
    print(f"\n{'文件':<26}{'标签':<13}{'得分':>7}{'判定':>9}{'置信':>7}{'词数':>7}{'耗时':>9}")
    print("-" * 78)
    for r in rows:
        sc = f"{r['score']:.3f}" if r["score"] is not None else "—"
        print(f"{r['file']:<26}{r['label']:<13}{sc:>7}"
              f"{r.get('verdict',''):>9}{r.get('confidence',''):>7}{r.get('words',''):>7}"
              f"{r.get('ms',''):>7}ms")

    print("\n" + "-" * 78)
    if human and ai:
        a = auc(ai, human)
        gap = min(ai) - max(human)
        correct = sum(1 for r in rows
                      if (r["label"] == "human" and r["verdict"] == "human")
                      or (r["label"] == "ai" and r["verdict"] == "ai"))
        total = len(human) + len(ai)
        print(f"  人写区间     {min(human):.3f} ~ {max(human):.3f}   （{len(human)} 份）")
        print(f"  AI 区间      {min(ai):.3f} ~ {max(ai):.3f}   （{len(ai)} 份）")
        print(f"  间隔         {gap:+.3f}")
        print(f"  AUC          {a:.3f}")
        print(f"  判定准确率   {correct}/{total}")
        if total < 20:
            print(f"  ⚠ 样本仅 {total} 份，AUC 与准确率波动很大，"
                  "仅用于回归检查，不能当作对外宣称的精度指标。")
    if para:
        hit = sum(1 for r in rows if r["label"] == "paraphrased" and r.get("rewrite"))
        print(f"\n  AI 改写样本  {len(para)} 份，得分 "
              f"{min(para):.3f}~{max(para):.3f}，改写提示命中 {hit}/{len(para)}")
        print("  （改写会压低绝大多数信号，总分参考价值低，以 rewrite 提示为准）")

    ms = [r["ms"] for r in rows if "ms" in r]
    if ms:
        ms_sorted = sorted(ms)
        print(f"\n  单篇耗时    中位 {ms_sorted[len(ms)//2]}ms"
              f"   最长 {max(ms)}ms   （中文含模型推理约 1.3s，英文纯统计约 20ms）")

    print("\n" + "=" * 78)
    print("  说明：分数是多个统计信号的加权结果，不是「AI 生成概率」的校准值。")
    print("  只有同一批文本之间的相对排序有意义。")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
