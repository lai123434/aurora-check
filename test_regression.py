# -*- coding: utf-8 -*-
"""
Aurora Check v2 回归测试
用法: python3 test_regression.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aurora_check as A

BASE = os.path.dirname(os.path.abspath(__file__))
CASES = [
    # (名称, 文件, 期望上限, 期望下限, 类别说明)
    ("human_sample_1",       "examples/test_set/human_sample_1.txt", 0.40, 0.00, "人写"),
    ("paraphrased_sample_1", "examples/test_set/paraphrased_sample_1.txt", 0.45, 0.00, "AI 改写（应命中改写提示）"),
    ("ai_sample_1",          "examples/test_set/ai_sample_1.txt", 1.00, 0.55, "AI 原创"),
]

def main():
    ok = 0
    fails = []
    rows = []
    for name, rel, hi, lo, note in CASES:
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            path = os.path.join(BASE, "examples/test_set", os.path.basename(rel))
        if not os.path.exists(path):
            print(f"跳过（文件不存在）: {rel}")
            continue
        text = A.extract_text_from_file(path)
        r = A.run_check(text).ai
        p = r.ai_probability
        passed = lo <= p <= hi
        ok += passed
        if not passed:
            fails.append((name, p, lo, hi))
        rows.append((name, note, p, r.label, r.confidence, "改写提示" if r.rewrite_hint else "—", "PASS" if passed else "FAIL"))

    print(f"\n{'样本':<22}{'类别':<22}{'AI概率':>8}{'判定':>9}{'置信':>8}{'提示':>10}{'结果':>8}")
    print("-" * 92)
    for n, note, p, lab, conf, hint, st in rows:
        print(f"{n:<22}{note:<22}{p:>8.3f}{lab:>9}{conf:>8}{hint:>10}{st:>8}")

    # 分离度检查：AI 原创应显著高于人写
    d = {r[0]: r[2] for r in rows}
    if "ai_sample_1" in d and "human_sample_1" in d:
        gap = d["ai_sample_1"] - d["human_sample_1"]
        print(f"\n人写 / AI 分离度: {gap:+.3f}（v1 为 +0.266，目标 ≥0.30）")
        if gap < 0.28:
            fails.append(("分离度", gap, 0.25, 1.0))

    print(f"\n通过 {ok}/{len(rows)}")
    if fails:
        print("失败项：")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("全部通过")

if __name__ == "__main__":
    main()
