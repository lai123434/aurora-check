# -*- coding: utf-8 -*-
"""复测对比：校准后（3.13）朋友 v2 与用户 Rev9 文章的检测结果。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aurora_check as ac

CASES = [
    ("朋友 v2（100分标准）", "/workspace/朋友_v2_body_final.txt"),
    ("用户 Rev9（90分标准）", "/workspace/Precheck_Oct30_Rev9_body.txt"),
]


def main():
    print("=" * 78)
    print("  Aurora Check · 校准后复测（corpus/calibration.json 已生效）")
    print("=" * 78)
    for label, path in CASES:
        if not os.path.exists(path):
            print(f"\n  ✗ 找不到 {path}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        wc = len(text.split())
        print(f"\n{'─' * 78}")
        print(f"  【{label}】 {wc} 词")
        for mode in ("normal", "strict"):
            t0 = time.perf_counter()
            r = ac.run_check(text, mode=mode)
            dt = time.perf_counter() - t0
            if not r.ai:
                print(f"    [{mode}] rejected: {r.note}")
                continue
            a = r.ai
            print(f"    [{mode:<6}] AI分={a.ai_probability:.3f}  判定={a.label}"
                  f"  置信={a.confidence}  耗时={dt:.1f}s")
            # 伪困惑度原始值
            pp = getattr(r, "_ml_detail", None) or {}
            raw_pp = None
            try:
                raw_pp = getattr(r.ai, "_raw_pp", None)
            except Exception:
                pass
            # 从 report 里找 pseudo_perplexity
            d = r.to_dict()
            eng = (d.get("ai") or {}).get("engines") or {}
            pp_engine = eng.get("mlm-perplexity") if isinstance(eng, dict) else None
            print(f"            伪困惑度引擎得分={pp_engine}  raw_pp={raw_pp}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
