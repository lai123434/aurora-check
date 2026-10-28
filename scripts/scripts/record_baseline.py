# -*- coding: utf-8 -*-
"""录制 AI 检测回归基线。

    python scripts/record_baseline.py > tests/baseline_scores.json

只在**有意**调整权重/阈值并重新校准后运行，并提交新的基线文件；
PR 里需说明校准依据与所用语料。切勿为了让红灯变绿而随手重录。

真人对照语料通过环境变量指定（真实学生作业不入库）：
    AURORA_HUMAN_SAMPLE=/path/to/human.docx python scripts/record_baseline.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import aurora_check as ac  # noqa: E402

from tests.test_detection_regression import AI_SAMPLE_NAMES, HUMAN_KEY  # noqa: E402


def main():
    import evasion_suite

    out = {}

    for name in AI_SAMPLE_NAMES:
        txt = evasion_suite.SAMPLES[name]
        r = ac.run_ai_detection(txt)
        out[name] = {
            "score": round(r.ai_probability, 3),
            "label": r.label,
            "primary": round(r.primary_score or 0.0, 3),
            "stat_mean": round(r.stat_mean, 3),
            "truth": "AI",
        }

    human_path = os.environ.get("AURORA_HUMAN_SAMPLE", "")
    if not human_path:
        for c in ("/workspace/BUS100_A1_Annotated_Bibliography_Final.docx",
                  os.path.join(BASE, "tests", "fixtures", "human_sample.docx")):
            if os.path.exists(c):
                human_path = c
                break

    if human_path and os.path.exists(human_path):
        r = ac.run_ai_detection(ac.extract_text_from_file(human_path))
        out[HUMAN_KEY] = {
            "score": round(r.ai_probability, 3),
            "label": r.label,
            "primary": round(r.primary_score or 0.0, 3),
            "stat_mean": round(r.stat_mean, 3),
            "truth": "human",
            # 只记文件名，不把学生作业路径写进仓库
            "source": os.path.basename(human_path),
        }
    else:
        print(
            "# 警告：未找到真人对照语料，基线不含 H1。"
            "请设 AURORA_HUMAN_SAMPLE 或放到 tests/fixtures/human_sample.docx",
            file=sys.stderr,
        )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
