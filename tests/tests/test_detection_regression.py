# -*- coding: utf-8 -*-
"""AI 检测回归快照（slow：需 --runslow 且需深度学习模型）。

为什么要有快照
--------------
本工具是**多引擎加权**系统，改一个权重/阈值就可能让一整类文本的判定
集体漂移，而这种漂移肉眼很难发现。这份测试把 5 篇纯 AI 文本 + 1 篇真人
文本的分数与标签冻结下来，任何超出容差的变化都会报警。

两类断言，缺一不可
------------------
1. **数值快照**：分数在基线 ±TOL 内，防止暗中漂移。
2. **语义不变量**：比数值更稳、也更重要——
   - 5 篇纯 AI 文本**不得被判为 human**（这是我们专门修过的「漏判」，
     详见 CHANGELOG 3.3.0，绝不能退回去）
   - 真人文本**不得被判为 ai**（防止把真人误伤成确定 AI）

运行
----
    pytest --runslow tests/test_detection_regression.py

基线不吻合怎么办
----------------
若你是**有意**调整了权重/阈值并重新校准，跑：
    python scripts/record_baseline.py > tests/baseline_scores.json
再提交新的基线，并在 PR 里说明校准依据与所用语料。
"""
import json
import os
import pytest

import aurora_check as ac

TOL = 0.06          # 数值容差：允许小幅抖动，抓得住真实漂移
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline_scores.json")

# 5 篇纯 AI 合成样本（定义在 evasion_suite.py，均为内联合成文本、无隐私风险）
AI_SAMPLE_NAMES = [
    "S1_obvious_ai", "S2_casual_ai", "S3_student_ai",
    "S4_imperfect_ai", "S5_natural_essay",
]
HUMAN_KEY = "H1_real_student_final"


def _load_baseline():
    if not os.path.exists(BASELINE_FILE):
        pytest.fail(
            f"缺少基线文件 {BASELINE_FILE}。"
            "请先运行 scripts/record_baseline.py 生成并提交基线。"
        )
    with open(BASELINE_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module", params=AI_SAMPLE_NAMES, ids=lambda n: f"AI:{n}")
def ai_case(request, require_models, ai_samples):
    name = request.param
    return name, ac.run_ai_detection(ai_samples[name])


@pytest.mark.slow
def test_ai_never_confidently_human(ai_case):
    """核心不变量：纯 AI 文本不得被自信判为 human。

    这不是「越低越好」的游戏——本工具刻意采用召回优先策略，
    宁可把 AI 判进 mixed 交人工复核，也绝不能给出「这是人写的」这种结论。
    """
    name, r = ai_case
    assert r.label != "human", (
        f"[{name}] 实为纯 AI 却被判为 human（score={r.ai_probability:.3f}）。"
        "这是 3.3.0 专门修复的漏判回归，请检查 ML 锚定/困惑度反向锚定是否失效。"
    )


@pytest.mark.slow
def test_ai_scores_match_baseline(ai_case):
    """数值快照：防止分数暗中漂移。"""
    name, r = ai_case
    base = _load_baseline()
    if name not in base:
        pytest.skip(f"基线中没有 {name}，跳过数值比对")
    expected = base[name]["score"]
    assert abs(r.ai_probability - expected) <= TOL, (
        f"[{name}] 分数漂移：基线 {expected:.3f} → 实际 {r.ai_probability:.3f}"
        f"（差 {abs(r.ai_probability - expected):.3f} > 容差 {TOL}）。"
        "若是有意调参，请重新生成基线并说明校准依据。"
    )
    assert r.label == base[name]["label"], (
        f"[{name}] 标签变化：基线 {base[name]['label']} → 实际 {r.label}"
    )


@pytest.mark.slow
def test_human_not_flagged_as_ai(require_models, human_sample_path):
    """真人文本不得被判为 ai（越这条线就是误伤真人）。"""
    text = ac.extract_text_from_file(human_sample_path)
    r = ac.run_ai_detection(text)

    assert r.label != "ai", (
        f"真人对照稿被判为 ai（score={r.ai_probability:.3f}），属严重误伤。"
    )


@pytest.mark.slow
def test_human_score_matches_baseline(require_models, human_sample_path):
    text = ac.extract_text_from_file(human_sample_path)
    r = ac.run_ai_detection(text)

    base = _load_baseline()
    key = "H1_real_student_final"
    if key not in base:
        pytest.skip(f"基线中没有 {key}，跳过数值比对")
    expected = base[key]["score"]
    assert abs(r.ai_probability - expected) <= TOL, (
        f"真人稿分数漂移：基线 {expected:.3f} → 实际 {r.ai_probability:.3f}"
    )


@pytest.mark.slow
def test_score_is_explainable(require_models, human_sample_path):
    """可解释性回归：报告必须能回答「为什么是这个分」。

    终分若来自 ML 锚定，aggregation 里必须有 ①②③ 三步；
    少了任何一步，报告就又退回成「只给一个数字」。
    """
    text = ac.extract_text_from_file(human_sample_path)
    r = ac.run_ai_detection(text)

    assert r.aggregation, "aggregation 为空，报告无法解释分数来源"
    joined = " ".join(r.aggregation)
    for step in ("①", "②", "③"):
        assert step in joined, f"缺少判定步骤 {step}：{r.aggregation}"
    assert r.primary_score is not None, "未记录主力检测器分数"
    assert 0.0 <= r.stat_mean <= 1.0, f"stat_mean 越界：{r.stat_mean}"
