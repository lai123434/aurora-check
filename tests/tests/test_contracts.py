"""接口契约测试——快速、不依赖任何模型，任何时候都应当通过。

这里的每一条都对应评审指出的具体缺陷，目的是防止改回去：
- external_similarity 曾被写死 0.0 假装「查重过」（名实不符）
- _detect_mlm_perplexity 注解与实际返回值不一致
- _compute_engine_scores 文档与返回元组长度不一致
"""
import inspect
import re

import pytest

import aurora_check as ac


# ---------------------------------------------------------------
# 1. 外部查重：必须诚实标记为未实现
# ---------------------------------------------------------------

def test_external_similarity_none_when_network_disabled():
    """关闭联网时，外部相似度必须是 None，绝不能伪装成 0.0。

    回归背景：旧实现返回 external_similarity=0.0，会让用户以为
    「查过外部库、相似度为 0%」，属严重的名实不符。
    现联网查重（Crossref）已实现，但「未查」（enable_network=False）
    仍应为 None，与「查了但命中 0.0」严格区分。
    """
    text = ("This is a sufficiently long paragraph for the detector "
            "to accept as valid natural language prose. " * 3)
    result = ac.run_plagiarism_check(text + "\n\n" + text, enable_network=False)

    assert result.external_similarity is None, (
        f"关闭联网时外部相似度应为 None，实际为 {result.external_similarity!r}。"
        "若为 0.0 说明退回成「假装查过」的旧行为。"
    )


def test_self_overlap_is_measured():
    """内部自重叠是真实实现的功能，重复文本应检出高重叠。"""
    para = ("artificial intelligence employability graduates skills "
            "development curriculum industry demand " * 4)
    result = ac.run_plagiarism_check(para + "\n\n" + para, enable_network=False)

    assert result.self_overlap > 0.5, (
        f"两段完全相同的段落，自重叠应接近 1，实际 {result.self_overlap}"
    )


# ---------------------------------------------------------------
# 2. 返回契约与注解一致性（防止 docs 和代码继续漂移）
# ---------------------------------------------------------------

def test_mlm_perplexity_returns_tuple_or_none():
    """注解是 Optional[Tuple[float, float]]，实际就必须是元组或 None。"""
    anno = inspect.signature(ac._detect_mlm_perplexity).return_annotation
    assert "Tuple" in str(anno), f"返回注解应含 Tuple[float, float]，实际 {anno}"

    res = ac._detect_mlm_perplexity("word " * 5)  # 文本过短
    assert res is None or (isinstance(res, tuple) and len(res) == 2), (
        f"应返回 None 或 (ai_score, pseudo_pp) 二元组，实际 {res!r}"
    )


def test_compute_engine_scores_returns_four_tuple():
    """文档字符串承诺四元组，返回就必须是四项。"""
    src = inspect.getsource(ac._compute_engine_scores)
    doc = ac._compute_engine_scores.__doc__ or ""
    returns = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("return ")]

    assert returns, "找不到 return 语句"
    n_ret = returns[0].count(",") + 1
    assert n_ret == 4, (
        f"_compute_engine_scores 实际返回 {n_ret} 个值，"
        f"但文档字符串写的是四元组，二者不一致：{returns[0]}"
    )
    assert "ml_detail" in returns[0], "返回值应包含 ml_detail"
    assert "window_scores" in doc, "文档字符串应描述 window_scores"


def test_registry_has_no_bare_tuples():
    """注册表条目必须统一为 EngineSpec，不能混用裸三元组。

    背景：失效引擎曾用 `(0, 0.0, "说明")` 裸元组，虽然解包兼容，
    但类型不统一，改表时极易出错。
    """
    src = inspect.getsource(ac.run_ai_detection)
    bare = re.findall(r'^\s*"[\w-]+":\s+\(', src, flags=re.M)
    assert not bare, f"注册表仍有裸元组条目：{bare}"


# ---------------------------------------------------------------
# 3. 输入校验：拒绝垃圾输入而不是硬给分
# ---------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "ab", "aaa", "!!!", "12345"])
def test_validate_rejects_trash(bad):
    """无意义/过短输入应被拒绝，而不是输出一个看似正常的分数。

    注意：validate_text 返回 (ok: bool, reason: str) 元组，不是裸布尔。
    """
    ok, reason = ac.validate_text(bad)
    assert ok is False, f"本应拒绝的垃圾输入被放行：{bad!r}"
    assert reason, "拒绝时必须给出可读原因，便于用户知道为什么判不了"


def test_validate_accepts_real_prose():
    # 门槛：≥200 字符、≥50 词、≥5 个完整句子
    prose = (
        "This article examines how artificial intelligence reshapes graduate "
        "employability in China, and I chose it because I need this evidence "
        "for my second essay paragraph. The authors surveyed 148 graduates "
        "from a single university in Peru, which is a small sample for such "
        "a broad claim. Their findings suggest that AI skills help graduates "
        "find work related to their major rather than any job at all. I "
        "think this limitation matters for my topic, since Chinese employers "
        "may value different abilities than Peruvian graduates described. "
        "In my next essay I will therefore compare these results with job "
        "advertisements collected from mainland recruitment websites."
    )
    ok, reason = ac.validate_text(prose)
    assert ok, f"正常散文被误拒：{reason}"


# ---------------------------------------------------------------
# 4. 缺失引擎可见化（收敛静默异常：报告要告诉用户「少了什么」）
# ---------------------------------------------------------------

def test_missing_engines_reported_when_ml_unavailable(monkeypatch):
    """不可用的引擎必须出现在 missing_engines，而不是被静默跳过。

    回归背景：评审第 5 条指出 ML 模型/外部 CLI 失败时静默回退，
    用户只看到分数，不知道少了哪些引擎。现在必须明确登记。
    """
    prose = (
        "This article examines how artificial intelligence reshapes graduate "
        "employability in China, and I chose it because I need this evidence "
        "for my second essay paragraph. The authors surveyed 148 graduates "
        "from a single university in Peru, which is a small sample for such "
        "a broad claim. Their findings suggest that AI skills help graduates "
        "find work related to their major rather than any job at all. I "
        "think this limitation matters for my topic, since Chinese employers "
        "may value different abilities than Peruvian graduates described. "
        "In my next essay I will therefore compare these results with job "
        "advertisements collected from mainland recruitment websites."
    )

    # 模拟「ML 检测器与伪困惑度均不可用」的环境
    monkeypatch.setattr(ac, "_detect_with_ml", lambda text: None)
    monkeypatch.setattr(ac, "_detect_mlm_perplexity", lambda text: None)
    ac._reset_gaps()

    result = ac.run_ai_detection(prose)
    names = {m["name"] for m in result.missing_engines}

    assert "ml-detector" in names, (
        f"ML 检测器不可用时必须登记进 missing_engines，实际缺失列表：{names}"
    )
    assert "mlm-perplexity" in names, (
        f"伪困惑度不可用时必须登记进 missing_engines，实际缺失列表：{names}"
    )
    # 每条缺失都必须带可读原因，不能只给名字
    for m in result.missing_engines:
        assert m.get("reason"), f"缺失引擎 {m.get('name')} 缺少原因说明"


# ---------------------------------------------------------------
# 5. 自检找茬（Strict）模式：严苛阈值 / 学生解读 / 高压预警 / 逐句策略
# ---------------------------------------------------------------

def test_student_verdict_thresholds():
    """学生向五级判定：阈值必须覆盖「安全/危险/致命」三档且单调不降。"""
    assert ac._student_verdict(0.10, "human") == "安全（你自己写的）"
    assert ac._student_verdict(0.40, "mixed") == "危险（会被老师怀疑用了 AI）"
    assert ac._student_verdict(0.70, "ai") == "致命（准备好重写吧）"


def test_high_pressure_warning_threshold():
    """高压预警只在 AI 率 >40% 时触发。"""
    assert ac._high_pressure_warning(0.41) != "", ">40% 应触发高压预警"
    assert ac._high_pressure_warning(0.40) == "", "<=40% 不应触发高压预警"


def test_strict_sentence_strategy_detects_advanced_word():
    """高级词汇句（delve 等）应给出「换大白话」的修改策略。"""
    s = "The authors delve into the multifaceted nature of employability."
    strat = ac._strict_sentence_strategy(s, s, {})
    assert "高级词汇" in strat and "delve" in strat, f"应命中高级词汇，实际：{strat}"


def test_strict_sentence_strategy_detects_template_starter():
    """四六级模板句应给出「模板句重写」策略。"""
    s = "In conclusion, this article is useful for my essay."
    strat = ac._strict_sentence_strategy(s, s, {})
    assert "四六级模板" in strat, f"应命中模板句，实际：{strat}"


def test_strict_mode_returns_mode_field():
    """AIResult 必须回传 mode 字段，供报告区分严苛/普通。"""
    prose = (
        "This article examines how artificial intelligence reshapes graduate "
        "employability in China, and I chose it because I need this evidence "
        "for my second essay paragraph. The authors surveyed 148 graduates "
        "from a single university in Peru, which is a small sample for such "
        "a broad claim. Their findings suggest that AI skills help graduates "
        "find work related to their major rather than any job at all. I "
        "think this limitation matters for my topic, since Chinese employers "
        "may value different abilities than Peruvian graduates described. "
        "In my next essay I will therefore compare these results with job "
        "advertisements collected from mainland recruitment websites."
    )
    import aurora_check as _ac
    # 不依赖模型：mock 掉 ML 检测器与伪困惑度，只验证 mode 字段与严苛阈值契约
    _orig_ml = _ac._detect_with_ml
    _orig_pp = _ac._detect_mlm_perplexity
    _ac._detect_with_ml = lambda text: None
    _ac._detect_mlm_perplexity = lambda text: None
    _ac.run_ai_detection.cache_clear()
    try:
        normal = _ac.run_ai_detection(prose, mode="normal")
        strict = _ac.run_ai_detection(prose, mode="strict")
    finally:
        _ac._detect_with_ml = _orig_ml
        _ac._detect_mlm_perplexity = _orig_pp
        _ac.run_ai_detection.cache_clear()

    assert normal.mode == "normal"
    assert strict.mode == "strict"
    # strict 是排雷视角：同样文本，strict 分数不应低于 normal（往高了打）
    assert strict.ai_probability >= normal.ai_probability, (
        f"strict({strict.ai_probability}) 不应低于 normal({normal.ai_probability})"
    )
    # strict 必须给出学生向解读，normal 保持不干扰
    assert strict.student_verdict != ""
    assert normal.student_verdict == ""


