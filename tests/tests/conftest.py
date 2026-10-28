# -*- coding: utf-8 -*-
"""pytest 公共夹具。

设计要点：
1. **默认快**：深度学习/web 相关的重活标记 slow，需 `--runslow` 才跑。
   避免没装模型的机器上一片红，也避免每次改动都跑十几分钟。
2. **缺失即跳过、不伪装通过**：模型/真实语料不可用时 skip 并写明原因，
   绝不改成 assert True 蒙混。
3. **真人语料不入库**：真实学生作业含个人信息，不复制进仓库；
   通过环境变量 AURORA_HUMAN_SAMPLE 指定，缺失则跳过。
"""
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import aurora_check as ac  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="运行依赖深度学习模型（耗时数分钟）的测试",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --runslow 才会运行（依赖 ML 模型，耗时）")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------
# 语料
# ---------------------------------------------------------------

@pytest.fixture(scope="session")
def ai_samples():
    """5 篇纯 AI 生成的合成文本（内联、无隐私风险）。"""
    import evasion_suite
    return dict(evasion_suite.SAMPLES)


@pytest.fixture(scope="session")
def human_sample_path():
    """真实学生作业语料路径。

    不入库（隐私），按以下顺序查找；找不到则跳过相关测试。
    """
    candidates = [
        os.environ.get("AURORA_HUMAN_SAMPLE", ""),
        "/workspace/BUS100_A1_Annotated_Bibliography_Final.docx",
        os.path.join(BASE, "fixtures", "human_sample.docx"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    pytest.skip(
        "找不到真人对照语料。可用 AURORA_HUMAN_SAMPLE=/path/to/file.docx 指定，"
        "或放到 tests/fixtures/human_sample.docx（真实学生作业请勿提交公开仓库）。"
    )


# ---------------------------------------------------------------
# 深度模型可用性（会话级只探测一次）
# ---------------------------------------------------------------

def _models_ready() -> bool:
    """三个模型（ModernBERT 分类器 ×2 + ModernBERT-base MLM）是否都可加载。"""
    try:
        cls = ac._load_ml_models()
        m, t = ac._load_mlm_model()
    except Exception:
        return False
    return bool(cls) and m is not None and t is not None


@pytest.fixture(scope="session")
def require_models():
    if not _models_ready():
        pytest.skip(
            "深度学习模型不可用（未下载或离线）。"
            "快照依赖完整三模型，缺少会导致分数不可比，故跳过而非放行。"
        )
    return True
