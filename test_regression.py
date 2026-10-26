#!/usr/bin/env python3
"""
Aurora Check 回归测试脚本
确保每次调整引擎时不会误伤人类文本
"""
import sys
import os

# 导入主模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurora_check import run_ai_detection

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "test_set")

def run_test(name, filename, expected_min, expected_max):
    """运行单个测试"""
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.exists(filepath):
        print(f"❌ {name}: 文件不存在 - {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    result = run_ai_detection(text)
    ai_prob = result.ai_probability

    status = "✅" if expected_min <= ai_prob <= expected_max else "❌"
    print(f"{status} {name}: AI概率={ai_prob:.1%} (期望范围: {expected_min:.0%}-{expected_max:.0%})")

    return expected_min <= ai_prob <= expected_max


def main():
    print("=" * 60)
    print("Aurora Check 回归测试")
    print("=" * 60)
    print()

    tests = [
        # (名称, 文件名, 最小AI概率, 最大AI概率)
        ("人类样本1", "human_sample_1.txt", 0.0, 0.35),
        ("AI样本1", "ai_sample_1.txt", 0.4, 1.0),
        ("改写样本1", "paraphrased_sample_1.txt", 0.3, 0.8),
    ]

    passed = 0
    failed = 0

    for name, filename, min_prob, max_prob in tests:
        if run_test(name, filename, min_prob, max_prob):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
