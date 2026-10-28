# -*- coding: utf-8 -*-
"""联网实测 AI 检测：观察 ModernBERT 能否下载、是否参与加权。"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import aurora_check as ac

def probe(name, path):
    print(f"\n========== 测试样本：{name} ({path}) ==========")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    t0 = time.time()
    # 先用 verbose 看模型加载过程（stderr 会打印）
    report = ac.run_check(text=text)
    dt = time.time() - t0
    ai = report.ai
    print(f"耗时 {dt:.1f}s | 正文长度 {report.text_length}")
    if ai is None:
        print("ai 结果为 None（输入不合格）")
        return
    print(f"综合判定: {ac._label_to_cn(ai.label)} | AI特征强度: {ai.ai_probability:.1%}")
    print(f"置信度: {ai.confidence} | 方法: {ai.method}")
    ml = ai.engine_scores.get("ml-detector")
    print(f"ml-detector(ModernBERT) 引擎分: {ml}")
    print(f"参与加权引擎数: {len(ai.adopted_engines)} | 引擎分项数: {len(ai.engine_scores)}")
    if ai.rejected_engines:
        print("被弃用引擎:", [(e['name'], e['reason']) for e in ai.rejected_engines])

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    # 先确认 ML 模型能否加载（verbose 打印细节）
    print(">>> 尝试加载 ML 检测器（默认 ModernBERT 降级链）...")
    os.environ["AURORA_VERBOSE"] = "1"
    mdl, tok, idx = ac._load_ml_model()
    if mdl is None:
        print("!!! ML 检测器未能加载（离线/下载失败）— 仅统计引擎可用")
    else:
        print(f"!!! ML 检测器已加载 ai_idx={idx}")
    print("-------------------------------------------------")
    probe("AI腔样本", os.path.join(here, "test_ai.txt"))
    probe("人类腔样本", os.path.join(here, "test_human.txt"))
