# -*- coding: utf-8 -*-
"""用真实人类学术文献语料，重测并校准查重模型的原始特征分布。

两件事：
  1) 伪困惑度（mlm-perplexity）原始值分布 —— 用于重拟合 PIVOT/SCALE
     （现状基于 6 篇小样本，换到真实发表文献域会系统性漂移）
  2) 各统计引擎在「真人文献」上的得分分布 —— 用于识别哪些引擎会误触发
     （真人文献却给高 AI 分的引擎，权重应下调或阈值上修）

产出：
  corpus/calib_raw.json        每篇的 raw_pp + 各引擎原始得分
  corpus/calib_summary.json    分布摘要（均值/中位/分位数）
"""
import json
import glob
import os
import statistics

BASE = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE, "corpus", "text")

import aurora_check as ac

# 纯统计引擎（全部离线，无需 ML 检测器联网下载）
# 每个引擎拿「原始得分」，方向 +1 = 分数高越像 AI
ENGINES = {
    "transition-density":    ac._detect_transition_density,
    "linguistic-markers":    ac._detect_linguistic_markers,
    "formulaic-patterns":    ac._detect_formulaic,
    "formulaic-markers":     ac._detect_formulaic_markers,
    "sentiment-hedging":     ac._detect_sentiment_hedging,
    "sentence-length":       ac._detect_sentence_length_distribution,
    "burstiness-stat":       ac._detect_perplexity_burstiness,
    "vocabulary-richness":   ac._detect_vocabulary_richness,
    "starter-patterns":      ac._detect_starter_patterns,
    "specific-details":      ac._detect_specific_details,
    "compressibility":       ac._detect_compressibility,
    "human-imperfection":    ac._detect_human_imperfection,
    "paragraph-rhythm":      ac._detect_paragraph_rhythm,
    "emotion-curve":         ac._detect_emotion_curve,
    "ngram-perplexity":      lambda t: ac._compute_perplexity_ngram(t, n=3),
    "statistical-baseline":  lambda t: ac._detect_with_baseline(t)[0],
}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main():
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    print(f"加载 {len(files)} 篇真实文献，开始计算特征（伪困惑度较慢，约 2 分钟）...")

    rows = []
    pp_raws = []
    engine_collect = {k: [] for k in ENGINES}

    for i, fp in enumerate(files, 1):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        fid = os.path.basename(fp)[:-4]
        prose, _ = ac._extract_prose_only(text)

        # 伪困惑度原始值（用引擎同一套逻辑）
        pp_raw = None
        try:
            res = ac._detect_mlm_perplexity(prose)
            if res is not None:
                pp_raw = res[1]
                pp_raws.append(pp_raw)
        except Exception as e:
            print(f"  ✗ {fid} pp 失败: {e}")

        row = {"id": fid, "words": len(prose.split()), "pp_raw": pp_raw}
        for name, fn in ENGINES.items():
            try:
                v = fn(prose)
                if v is None:
                    v = 0.5
                v = float(v)
            except Exception:
                v = 0.5
            row[name] = round(v, 4)
            engine_collect[name].append(v)

        rows.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(files)} 完成")

    summary = {
        "n": len(rows),
        "pp_raw": {
            "mean": round(statistics.mean(pp_raws), 2) if pp_raws else None,
            "median": round(statistics.median(pp_raws), 2) if pp_raws else None,
            "min": round(min(pp_raws), 2) if pp_raws else None,
            "max": round(max(pp_raws), 2) if pp_raws else None,
            "p10": round(pct(pp_raws, 0.10), 2) if pp_raws else None,
            "p25": round(pct(pp_raws, 0.25), 2) if pp_raws else None,
            "p50": round(pct(pp_raws, 0.50), 2) if pp_raws else None,
            "p75": round(pct(pp_raws, 0.75), 2) if pp_raws else None,
            "p90": round(pct(pp_raws, 0.90), 2) if pp_raws else None,
        },
        "engines": {},
    }
    for name, vals in engine_collect.items():
        summary["engines"][name] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p75": round(pct(vals, 0.75), 4),
            "p90": round(pct(vals, 0.90), 4),
            "max": round(max(vals), 4),
        }

    with open(os.path.join(BASE, "corpus", "calib_raw.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, "corpus", "calib_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n== 伪困惑度 raw_pp 分布（真人学术文献）==")
    for k, v in summary["pp_raw"].items():
        print(f"  {k:<8} {v}")
    print("\n== 各统计引擎在真人文献上的得分分布 ==")
    print(f"  {'引擎':<22}{'均值':>8}{'中位':>8}{'p75':>8}{'p90':>8}{'max':>8}")
    for name, d in summary["engines"].items():
        print(f"  {name:<22}{d['mean']:>8.3f}{d['median']:>8.3f}{d['p75']:>8.3f}"
              f"{d['p90']:>8.3f}{d['max']:>8.3f}")
    print(f"\n完成，结果已写 corpus/calib_raw.json 与 corpus/calib_summary.json")


if __name__ == "__main__":
    main()
