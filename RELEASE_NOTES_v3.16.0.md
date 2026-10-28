# Aurora Check v3.16.0

> **A more honest AI-writing detector than Turnitin — it shows you *how* the score is computed, and *how* to fix it.**
> 一个比 Turnitin 更坦诚的 AI 写作自检工具——它告诉你分数是怎么算出来的，也告诉你怎么改。

Aurora Check is an **open-source, fully local** academic self-checking toolkit. It screens for AI-writing fingerprints, verifies references & DOI metadata, and performs document forensics — all on your own machine, no text ever leaves your device.

---

## ✨ Highlights

- **36 detection signals** — 29 purely statistical engines (zero model download required), plus a real ML classifier (ModernBERT), MLM pseudo-perplexity, a translationese detector, and adversarial robustness.
- **Calibrated on 52 real human papers** — 330k words scraped from Europe PMC + PLOS to absorb cross-document variance and stop the "invisible drift" that made earlier thresholds misjudge real academic prose.
- **Local LLM rewriting, on demand** — a bundled Qwen2.5-0.5b rewrites flagged sentences into a natural "student voice". Fully decoupled from detection (detection stays ~7s), with an `asyncio.Semaphore` queue so concurrent requests never thrash the CPU.
- **Turnitin-style Web UI** — offline ECharts gauge + engine radar + a "降重曲线" (de-AI progress curve) backed by SQLite history.
- **One-command deploy** — `docker compose up --build`, HuggingFace cache mounted so models download only once.

---

## 🚀 What's New in v3.16.0

| Area | Detail |
|---|---|
| 🔬 Real-corpus calibration | Re-fitted pseudo-perplexity pivot/scale on 52 real papers; fixed 2 engines that systematically misfired on academic prose |
| ✍️ On-demand LLM rewrite | `/api/rewrite` + `/api/rewrite-batch`, front-end queue + debounce, runaway-output guardrail (fallback to rules) |
| 🧱 Translationese detector | Catches DeepL-style machine-translation fingerprints (preposition-fronted clauses, long passives) |
| 🗄️ SQLite persistence | `/api/history` + ECharts trend line showing your de-AI progress |
| 🐳 Docker deploy | `python:3.10-slim` + CPU torch + HF cache volume + healthcheck |
| 🎨 UI polish | `word-break` for long sentences, real screenshots in README |

---

## 🧠 Architecture

```
text ──▶ 36 engines ──▶ weighted consensus ──▶ verdict (human / mixed / AI)
              │
              ├─ 29 statistical (burstiness, hedging, transition density, …)
              ├─ ML classifier (ModernBERT, swappable via AURORA_AI_MODEL)
              ├─ MLM pseudo-perplexity (ModernBERT-base, calibrated pivot=700)
              ├─ translationese detector (DeepL fingerprints)
              └─ adversarial perturbation detector

flagged sentences ──▶ on-demand local LLM rewrite (Qwen2.5-0.5b, semaphore-guarded)
```

Run modes: **normal** (稳健) and **strict** (严苛找茬 — deliberately over-flags so you catch everything before submission).

---

## 📦 Quick Start

```bash
docker compose up --build
# open http://localhost:8000
```

For a plain Python setup, see the README.

---

## ⚠️ Honest Limitations

- **Not a plagiarism checker.** There is no external corpus; the Crossref step is keyword-level comparison only and does **not** constitute a plagiarism finding.
- **AI score ≠ probability.** A high score means "looks AI-written", not "was AI-written".
- **Small calibration set.** 52 papers is a start, not a conclusion — expect some overfitting; re-calibrate per domain with `calibrate_corpus.py`.
- **For self-review only.** Not admissible as evidence of academic misconduct.

---

## 🙏 A Note from the Author

This started from a very personal panic: an essay that read **57% AI**. After weeks of rewriting by hand, I learned that the goal was never "0%" — the honest, human-sounding target for a first-year student was **around 34%**. "AI 率不是越低越好，34% 才是真实学生声线" — the de-AI curve is about sounding like *yourself*, not like a robot trying to hide.

Special thanks to **Doubao AI** for help with research and coding along the way.

---

### 🇨🇳 中文摘要

Aurora Check v3.16.0 是开源、全本地运行的学术写作自检工具：36 个检测信号（29 个纯统计引擎 + ML 分类器 + MLM 伪困惑度 + 翻译腔检测），基于 52 篇真实人类文献重新校准，内置本地 Qwen2.5 降维改写（按需触发 + 并发排队），提供 Turnitin 风格可视化（仪表盘 / 雷达图 / 降重曲线）+ SQLite 历史 + Docker 一键部署。它明确标注能力边界：不接外部语料库、不构成抄袭认定、AI 强度不等于概率、仅供自检复核。

**快速开始**：`docker compose up --build` → 打开 `http://localhost:8000`。