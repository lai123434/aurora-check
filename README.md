# Aurora Check · 一个比 Turnitin 更坦诚的 AI 查重工具

> **它告诉你分数是怎么算出来的，也告诉你怎么改。**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Engines](https://img.shields.io/badge/Engines-36-orange.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)

Aurora Check 是一个开源、本地运行的**学术写作自检工具**：AI 写作痕迹筛查 + Crossref 关键词级文献比对 + DOI/引用核验 + 一键「降维改写」。**36 个检测信号** + ModernBERT ML 分类器 + MLM 伪困惑度 + **52 篇真实人类文献校准**，全程本地运行、结果可复现，把每个分数怎么算出来的摊开给你看——而不是像商业产品那样只丢一个百分比。

---

## ⚠️ 使用前必读：这是什么、不是什么

> **一句话：这是辅助筛查原型，不是查重系统，更不能作为学术不端判定依据。**

| | 本工具能做 | 本工具**不能**做 |
|---|---|---|
| **AI 写作痕迹** | ✅ 给出多引擎集成的**风险强度**与可复现证据链 | ❌ 不能判定"这是 AI 写的"——见下方说明 |
| **外部查重** | ✅（可选联网）把长句丢进 Crossref 学术库做**关键词级**比对，返回疑似出处（标题 + DOI） | ❌ 不是逐字文本比对，**不构成抄袭认定**；且默认关闭（`external_similarity` 恒为 `None` = "没查"，不是"命中 0%"） |
| **内部自重叠** | ✅ 检测同一文档内部段落间的重复 | ❌ 不等于查重 |
| **文献核验** | ✅ DOI 注册状态、期刊画像（h-index / 年发文异常 / 被引百分位）、APA 引用规范检查 | ❌ 不做全文相似度、不做撤稿以外的出版伦理判定 |

### 🚫 三条硬性限制（请务必理解）

1. **AI 检测强度 ≠ 发生概率。** 代码内置的 `_positive_predictive_value()` 承认：在低基准率场景下，**被判为 AI 的文本里约有六成其实可能是人写的**。它只能提示"值得人工复核"，不能作为指控、扣分或处分的依据。
2. **没有外部语料库。** 名字里的"查重"是历史遗留。它不做抄袭比对——要查重请用 Turnitin/iThenticate 等有语料库授权的系统。
3. **校准样本很小。** 关键阈值基于 5 篇 AI + 1 篇真人英文学术语料校准，存在过拟合风险；换领域（中文/新闻/代码）必须用自己的语料重新校准。

**任何学术不端认定都必须走"人工复核 + 申诉机制 + 校准报告"流程；本工具至多提供其中的筛查线索。**

---

## 📖 Overview

Aurora Check is an open-source **research prototype** for screening AI-writing features and verifying academic references. It is designed to give reviewers *reproducible evidence* (which engine said what, and why) rather than a verdict.

Unlike commercial detectors (Turnitin, GPTZero, Originality.ai), Aurora Check is fully open-source, free, runs locally (optional online models), and shows its math — every score comes with the contribution of each engine.

**Not a plagiarism checker.** External similarity is `None` / NOT IMPLEMENTED by design.

### ✨ Key Features

- 🤖 **36 个检测信号**（英文下约 24 个活跃加权，含 6 个自研引擎 + 翻译腔检测 + 中文专用引擎） - Statistical features + ML models + zero-shot paradigms
- 🧠 **ML classifier (ModernBERT by default)** - A real deep learning model, not just simple statistical rules; configurable via `AURORA_AI_MODEL`
- 📚 **52 篇真实文献校准** - 伪困惑度与统计引擎阈值用真实人类学术文献重拟合，消掉跨文献「不明显跳变」
- 🆕 **6 novel innovation engines** - Human imperfection detector, emotion curve detector, specific details detector, self-reference detector, logical connectors detector, hidden watermark detector
- 🔬 **Zero-shot detection paradigms** - PAWN perplexity weighting, SpecDetect signal processing, DNA-DetectLLM repair-based detection
- 🛡️ **Adversarial robustness** - Adversarial perturbation detector to resist paraphrasing attacks
- 🌐 **Multi-language support** - Chinese AI detection engine
- 🔍 **Statistical watermark detection** - Detects green/red-list statistical watermarks (not just zero-width characters)
- 🌏 **Translationese detector** - 专治 DeepL 机翻洗稿（介词前置从句 + 冗长被动语态）
- ✍️ **真·AI 降维改写（按需触发）** - 高危句旁「🤖 帮我改写这句」按钮，本地 Qwen2.5 生成「90 分大一新生」口吻改写；查重流程保持秒级返回
- 📈 **降重曲线** - SQLite 记录每次查重，前端 ECharts 折线图看 AI 率从高危降到安全
- 📊 **Engine disagreement analysis** - Shows how much engines agree/disagree, identifies confident vs conflicting engines
- 📈 **Sentence-level uncertainty** - Visual confidence levels (high/medium/low) for each high-risk sentence
- 🎨 **Beautiful HTML report** - Turnitin-style, with highlighted AI-suspect sentences and confidence indicators
- 🔍 **Local document comparison** - Compare similarity between two documents
- 📚 **Automatic citation exclusion** - Automatically detects and excludes reference sections
- 🎓 **Academic-friendly** - Optimized specifically for academic papers
- 🆓 **Completely free** - MIT license, no restrictions

---

## 📸 Screenshots

启动后在浏览器打开 **http://localhost:8000**，一次检测即可同时看到三块可视化：

| 检测仪表盘 | 引擎雷达图 | 查重历史 · 降重曲线 |
|---|---|---|
| ![仪表盘](docs/screenshot-dashboard.png) | ![雷达图](docs/screenshot-radar.png) | ![降重曲线](docs/screenshot-trend.png) |

> 截图为真实运行画面：strict 模式检测一段典型 AI 腔文本（仪表盘 99%、引擎雷达图），
> 降重曲线展示一稿一周内从 91% 降到安全区的完整过程——10-28 上午的 99% 尖峰是
> 提交前用 strict 严苛模式做的最后一次复查。

---

## 🚀 Quick Start

### Docker 一键部署（推荐，给没有 Python 环境的同学）

```bash
docker compose up --build
```

完成后浏览器打开 **http://localhost:8000** 即可粘贴文本检测。首次启动会自动拉取镜像、安装依赖，
并下载约 2GB 的检测模型到本机 HuggingFace 缓存（`~/.cache/huggingface`，已挂载进容器，之后重启/重建不重复下载）。

> 国内网络下载模型慢：编辑 `docker-compose.yml`，取消 `HF_ENDPOINT: https://hf-mirror.com`
> 这行注释走镜像加速。模型缓存在宿主机，删容器不丢。

### Installation

```bash
# Clone the repository
git clone https://github.com/lai123434/aurora-check.git
cd aurora-check

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Detect AI-generated content in a text file
python aurora_check.py examples/sample_essay.txt

# Automatically generates both Markdown and HTML reports
# HTML report is saved in the same directory
```

### 可选：困惑度模型（强烈推荐，中文）

```bash
python download_model.py     # 约 950MB，从 ModelScope 下载
```

装好后，检测会多一个「语言模型困惑度」信号——它衡量这段文字对语言模型来说
有多"顺"。这是所有信号里对改写最鲁棒的一项：改写能改掉用词和句式，
但改不掉"这个词后面该接什么"的概率分布。

实测（Qwen2.5-0.5B，中文）：

```text
标准 AI 输出      困惑度  6.3 ~ 13.0
伪装成人类的 AI            14.4 ~ 59.7
真实人类                   22.9 ~ 116.5
```

它只作**单边证据**：困惑度低才构成"像 AI"的证据；困惑度高只能说明
"不像模型写的"，不能反证是人写的——人写得好一样很顺。所以超过阈值直接不参与。

不装也能用，只是少了这一项。模型约 944MB，不进仓库。

### Web UI（可视化首页）

```bash
pip install -r requirements-web.txt
python server.py         # → http://127.0.0.1:8000
```

在网页输入框粘贴正文，切换 **normal（稳健）/ strict（严苛找茬）** 模式即可测。
报告为交互式可视化：ECharts 仪表盘 + 引擎雷达图 + 高危句红色下划线（悬停看"怎么改"）。
strict 模式还会给每条高危句附上即时「怎么改」策略 + Mock 规则版改写建议
（高级词换大白话 + 长句拆短句，瞬时给出、不拖慢检测）。每条高危句旁还有一个
「🤖 帮我改写这句」按钮——点击后按需调用本地 LLM 生成更自然的降维改写，
查重流程本身保持秒级返回（改写是用户主动发起的独立请求，不影响检测速度）。
**并发安全**：后端全局信号量 + 前端队列双保险，一次点多个按钮会排队串行处理，
绝不并发打满 CPU 导致 504；顶部还有「🤖 一键改写前 3 句」总按钮走批量接口，
一次拿 3 条、约 60 秒。
ECharts 已本地化内联，导出的独立 HTML 断网也能完整渲染。
全部在本机运行，文档不出设备。

### Web API（FastAPI）

把检测能力暴露成 HTTP 服务，供网页后端、小程序、其他 AI（MCP）调用：

```bash
python server.py                    # 或 uvicorn server:app --reload
```

```bash
# 提交一段文本，返回完整 JSON 报告
curl -X POST http://127.0.0.1:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"text": "你的文章内容...", "mode": "strict"}'

# 上传文件检测（mode 走 query 参数）
curl -X POST "http://127.0.0.1:8000/api/check-file?mode=strict" \
  -F "file=@paper.docx"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 可视化首页 |
| `/api/check` | POST | `{"text", "mode"}` → `CheckReport.to_dict()` 完整 JSON |
| `/api/check-file` | POST | 上传 `.txt/.md/.docx`，`?mode=strict` |
| `/api/rewrite` | POST | `{"sentence"}` → 按需 AI 降维改写（本地 LLM，无模型回退规则版；全局信号量排队） |
| `/api/rewrite-batch` | POST | `{"sentences": [...]}` → 一键批量改写（最多 3 句，内部串行） |
| `/api/history` | GET | 最近 10 条查重记录（前端画降重曲线） |
| `/api/health` | GET | 健康检查 |

> 大模型跑在 CPU 上较慢，接口内部用 `asyncio.to_thread` 丢进线程池，
> 不阻塞事件循环；`run_check` 的 ContextVar + LRU 缓存跨线程隔离、线程安全。

### Benchmark

```bash
python benchmark.py
```

在 `examples/benchmark/` 上输出逐样本得分、人写/AI 区间、AUC 与判定准确率。
文件名前缀即标签（`human_*` / `ai_*` / `paraphrased_*`），加自己的样本即可扩充。

### 本地双文档比对（你自己提供参考文档）

```bash
# 比对两份文档之间的相似度
python aurora_check.py compare document1.txt document2.txt
```

> **能力边界**：这是**你自己提供的两份本地文档**之间的相似度比对，是真实可用的功能。
> 但它**不等于查重**——本工具不会去检索外部数据库或互联网语料。
> 要做真正的查重，需自行把候选文献作为「参考文档」喂进来，或改用有语料库授权的系统。

---

## 🧠 Detection Engines Explained

Aurora Check 内置 **36 个检测信号**（英文下约 24 个活跃加权 + 2 个中文专用 + 若干已归零的历史引擎），
外加一个可选的 ML 分类器作为主力锚点。核心引擎分两类：

> **关键：阈值不再靠「拍脑袋」。** 3.13 起，伪困惑度与统计引擎的权重都用
> **52 篇真实人类英文学术文献**（Europe PMC + PLOS，约 33 万词）重校准过，
> 消掉了「换一篇文献，判定就跳来跳去」的过拟合。校准参数在 `corpus/calibration.json`，
> 检测时自动加载；想用自己的语料重新标定，跑 `python collect_corpus.py` + `python calibrate_corpus.py` 即可。

### Category 1: Machine Learning Models (Highest Weight)

| Engine | Description | Source |
|--------|-------------|--------|
| **ML AI Detector (ensemble)** | Default: ModernBERT (2024, trained on MAGE+RAID) as **primary**, plus `Hello-SimpleAI/chatgpt-detector-roberta` as an **independent second opinion** for cross-validation (the two differ in architecture/training data and cover each other's blind spots; the legacy model is deliberately *not* the primary). Customize the whole chain with `AURORA_AI_MODEL="a|b|c"`. | GeorgeDrayson/modernbert-ai-detection-raid-mage |
| **MLM Pseudo-Perplexity** | 专攻「表演式不完美」拟人 AI：表层故意粗糙，但底层统计上仍过度可预测。伪困惑度分离度大：**AI ≈ 152–451 vs 人类 ≈ 9200**。**52 篇真实文献校准后**，敏感区从 1650 下移到 **700**（真实文献误判率从 13% 降到 2%）。可用 `AURORA_PPL_PIVOT` / `AURORA_PPL_SCALE` 按领域重标，换 backbone 用 `AURORA_PPL_MODEL`。 | answerdotai/ModernBERT-base |

### Category 2: Statistical Feature Engines (Reverse-Engineered from Multiple Open-Source Projects)

| Engine | Description | Source |
|--------|-------------|--------|
| **lmscan** | Local statistical feature detection (Zipf deviation, slop word density, burstiness, etc.) | stef41/lmscan |
| **formulaic-markers** | AI common phrase detection (delve, nuanced, testament to, etc.) | lynote-ai/ai-text-detector |
| **transition-density** | Transition word density detection (first, moreover, however, etc.) | lynote-ai/ai-text-detector |
| **compressibility** | Compression ratio detection (AI text is more compressible) | lynote-ai/ai-text-detector |
| **structured-shape** | List/heading structure detection | lynote-ai/ai-text-detector |
| **formulaic-patterns** | AI opening/closing template detection | pablocaeg/sloptotal |
| **linguistic-markers** | Weighted AI vocabulary detection (100+ marker words, 3 tiers) | pablocaeg/sloptotal |
| **vocabulary-richness** | Vocabulary richness detection (Root TTR, Hapax ratio, Yule's K) | pablocaeg/sloptotal |
| **structural-patterns** | Structural feature detection (colon density, list density, paragraph uniformity) | pablocaeg/sloptotal |
| **sentiment-hedging** | Sentiment and hedging detection (over-hedging, vague positivity, filler qualifiers) | pablocaeg/sloptotal |
| **human-imperfection** | **Novel!** Human imperfection detector: the more "perfect" a text is, the more likely it's AI | **Custom innovation** |
| **emotion-curve** | **Novel!** Emotion curve detector: AI emotion is flat, humans have ups and downs | **Custom innovation** |
| **specific-details** | **Novel!** Specific details detector: humans have many concrete numbers/names, AI is abstract | **Custom innovation** |
| **self-references** | **Novel!** Self-reference detector: humans say "I think", AI rarely does | **Custom innovation** |
| **logical-connectors** | **Novel!** Logical connectors detector: AI uses many "therefore/however", humans use less | **Custom innovation** |
| **ngram-perplexity** | N-gram language model perplexity | Custom |
| **sliding-window** | Paragraph-level sliding window detection | Custom |
| **burstiness-stat** | Traditional sentence length burstiness detection | Custom |
| **statistical-baseline** | Statistical baseline detection | Custom |

---

## 🌐 联网与镜像

`ml-detector` 引擎默认从 HuggingFace 拉取 ModernBERT 检测器，**需要联网**。其余 30 个统计引擎完全离线可用，断网时引擎会自动跳过 ML 项并照常出分。

- **自动镜像回退**：若直连 `huggingface.co` 不通（国内/受限网络常见），引擎会**自动切换**到 `hf-mirror.com` 镜像完成下载，无需手动配置。仅在镜像也不可达时才跳过 ML 引擎。
- **手动指定镜像 / 自定义模型**：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com   # 强制走镜像
  export AURORA_AI_MODEL=你的模型ID          # 覆盖默认 ModernBERT（| 分隔可写降级链）
  ```
- **本地缓存复用**：模型首次下载后缓存在 `~/.cache/huggingface`，之后离线也能直接加载。加载器**缓存优先**（先试本地、跳过 hub 的 etag 校验），因此模型一旦下载过，断网也能正常跑；只有需要新下载时才做网络探测。
- **缺失引擎透明化**：任何引擎未能产出分数（连不上模型、未装 `lmscan` CLI、`lm-perplexity` 仅中文启用等），都会在报告的「**本次未参与检测的引擎**」段落明确列出名称与原因，**不再静默跳过**——你永远知道这次到底跑了哪些引擎、缺了哪些。
- **多检测器交叉验证**：为对抗单一分类器的风格盲区，可以一次性挂多个模型：
  ```bash
  export AURORA_AI_MODEL="GeorgeDrayson/modernbert-ai-detection-raid-mage|Hello-SimpleAI/chatgpt-detector-roberta"
  ```
  链中**全部**模型都会被加载：第一个有效结果作**主力锚点**参与 50% 加权，其余作**第二意见**——仅当主力可能处于盲区（≤0.45）而另有检测器给出高危分（≥0.60）时才把结论拉回「需人工复核」，避免自信误判。
- **实测表现（拟人 AI 压力测试，详见 CHANGELOG 3.3.0）**：纯 AI 生成的拟人化文本，升级前 **5 篇里 4 篇被自信判为 human（完全漏判）**；升级后明显 AI 仍抓 `ai`，其余拟人 AI **全部进入 `mixed` 待人工复核**，不再有自信 false-human。
- **⚠️ 召回优先的取舍**：为消除漏判，真人对照稿同样会被抬到 `mixed` 区。若你的场景更在意「别误伤真人」，调低 `ml-detector` / `mlm-perplexity` 权重即可；默认按「宁可复核、不可放过」取值。
- **实测表现**（ModernBERT，`hf-mirror.com`）：明显 AI 腔文本 ML 分 ≈ **0.99**，自然人类腔文本 ≈ **0.45**，区分度强。

## 📊 Output Example

### Markdown Report

```
# Aurora Check Report

- **Text Length:** 1200 characters

## AI Detection

- **Overall Verdict:** Likely human-written
- **AI Probability:** 12.5%
- **Human Probability:** 87.5%
- **Method:** ensemble (15 engines, references excluded)

- **Individual Engine Scores:**
  - ml-detector: 8.2%
  - lmscan: 6.7%
  - formulaic-patterns: 0.0%
  - ...

- **High-Risk Sentences (Top 5):**
  - (0.65) This is a sample sentence...
  - ...
```

### HTML Report

The generated HTML report features:

- 🎨 Gradient purple-blue header
- 📊 Large AI score display (red/yellow/green color coding)
- 📈 Detailed score bars for each engine
- 🔴 Highlighted high-risk sentences
- 📱 Responsive design, works on mobile

---

## 🔧 Advanced Usage

### Python API

```python
from aurora_check import run_check

# Detect AI content in text（mode="normal" 稳健 / "strict" 严苛找茬）
report = run_check(text="Your text here...", mode="strict")

# Get AI probability
ai_prob = report.ai.ai_probability

# Get high-risk sentences（strict 模式每条带 strategy「怎么改」+ rewrite_suggestion「直接抄的改写」）
flagged = report.ai.flagged_sentences

# Generate interactive HTML report（ECharts 雷达图 + 仪表盘 + 高危句气泡）
html = report.to_html()

# Export as dict for JSON API
data = report.to_dict()
```

### Adjusting Engine Weights

You can adjust the weight of each engine based on your needs. Find the `weights` dictionary in `aurora_check.py`:

```python
weights = {
    "ml-detector": 0.25,          # ML 模型权重（默认 ModernBERT，可配置）
    "lmscan": 0.20,                # lmscan weight
    "formulaic-patterns": 0.12,   # Formulaic patterns weight
    # ... other engines
}
```

---

## 📋 Tier Positioning

| Tier | Representative Software | Features |
|------|------------------------|----------|
| **Tier 1** | Turnitin, GPTZero, Originality.ai | Multiple pre-trained models + full-web comparison + commercial support |
| **Tier 2** | **Aurora Check** | 29-engine ensemble + ModernBERT ML model + local comparison |
| **Tier 3** | Simple single-engine detectors | Only one or two statistical rules |

---

## 📊 Measured on this build

### English（`python benchmark.py`，随仓库分发）

```text
人写区间     0.163 ~ 0.284
AI 区间      0.456 ~ 0.781
间隔         +0.172
AUC          1.000  (n=9，样本量小，仅作回归基线)
```

### 中文（本地实测，语料因版权不随仓库分发）

采集 24 篇真实中文文章（少数派 / 36氪 / 简书）+ 8 篇生成式模型输出：

```text
                 均值     区间              检出(≥0.5)
标准 AI 输出     0.667    0.585 ~ 0.775     8/8
伪装成人类的 AI  0.611    0.494 ~ 0.672     3/4
真实人类         0.249    0.129 ~ 0.375     0/24（零误判）
```

第三组是**对抗测试**：刻意加口语、编具体数字、打乱段落，模仿人类笔法。
它测出了统计检测器的天花板——只靠表层特征，始终可以被改写绕过。

对抗中唯一稳住的是 `zh-char-entropy`（字符级 3-gram 熵）：
AI 文本 8.4~8.7 比特，人类 10.5 比特。加几句口语改变不了整篇的用字分布，
这一项把对抗样本的检出率从 50% 拉到 75%。

中文不是英文引擎的副产品：英文词表（delve / furthermore）对中文恒为 0，
必须单独建模。当前中文引擎依赖六组信号——模板连接词、公文抽象词、
口语标记、具体数字、句长变异、引号密度——并会在检测到中文时自动
把英文词表类引擎降到 20% 权重。

## ⚠️ Known Limitations

Measured on 6 texts of known origin (see `test_regression.py`):

| Issue | Detail |
|---|---|
| **Paraphrased AI text is not detected** | Rewriting destroys sentence rhythm, template and emotion signals. Total score drops to the human band. Use the `rewrite_hint` field instead of the score. |
| **Non-native academic writing skews "human"** | Engines such as `formulaic-markers` use native-English marker lists; ESL writing scores 0 on them. |
| **Short texts are unreliable** | Below ~200 characters the tool refuses to output a score (`note` field). |
| **Absolute scores are not probabilities** | Do not read "43%" as "43% chance of AI". Only relative comparison between texts is meaningful. |

Regardless of score, **defer to human judgement** for academic-integrity decisions.

## 🤝 Contributing

We welcome any form of contribution!

### How to Contribute

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Want to Add a New Engine?

If you want to add a new detection engine:

1. Add a new detection function in `aurora_check.py`
2. Register the new engine in `run_ai_detection`
3. Adjust the weight in the `weights` dictionary
4. Test to ensure existing functionality is not broken

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Open Source Contributors

This project reverse-engineered and integrated core algorithms from the following open-source projects. We thank the authors of these projects:

- [lynote-ai/ai-text-detector](https://github.com/lynote-ai/ai-text-detector) - 6-signal detection algorithm
- [pablocaeg/sloptotal](https://github.com/pablocaeg/sloptotal) - 24-engine ensemble approach
- [stef41/lmscan](https://github.com/stef41/lmscan) - Local statistical feature detection
- [DW-Studio/sloplint](https://github.com/DW-Studio/sloplint) - 中文 AI 味特征清单（283 万字对照语料）
- [GeorgeDrayson/modernbert-ai-detection-raid-mage](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage) - 默认 ML 检测器（ModernBERT，MAGE+RAID 训练，泛化到现代 LLM）
- [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) - 降级用旧 RoBERTa ChatGPT 检测模型
- [followsci/bert-ai-text-detector](https://huggingface.co/followsci/bert-ai-text-detector) - BERT academic text detection model

### 关于 AI 率：一段真实的踩坑记录

这个项目是作者亲手「降维改稿」时踩坑踩出来的。同一篇作业，AI 率一路从 **57% → 19.6% → 34%**：

- **57%**：初稿直接用 AI 生成，AI 特征拉满；
- **19.6%**：疯狂删高级词、塞口语、压模板，AI 率掉到 19.6%——但老师一眼看出「不像人写的」，因为过度降维反而失真；
- **34%**：找回真实学生声线——保留少量小错误、保留 Chinglish、保留个人细节，AI 率稳定在 34% 上下。

**结论：AI 率不是越低越好。** 一个高考英语 90 分的大一新生，正常写出来的作业本身就带一点「人味瑕疵」，对应到检测器就是 30% 上下的 mixed 区。硬把 AI 率压到个位数，反而暴露了你「改过头了」。**34% 才是真实学生声线。**

### Special Thanks

> This project was developed with assistance from **Doubao AI (豆包)**. Special thanks to Doubao for invaluable help in research, coding, and project development.

---

## ⚠️ Disclaimer

Aurora Check's detection results are for reference only and cannot serve as the final verdict on academic integrity. AI detection technology is still rapidly evolving, and no tool can achieve 100% accuracy.

Always defer to human judgment.

---

## 📞 Contact

If you have any questions or suggestions, feel free to:

- Submit an Issue
- Send a Pull Request
- Or contact the project maintainer directly

---

**⭐ If this project helps you, please give it a Star!**
