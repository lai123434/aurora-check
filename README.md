# 极光查重 (Aurora Check)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Engines](https://img.shields.io/badge/Engines-15-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**一个开源的 AI 文本检测与查重工具，集成 15 个检测引擎，包括真正的 RoBERTa 机器学习分类器。**

---

## 📖 项目简介

极光查重 (Aurora Check) 是一个开源的 AI 生成文本检测工具，旨在帮助学生、教师和研究人员判断一篇文本是否由 AI 生成。

与市面上的商业软件（如 Turnitin、GPTZero、Originality.ai）不同，极光查重完全开源、免费、离线运行（可选在线模型），任何人都可以查看、修改和分发代码。

### ✨ 核心特性

- 🤖 **15 个检测引擎集成** - 统计特征 + 机器学习模型
- 🧠 **RoBERTa ML 分类器** - 真正的深度学习模型，不是简单的统计规则
- 📊 **多维度评分** - 从词汇、句法、结构、情感等多个角度分析
- 📄 **漂亮的 HTML 报告** - Turnitin 风格，高亮显示 AI 嫌疑句
- 🔍 **本地文档比对** - 比对两篇文档的相似度
- 📚 **自动引用排除** - 自动识别并排除参考文献部分
- 🎓 **学术友好** - 专门针对学术论文优化
- 🆓 **完全免费** - MIT 许可证，无任何限制

---

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-username/aurora-check.git
cd aurora-check

# 安装依赖
pip install -r requirements.txt
```

### 基本用法

```bash
# 检测一篇文本
python aurora_check.py examples/sample_essay.txt

# 自动生成 Markdown 报告和 HTML 报告
# HTML 报告会保存在同目录下
```

### 文档比对（查重）

```bash
# 比对两篇文档的相似度
python aurora_check.py compare document1.txt document2.txt
```

---

## 🧠 检测引擎详解

极光查重集成了 **15 个检测引擎**，分为两大类：

### 第一类：机器学习模型（权重最高）

| 引擎 | 说明 | 来源 |
|------|------|------|
| **RoBERTa ChatGPT Detector** | 专门训练来检测 ChatGPT 生成文本的 RoBERTa 模型 | Hello-SimpleAI/chatgpt-detector-roberta |

### 第二类：统计特征引擎（从多个开源项目逆向集成）

| 引擎 | 说明 | 来源 |
|------|------|------|
| **lmscan** | 本地统计特征检测（Zipf 偏离、slop 词密度、burstiness 等） | stef41/lmscan |
| **formulaic-markers** | AI 常用短语检测（delve, nuanced, testament to 等） | lynote-ai/ai-text-detector |
| **transition-density** | 过渡词密度检测（first, moreover, however 等） | lynote-ai/ai-text-detector |
| **compressibility** | 压缩率检测（AI 文本更可压缩） | lynote-ai/ai-text-detector |
| **structured-shape** | 列表/标题结构检测 | lynote-ai/ai-text-detector |
| **formulaic-patterns** | AI 开头/结尾套路检测 | pablocaeg/sloptotal |
| **linguistic-markers** | 加权 AI 词汇检测（100+ 个标记词，分 3 个层级） | pablocaeg/sloptotal |
| **vocabulary-richness** | 词汇丰富度检测（Root TTR, Hapax ratio, Yule's K） | pablocaeg/sloptotal |
| **structural-patterns** | 结构特征检测（冒号密度、列表密度、段落均匀度） | pablocaeg/sloptotal |
| **sentiment-hedging** | 情感与对冲检测（过度对冲、模糊积极表达、填充限定词） | pablocaeg/sloptotal |
| **ngram-perplexity** | n-gram 语言模型困惑度 | 自研 |
| **sliding-window** | 段落级滑动窗口检测 | 自研 |
| **burstiness-stat** | 传统句长突发度检测 | 自研 |
| **statistical-baseline** | 统计基线检测 | 自研 |

---

## 📊 输出示例

### Markdown 报告

```
# 极光查重报告

- **正文长度：** 1200 字符

## AI 检测

- **综合判定：** 大概率人工写作
- **AI 概率：** 12.5%
- **人工概率：** 87.5%
- **检测方法：** ensemble (15 engines, references excluded)

- **各引擎单独评分：**
  - roberta-chatgpt：8.2%
  - lmscan：6.7%
  - formulaic-patterns：0.0%
  - ...

- **高风险句（Top 5）：**
  - (0.65) This is a sample sentence...
  - ...
```

### HTML 报告

生成的 HTML 报告具有以下特点：

- 🎨 渐变紫蓝色头部
- 📊 大字号 AI 率显示（红/黄/绿三色预警）
- 📈 各引擎详细评分条
- 🔴 高风险句高亮显示
- 📱 响应式设计，手机也能看

---

## 🔧 高级用法

### Python API

```python
from aurora_check import run_check

# 检测文本
report = run_check(text="Your text here...")

# 获取 AI 概率
ai_prob = report.ai.ai_probability

# 获取高风险句
flagged = report.ai.flagged_sentences

# 生成 HTML 报告
html = report.to_html()
```

### 调整引擎权重

你可以根据自己的需求调整各引擎的权重，在 `aurora_check.py` 中找到 `weights` 字典：

```python
weights = {
    "roberta-chatgpt": 0.30,      # RoBERTa 模型权重
    "lmscan": 0.20,                # lmscan 权重
    "formulaic-patterns": 0.12,   # 公式化模式权重
    # ... 其他引擎
}
```

---

## 📋 梯队定位

| 梯队 | 代表软件 | 特点 |
|------|---------|------|
| **第一梯队** | Turnitin, GPTZero, Originality.ai | 多个预训练模型 + 全网比对 + 商用支持 |
| **第二梯队** | **极光查重 (Aurora Check)** | 15 引擎集成 + RoBERTa ML 模型 + 本地比对 |
| **第三梯队** | 简单的单引擎检测器 | 只有一两个统计规则 |

---

## 🤝 贡献指南

我们欢迎任何形式的贡献！

### 如何贡献

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 想加入新引擎？

如果你想加入新的检测引擎，请：

1. 在 `aurora_check.py` 中添加新的检测函数
2. 在 `run_ai_detection` 中注册新引擎
3. 调整 `weights` 字典中的权重
4. 测试确保不破坏现有功能

---

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

### 开源项目贡献者

本项目逆向集成了以下开源项目的核心算法，向这些项目的作者表示感谢：

- [lynote-ai/ai-text-detector](https://github.com/lynote-ai/ai-text-detector) - 6 信号检测算法
- [pablocaeg/sloptotal](https://github.com/pablocaeg/sloptotal) - 24 引擎集成方案
- [stef41/lmscan](https://github.com/stef41/lmscan) - 本地统计特征检测
- [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) - RoBERTa ChatGPT 检测模型
- [followsci/bert-ai-text-detector](https://huggingface.co/followsci/bert-ai-text-detector) - BERT 学术文本检测模型

### 特别感谢

> 本项目由 **Doubao AI (豆包)** 协助开发，特别感谢 Doubao 在研究、编码和项目开发中的宝贵帮助。

---

## ⚠️ 免责声明

极光查重的检测结果仅供参考，不能作为学术诚信的最终裁决。AI 检测技术仍然在快速发展中，没有任何工具能做到 100% 准确。

请始终以人类判断为准。

---

## 📞 联系方式

如果有任何问题或建议，欢迎：

- 提交 Issue
- 发送 Pull Request
- 或者直接联系项目维护者

---

**⭐ 如果这个项目对你有帮助，请给它一个 Star！**
