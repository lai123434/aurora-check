---
name: aurora-check
description: "学术诚信核查：DOI 真伪 + 期刊等级 + AI 写作痕迹。当需要核验一篇论文/参考文献是否真实存在、判断期刊是正规刊还是水刊、检测文本是否 AI 生成或人工改写时使用。关键词：DOI 校验、DOI 查真伪、期刊等级、期刊分区、水刊、掠夺性期刊、AI 检测、AI 率、查重、学术诚信、论文核查、references 核对、文献真伪"
version: "3.0.0"
author: "upstream: lai123434 (MIT) / academic verification added"
---

# Aurora Check · 学术核查

三件事，一条命令跑完：**这篇论文是真的吗、发的刊物什么水平、文字像不像 AI 写的。**

## 快速用

```bash
cd ~/.codebuddy/skills/aurora-check/vendor

python check.py paper 论文.docx      # 完整核查（自动提取 DOI）
python check.py doi 10.3389/frai.2025.1629320
python check.py text 文稿.txt        # 只看 AI 痕迹
python check.py batch ./论文目录/    # 批量

# 加 --json 输出结构化结果，--no-ai 跳过 AI 检测（快）
```

## 三个能力

### 1. DOI 核验
调 `doi.org` 的 handle API 判断是否注册，再用 Crossref 取元数据。
**编造的 DOI 一定查不到**——这是最硬的证据。

- 已注册 → 返回标题、作者、期刊、卷期页码
- 未注册 → 编号不存在，可能是编的
- 网络失败 → 明确说"无法确认"，**不猜**

### 2. 期刊分级
用 OpenAlex 取期刊档案（h-index、2 年均被引、累计发文、是否 DOAJ 白名单、版面费）。

| 档位 | 依据 |
|---|---|
| **A** | h-index ≥ 100 |
| **B** | h-index 45–99 |
| **C** | h-index 12–44 |
| **D** | h-index < 12，需核实办刊质量 |

同时给出风险标记：不在 DOAJ 却宣称开放获取、累计发文过少、版面费异常高。

⚠️ **档位只反映影响力，不等于质量。** 报告里永远带理由，让用户自己判断。

### 3. AI 痕迹
29 个统计信号 + 可选的语言模型困惑度。中文和英文分别标定。

- 特征强度 0~100%，**不是"AI 概率"**
- 判为 AI 时，报告会给出"若据此认定，命中率约 X%"——因为低基准率下误判会被放大

## 输出长什么样

```
## 论文.docx

### DOI 核验
- 10.3389/frai.2025.1629320 — ✅ 已注册
  - 标题：Artificial intelligence skills and their impact...
  - 期刊：Frontiers in Artificial Intelligence　2025　8
  - 期刊档案：B 档 · 领域正规
    - h-index 89 · 2 年均被引 6.2 · 累计发文 2817
    - 在 DOAJ 开放获取白名单内

### AI 痕迹
- 特征强度 23%（human，置信度 medium）
```

## 什么时候用 / 不用

| 场景 | 用不用 |
|---|---|
| 核对参考文献列表里的 DOI 是否真实 | ✅ 最可靠 |
| 学生论文里的期刊是不是水刊 | ✅ 给事实，不给定论 |
| 判断一段文字是否 AI 生成 | ⚠️ 只作参考，见下 |
| 认定学术不端 | ❌ 绝不。工具只提供证据，认定权在人 |

## 边界（必须转达给用户）

1. **DOI 查不到 ≠ 一定造假**。可能是新注册、其他注册机构（DataCite 等）、或网络问题。工具会区分"未注册"和"无法确认"。
2. **期刊档位不等于质量**。h-index 高也可能是有争议的巨型 OA 刊。
3. **AI 痕迹不是抄袭证据**。统计方法的漏检率对"刻意改写"高达 25%~50%，误判在低基准率场景下可被放大数倍。
4. **没有本地模型时**，AI 检测退化为纯统计，对改写稿几乎无效。

## 数据源（全部公开、免 key）

| 用途 | 接口 |
|---|---|
| DOI 是否注册 | `doi.org/api/handles/{doi}` |
| 论文元数据 | `api.crossref.org/works/{doi}` |
| 期刊档案 | `api.openalex.org/sources/issn:{issn}` |
| OA 白名单 | `doaj.org/api/search/journals/issn:{issn}` |

全部只读，不上传用户文本。唯一外发的是 DOI 和 ISSN 字符串。

## 可选：装语言模型（提升中文 AI 检测）

```bash
cd ~/.codebuddy/skills/aurora-check/vendor
python download_model.py     # 约 950MB，ModelScope 源
```

装了之后中文检测会多一个「困惑度」信号，这是目前对改写最鲁棒的一项。
不装也能用，只是弱一些。
