# EVIR 项目说明文档（供 GPT 分析）

本文档用于把当前项目的目标、结构、实现状态和待改进方向整理成一个便于大模型分析的 README。它不是原始 README 的替代品，而是一个更偏“项目体检/架构说明/下一步分析入口”的上下文包。

## 1. 项目一句话概括

EVIR（Evidence-Versioned Investment Research）是一个 Python 原型项目，目标是构建一个可审计、证据驱动、多 Agent 协作的股票投资研究系统。

它不是简单输出“买入/卖出”的黑盒模型，而是把投研过程拆成多个角色：

- 基本面分析
- 财务报表分析
- 估值分析
- 行业比较
- 宏观环境分析
- 技术面分析
- 回测分析
- 组合约束分析
- 多头观点
- 空头/谨慎观点
- 风险审查
- 反证检查
- 合并决策
- 报告生成
- 事后评估与新证据修正

每个 Agent 会生成结构化的 claim/evidence commit，并写入 ticker 专属 workspace。最后系统根据这些证据合并出投资建议、生成 Markdown/JSON 报告，并在出现新证据后做决策修正。

## 2. 当前项目状态

截至当前检查，项目已经是一个能完整运行的工程原型：

- 主研究流程已经跑通。
- 支持 mock 数据和 yfinance 真实数据。
- 支持部分 LLM Agent，当前默认 LLM provider 为 DeepSeek。
- 如果 LLM 不可用，支持回退到确定性规则逻辑。
- 已生成 AAPL、MU、INTC 的研究输出。
- 已有测试套件，当前测试结果为 `82 passed`。
- 项目当前目录不是 git 仓库，因此无法从提交历史判断迭代过程。

整体成熟度可以理解为：

- 研究系统框架：基本完成。
- 多 Agent 分工：已经搭好且能产出结构化结果。
- 证据追踪与报告：已经可用。
- LLM 化程度：部分完成。
- 真实投研质量：仍属于原型阶段，需要加强数据质量、Agent 推理、冲突合并和组合评估。
- 模拟盘：代码已实现，但当前没有发现 `outputs/paper_trading/` 下的实际运行产物。

## 3. 技术栈

主要依赖见 `requirements.txt`：

```text
pydantic>=2.7,<3.0
pytest>=8.0,<9.0
yfinance>=0.2.40
pandas>=2.0,<3.0
numpy>=1.26,<3.0
```

项目是纯 Python 工程，主要使用：

- Pydantic：结构化 schema 和结果校验。
- pytest：测试。
- yfinance：真实市场与财务数据。
- pandas/numpy：数据处理、指标和回测。
- 本地 JSON：workspace、报告、缓存和模拟盘状态存储。
- DeepSeek API：可选 LLM 分析。

## 4. 主流程概览

主入口是 `main.py`。

完整 pipeline 大致如下：

```text
选择数据源
  -> 获取公司数据
  -> 解析 LLM client
  -> 创建 ticker workspace
  -> 创建多个研究分支
  -> ResearchCoordinatorAgent 生成初始计划
  -> 多个 source-analysis Agent 并行分析
  -> LLMAnalysisAgent 综合分析
  -> BullAgent 生成多头观点
  -> BearAgent 生成谨慎/空头观点
  -> RiskAgent 做风险审查
  -> Evidence processor 做证据质量和时效处理
  -> CounterEvidenceAgent 生成反证问题
  -> 再次处理证据
  -> ResearchCoordinatorAgent 做合并前覆盖审查
  -> 保存 workspace
  -> MergeAgent 合并出最终建议
  -> ReportAgent 生成 Markdown 和 JSON 报告
  -> Evaluator 评估报告和审计质量
  -> RevisionEngine 根据新证据修正原建议
  -> 保存修正报告
```

主流程返回：

```python
tuple[Workspace, MergeResult, str, str]
```

分别代表：

- workspace
- 合并决策结果
- Markdown 报告路径
- JSON 报告路径

## 5. 目录结构说明

```text
.
├── README.md
├── PROJECT_README_FOR_GPT.md
├── requirements.txt
├── config.py
├── main.py
├── paper_trade.py
├── agents/
├── data/
├── evidence/
├── revision/
├── evaluation/
├── memory/
├── models/
├── llm/
├── paper_trading/
├── tests/
├── outputs/
└── scripts/
```

### 5.1 根目录

#### `main.py`

主研究 pipeline。负责串联数据、workspace、Agent、证据处理、合并、报告、评估和修正。

支持命令行参数：

```bash
python main.py
python main.py --ticker AAPL --data-provider mock --llm-provider none
python main.py --ticker AAPL --data-provider real
python main.py --preview-lines 0
```

#### `paper_trade.py`

模拟盘 CLI 入口。它会对一组 ticker 运行 EVIR 研究流程，然后把最终建议转成模拟交易动作。

示例：

```bash
python paper_trade.py
python paper_trade.py --tickers MU,INTC,NVDA,AMD
python paper_trade.py --llm-provider none
```

#### `config.py`

项目级配置，包括：

- 默认 ticker：`AAPL`
- 是否使用真实数据：`USE_REAL_DATA = False`
- 是否使用 LLM：`USE_LLM = True`
- 默认 LLM provider：`deepseek`
- 默认模型：`deepseek-v4-pro`
- 缓存目录
- 证据时效阈值
- 投资决策阈值
- 模拟盘资金、最大仓位、最小交易金额、默认股票池

注意：当前 `.env` 文件存在，但不要把其中的密钥内容提供给模型或提交到公开仓库。

## 6. Agent 系统

Agent 代码集中在 `agents/`。

### 6.1 已有 Agent

#### `ResearchCoordinatorAgent`

负责研究协调记录，包括：

- 初始研究计划
- 分支覆盖检查
- 合并前审查
- 后续优先级建议

当前它更像审计与计划记录 Agent，还没有真正主动调度其他 Agent。未来目标是成为核心 orchestrator。

#### `FundamentalAgent`

基本面分析，包括：

- 收入增长
- 利润率
- 自由现金流
- 资产负债表
- 资本配置
- 基本面估值风险

该 Agent 已支持 LLM-first，并有规则 fallback。还包含简单校验逻辑，避免 LLM 生成明显违背输入指标方向的结论。

#### `FinancialStatementAgent`

财务报表质量分析，主要是规则逻辑，关注：

- 三年收入增长
- 自由现金流
- ROE
- 杠杆
- 财务质量

#### `ValuationAgent`

估值分析，包括：

- forward P/E
- sector P/E
- earnings growth
- free cash flow yield
- 相对估值溢价/折价

已支持 LLM-backed 输出和 deterministic fallback。

#### `IndustryComparisonAgent`

行业比较，使用粗粒度 sector benchmark 比较：

- 收入增长
- 净利率
- forward P/E

当前 benchmark 仍较粗糙，属于占位性质。

#### `MacroAgent`

宏观环境分析，关注：

- 利率环境
- 需求环境
- 通胀压力
- 汇率/宏观背景

当前主要是规则和模板驱动。

#### `TechnicalAgent`

技术面分析，包括：

- 20 日/50 日均线趋势
- RSI
- 波动率
- 支撑位/阻力位

已支持 LLM-backed 输出和 deterministic fallback，并有基本方向校验。

#### `BacktestAgent`

简单回测摘要，当前策略主要是 MA20/MA50 趋势跟随。输出包括：

- 胜率
- 最大回撤
- 年化收益
- 总收益
- 买入持有收益
- 超额收益

#### `PortfolioAgent`

组合适配性审查，关注：

- 当前仓位
- 最大仓位
- 相关性
- 流动性
- 组合角色

#### `LLMAnalysisAgent`

LLM 综合分析 Agent。它不是 coordinator，而是 synthesis agent。它读取已有 evidence commits，然后生成结构化、证据关联的中文投研洞察。

如果 LLM 不可用或输出不合法，该 Agent 会安全失败，不阻断主流程。

#### `BullAgent`

根据已有正面证据生成多头投资假设。

#### `BearAgent`

根据已有风险、估值、波动、阻力位等证据生成谨慎或空头观点。

#### `RiskAgent`

审查关键风险，包括：

- 估值风险
- 波动风险
- 证据时效风险
- 反证缺口
- 质量/置信度问题

已支持 LLM risk review 和 fallback。

#### `CounterEvidenceAgent`

针对重要正面观点生成反证问题。它会问类似：

- 收入增长是否可能放缓？
- 盈利能力是否受到成本或周期压力？
- 技术动量是否已经过期？

支持额外使用 LLM 生成一个经过校验的反证问题。

#### `MergeAgent`

合并所有分支，输出最终建议。当前主要是确定性评分和规则逻辑。

输出包括：

- final_recommendation
- confidence
- supporting claims
- opposing claims
- key conflicts
- risk adjustment
- decision rationale
- conditions for revision
- decision scorecard

当前改进空间较大：合并逻辑仍偏关键词和规则，未来可引入更强的冲突推理和 LLM-assisted review。

#### `ReportAgent`

生成最终投资报告，包括：

- Markdown 报告
- JSON 报告
- 评分卡
- 支持证据
- 反对与谨慎证据
- 冲突
- 修正条件

## 7. 数据层

数据代码在 `data/`。

### 7.1 数据 provider

`data/providers.py` 定义两个 provider：

- `mock`
- `real`

mock provider 主要用于测试和 demo，目前 mock dataset 主要支持 AAPL。

real provider 通过 `yfinance` 获取真实市场和财务数据。

### 7.2 重要数据模块

```text
data/mock_data.py              # 本地 mock 公司数据
data/real_data.py              # 真实公司数据聚合
data/market_data.py            # 市场价格数据
data/financial_data.py         # 财务数据
data/technical_indicators.py   # 技术指标计算
data/backtest.py               # 简单回测逻辑
data/cache.py                  # 本地缓存
data/mock_new_evidence.py      # mock 新证据
data/real_new_evidence.py      # 真实模式新证据
```

真实数据缓存目录：

```text
outputs/cache/prices/
outputs/cache/financials/
```

当前已存在 MU、INTC 的价格和财务缓存。

## 8. 证据系统

证据处理代码在 `evidence/`。

主要职责：

- 计算 evidence quality score。
- 判断 evidence temporal status。
- 标记证据是否有效、陈旧或过期。

相关模块：

```text
evidence/evidence_scorer.py
evidence/temporal_checker.py
evidence/processor.py
```

证据时效阈值在 `config.py`：

```python
EVIDENCE_RECENCY_THRESHOLDS_DAYS = {
    "technical_stale": 7,
    "technical_expired": 14,
    "news_stale": 30,
    "news_expired": 90,
    "financial_stale": 180,
    "financial_expired": 365,
    "default_stale": 90,
}
```

## 9. Workspace 和版本化研究记忆

相关代码在 `memory/`：

```text
memory/workspace.py
memory/storage.py
```

核心思想是：每个 ticker 有一个 workspace，workspace 内有多个 branch。每个 Agent 往自己的 branch 写入 commit。

当前主要分支包括：

```text
research-coordination
fundamental-analysis
financial-statement-analysis
valuation-analysis
industry-comparison
macro-analysis
technical-analysis
backtest-analysis
portfolio-review
llm-analysis
bull-case
bear-case
risk-review
counter-evidence
```

workspace 输出路径：

```text
outputs/workspaces/<TICKER>_workspace.json
```

workspace 是调试整个项目最重要的 artifact，因为它包含所有分支、commit、claim、evidence、risk tag、confidence、evidence score 和 temporal status。

## 10. LLM 系统

LLM 代码在 `llm/`。

```text
llm/base.py
llm/mock_client.py
llm/deepseek_client.py
llm/factory.py
llm/cache.py
llm/json_utils.py
```

当前支持：

- `mock`
- `deepseek`

LLM 使用策略：

- LLM-first where available。
- 结构化 JSON 输出。
- Pydantic 校验。
- 输出不合法或调用失败时 fallback。
- 可选本地缓存。

当前已经接入 LLM 的 Agent 包括：

- ResearchCoordinatorAgent
- FundamentalAgent
- ValuationAgent
- TechnicalAgent
- RiskAgent
- LLMAnalysisAgent
- CounterEvidenceAgent 的部分逻辑

当前大多数其他 Agent 仍是规则/模板驱动。

## 11. 报告、评估和修正

### 11.1 报告

报告输出在：

```text
outputs/reports/
```

每个 ticker 通常有：

```text
<TICKER>_report.md
<TICKER>_investment_report.json
<TICKER>_revision_report.md
```

### 11.2 评估

评估代码在 `evaluation/`：

```text
evaluation/evaluator.py
evaluation/metrics.py
```

评估维度包括：

- evidence coverage
- temporal validity
- conflict coverage
- decision traceability
- audit completeness
- overall score

评估输出：

```text
outputs/evaluation/<TICKER>_evaluation.json
```

### 11.3 修正

修正代码在 `revision/`：

```text
revision/revision_engine.py
```

RevisionEngine 会接收：

- 原 workspace
- 原 merge result
- new evidence

然后判断原有观点是：

- 获得支持
- 被削弱
- 被否定
- 证据过期
- 保持不变

最后输出 revised recommendation 和修正报告。

当前打开的文件 `outputs/reports/AAPL_revision_report.md` 就是修正报告。

## 12. 当前已有输出

当前 `outputs/` 下已经有以下 ticker 的结果：

### AAPL

文件：

```text
outputs/workspaces/AAPL_workspace.json
outputs/reports/AAPL_report.md
outputs/reports/AAPL_investment_report.json
outputs/reports/AAPL_revision_report.md
outputs/evaluation/AAPL_evaluation.json
```

报告结论：

- 原最终建议：持有/观察
- 置信度：低
- 当前买点评分：12.4/100
- 风险收益评分：17.2/100
- 风险等级：高
- 修正后建议：回避

AAPL 评估：

```json
{
  "evidence_coverage_score": 1.0,
  "temporal_validity_score": 0.68,
  "conflict_coverage_score": 0.28,
  "decision_traceability_score": 1.0,
  "audit_completeness_score": 1.0,
  "overall_score": 0.79,
  "details": {
    "ticker": "AAPL",
    "branch_count": 14,
    "commit_count": 38,
    "conflict_count": 6,
    "final_recommendation": "Hold"
  }
}
```

### MU

文件：

```text
outputs/workspaces/MU_workspace.json
outputs/reports/MU_report.md
outputs/reports/MU_investment_report.json
outputs/reports/MU_revision_report.md
outputs/evaluation/MU_evaluation.json
```

报告结论：

- 最终建议：持有/观察
- 置信度：中

评估总分：

```text
0.82
```

### INTC

文件：

```text
outputs/workspaces/INTC_workspace.json
outputs/reports/INTC_report.md
outputs/reports/INTC_investment_report.json
outputs/reports/INTC_revision_report.md
outputs/evaluation/INTC_evaluation.json
```

报告结论：

- 最终建议：持有/观察
- 置信度：低

评估总分：

```text
0.83
```

## 13. 模拟盘系统

模拟盘相关代码：

```text
paper_trade.py
paper_trading/engine.py
```

模拟盘设计：

```text
ticker universe
  -> 对每只股票运行 EVIR research pipeline
  -> 读取最终建议
  -> 获取最新收盘价
  -> 将建议转成模拟交易
  -> 更新组合
  -> 写入交易流水、净值曲线和摘要报告
```

默认逻辑：

- Buy：买入到目标仓位。
- Hold：维持已有仓位。
- Sell/Avoid：退出已有仓位。
- confidence 影响仓位大小。
- 小于最小交易金额的交易会跳过。

预期输出：

```text
outputs/paper_trading/portfolio.json
outputs/paper_trading/trades.csv
outputs/paper_trading/equity_curve.csv
outputs/paper_trading/summary.md
```

当前检查时没有发现这些模拟盘输出，说明模拟盘功能可能还没有实际跑过，或输出目录尚未生成。

## 14. 测试情况

测试目录：

```text
tests/
```

当前测试文件数量：31。

已覆盖方向包括：

- schemas
- workspace
- evidence quality
- temporal checking
- data provider
- market data fallback
- financial data cache
- technical indicators
- backtest
- all major agents
- LLM fallback
- LLM clients
- pipeline
- CLI
- report generation
- merge logic
- revision engine
- paper trading

当前运行结果：

```text
82 passed in 1.80s
```

这说明当前代码在已有测试约束下是稳定的。

## 15. 当前主要限制

项目目前仍是研究和工程原型，有以下限制：

1. 多数 Agent 仍是规则/模板驱动，不是完整 LLM 专家 Agent。
2. ResearchCoordinatorAgent 还没有真正主动调度、回溯、要求 Agent 补充分析。
3. MergeAgent 仍偏确定性规则和关键词逻辑，冲突理解能力有限。
4. real data 依赖 yfinance 和 Yahoo Finance 可用性。
5. sector benchmark 比较粗糙，行业数据质量不足。
6. mock 数据主要支持 AAPL。
7. Paper trading 忽略交易成本、滑点、税、股息和真实执行约束。
8. 本地 JSON 存储适合原型，但不适合规模化多用户或长期历史追踪。
9. 当前报告虽然可审计，但可读性和投资逻辑表达还有提升空间。
10. 证据时效会影响技术面判断，AAPL 当前已有不少过期技术证据。

## 16. 适合让 GPT 重点分析的问题

可以把本项目发给 GPT，让它重点分析以下问题：

### 16.1 架构层面

- 当前多 Agent 架构是否合理？
- ResearchCoordinatorAgent 应该如何从“记录计划”升级为真正的 orchestrator？
- 是否应该引入统一的 LLM Agent base class？
- workspace/branch/commit 的抽象是否足够清晰？
- 是否需要把 MergeAgent 拆成 SynthesisAgent、ConflictReviewAgent、DecisionAgent？

### 16.2 投研逻辑层面

- 当前 claim/evidence schema 是否足够表达投资研究逻辑？
- 支持性证据和反对性证据的权重设计是否合理？
- 如何更好地区分“长期基本面观点”和“短期交易时机”？
- 当前评分卡是否容易误导用户？
- 如何设计更可信的投资建议分级？

### 16.3 LLM 迁移层面

- 哪些 Agent 最值得优先 LLM 化？
- 每个 Agent 的 prompt 和 JSON schema 应该如何设计？
- 如何防止 LLM 编造证据？
- 如何做 LLM 输出的事实一致性校验？
- 是否应该引入 multi-pass critique/revision？

### 16.4 数据层面

- yfinance 数据是否足够？
- 还应接入哪些数据源？
- 如何设计行业 benchmark？
- 如何处理财报、新闻、价格数据的时效差异？
- 如何把新证据修正做成更真实的事件驱动系统？

### 16.5 工程层面

- 当前模块边界是否清晰？
- 测试覆盖还缺什么？
- 本地 JSON 是否应该迁移到 SQLite/PostgreSQL？
- 如何设计可重复实验和回测评估？
- 如何把 paper trading 从单次 rebalance 扩展成持续模拟账户？

## 17. 推荐下一步路线

一个合理的下一阶段路线：

1. 增强 README 和开发文档，明确项目目标、输入输出和限制。
2. 把 `ResearchCoordinatorAgent` 升级为真正的调度器：
   - 读取已有 workspace。
   - 识别缺失分支。
   - 要求某些 Agent 重跑或补充。
   - 判断是否允许进入 merge。
3. 抽象统一的 LLM Agent base：
   - prompt builder
   - JSON parser
   - Pydantic validator
   - fallback handler
   - evidence binder
4. 优先 LLM 化：
   - FinancialStatementAgent
   - IndustryComparisonAgent
   - MacroAgent
   - BullAgent
   - BearAgent
5. 改进 MergeAgent：
   - 分离支持分、风险分、冲突分、时效分。
   - 增加 LLM-assisted conflict review。
   - 明确最终建议与仓位建议之间的关系。
6. 运行 paper trading：
   - 先用小股票池。
   - 禁用 LLM 跑 deterministic baseline。
   - 再启用 LLM 比较结果。
7. 加强真实数据和 benchmark：
   - 行业/sector 数据。
   - 财报历史。
   - 新闻事件。
   - 分析师预期。
8. 建立持续评估：
   - 每次 pipeline 输出保存版本。
   - 每次建议关联未来收益表现。
   - 评估建议质量，而不仅是报告完整性。

## 18. 给 GPT 的推荐提示词

可以把这个文档和项目文件一起发给 GPT，然后使用下面的提示词：

```text
你是一个资深 AI 工程师和量化/基本面投研系统架构师。

我有一个 Python 项目 EVIR，目标是构建一个证据驱动、多 Agent、可审计的股票投资研究系统。

请你阅读项目结构和说明后，从以下角度分析：

1. 当前架构是否清晰，是否存在职责混乱或过度耦合？
2. 多 Agent 设计是否合理？哪些 Agent 应该合并、拆分或升级？
3. LLM 接入方式是否安全可靠？如何防止幻觉和证据错配？
4. claim/evidence/workspace/branch/commit 这些核心抽象是否足够？
5. MergeAgent 的决策逻辑应该如何改进？
6. RevisionEngine 的新证据修正机制是否合理？
7. Paper trading 应该如何设计成能评估投研系统真实表现的模块？
8. 这个项目下一步最值得做的 5 个工程任务是什么？

请给出具体、可执行的改进建议，最好能按优先级排序。
```

## 19. 总结

EVIR 当前已经完成了一个端到端可运行的多 Agent 投研系统原型。它最有价值的部分是：

- 明确的多角色研究流程。
- claim/evidence 结构化记录。
- workspace/branch/commit 风格的审计轨迹。
- 合并建议、报告、评估、修正全链路。
- LLM-first + deterministic fallback 的迁移方向。
- 测试套件保持通过。

当前最大的问题不是“能不能跑”，而是“如何从规则驱动原型升级为真正可信的 LLM-assisted 投研系统”。下一步重点应放在 coordinator 调度、统一 LLM agent 抽象、冲突合并逻辑、真实数据质量、模拟盘评估和长期可审计存储上。
