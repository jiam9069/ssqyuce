# M3 LLM 离线评估记录（60 期三通道）

> 诚实边界：双色球开奖为独立随机事件。本评估的目的不是证明"LLM 能预测"，
> 而是把「LLM 相对纯统计/随机基线的差异与成本」量化为可复现证据，供默认配置决策。

## 口径

- **通道**：stat（纯统计+结构约束）/ stat_llm（stat + LLM 观察→选号两轮，llm_samples=1）/ random（同注数均匀随机，固定种子）
- **评估口径关闭第三轮校验**（critique/refine 仅生产启用，边际成本/收益待增量评估）
- **LLM 失败降级**：某期 LLM 无候选时如实回退为 stat 输出并计入 `llm_empty_issues`
- **统计**：paired wilcoxon / sign-test（无差异样本用符号检验），BH 校正跨对比
- **成本**：token 估算 × LOTT_LLM_EVAL_PRICE_PER_1M（默认 $1/1M）

## 运行记录

| run_id | 窗口 | 注数 | 种子 | 结果 |
|---|---|---|---|---|
| llm_eval_20260817_144707 | 4 | 2 | 42 | 冒烟完成（p 全 1.0，样本过小无差异） |
| llm_eval_20260817_171435 | 3 | 1 | 42 | stat 1.333 / stat_llm 0.667 / random 1.000（LLM 通道真实出票） |
| llm_eval_6fd26b28 | 60 | 5 | 42 | 🔄 跑批中（上游 gpt-5.6-sol 网关拥堵，重试率高，预计数十小时级） |

## 待填结论（跑批完成后）

- [ ] 三通道四指标（red_hits_mean / blue_hit_rate / prize_rate_ge5 / roi）汇总
- [ ] stat_llm_vs_stat、stat_llm_vs_random、stat_vs_random 的 paired p 值与 BH 校正
- [ ] llm_empty_issues（LLM 失败期数）与 token 成本汇总
- [ ] 默认配置决策：仅当样本外 paired p_adj<0.05 或结构配平工具支撑时才把 LLM 通道设为默认；否则保持研究开关 + 文档结论
