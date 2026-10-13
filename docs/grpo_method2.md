# 原生 GRPO 方法 2：全错组引导轨迹追加

实现目录：`C:\Users\JSWang\Documents\DLnotes\SQL_Agent`。本目录使用现有 Transformers/PyTorch 自定义 GRPO，不使用 veRL。

## 行为

- 默认每题生成 G=6 条普通轨迹。只有执行验证奖励全部为 0，才开始引导重试。
- 每个全错组最多生成 3 次，每次从原始问题重新开始；第一条通过执行验证和独立审核的轨迹追加后立即停止，只追加一条。
- 三次都失败时保留原 G 条轨迹，不注入；它们的普通 GRPO 优势为 0。非全错组按原有流程训练。
- 追加轨迹由本轮冻结旧学生生成。DeepSeek 仅用于独立审核，不生成追加轨迹，也不提供训练分母。
- 当前学生评分使用原始提示和公开轨迹；来源评分使用包含私有引导的真实生成提示。旧学生来源 log-prob 在更新前计算并保存，更新期间不刷新。

## 提示和审核

直接复用本目录 SFT/DPO 的两份原有模板及统计函数，没有修改 SFT/DPO 文件：

- `sft/prompts/gold_repair_supplement_zh.md`
- `sft/prompts/external_answer_leak_audit_zh.md`
- `sft.regenerate_reasoning.trajectory_profile`、`trajectory_comparison`

生成消息顺序与 SFT 一致：原始 system → 私有 Gold 补充 user → 原始问题 user。三次尝试固定以原组第 1 条失败轨迹作为长度、工具次数和表达风格对照。统计使用当前学生 tokenizer，公开 assistant 内容计入长度，工具观察和提示不计入长度。

引导生成先预验证 Gold SQL，要求从 list_tables 开始、execute_sql 执行完全一致的 Gold SQL，执行成功后立即 submit_sql；不改写 SQL，不修饰生成的 reasoning。

审核使用现有 `DeepSeekClient.complete_thinking_text`。输入与 SFT/DPO 一致，包含 question、公开 messages 和 trajectory_comparison，排除私有 Gold 补充。审核必须完整返回精确的 accept；reject、其他输出、截断、请求失败都不注入，并计入三次尝试预算。

## 训练目标

追加轨迹每个训练 token 的概率比为：

$$
\rho_t=\exp\left(\log\pi_\theta(y_t\mid q,y_{<t})-
\log\pi_{\mathrm{old}}(y_t\mid g(q),y_{<t})\right).
$$

保留教师来源的同一串生成 token ID，将源提示替换为原始提示后形成训练序列；源提示长度变化时，按生成 token 重新对齐分母，避免按两个序列的绝对位置直接相除。

原轨迹继续使用原始条件下的学生新旧概率比。G+1 奖励组直接复用 `normalize_group_rewards` 的总体标准差，优势约为错误轨迹 -1/sqrt(G)、新增轨迹 sqrt(G)。损失复用现有 `grpo_token_loss`，先在轨迹内平均，再在组内平均，最后对题目组平均。

提示和工具观察不计算损失，延续原实现对特殊 token 的屏蔽。temperature=1、top_p=1、Top-K 关闭、KL 关闭，确保缓存的原始模型概率对应实际采样分布。原有旧版 gold_injection 仍被拒绝；新功能由独立的 method2_guided_injection 标志启用。

来源分母记录筛选前的生成概率，未加入“执行正确且审核 accept”的筛选归一化项，不宣称整个逐 token 裁剪目标无偏。

## 运行

环境依赖仍使用本目录原有 `[train]`，无需安装 veRL 或 TransferQueue。GPU 训练使用项目要求的 Python 3.12、PyTorch 2.8.0 和 Transformers 4.57.1。

先编辑 `configs/grpo_method2.yaml` 的 student_model、reference_model、train_data，以及 `configs/env.yaml` 的数据路径。KL 为 0 时不加载 reference 模型，但原有配置校验仍要求 reference_model 指向存在的路径，可与 student_model 相同。

```bash
export DEEPSEEK_API_KEY='<在这里填入审核 Key>'
python -m grpo.prepare --config configs/grpo_method2.yaml
python -m grpo.train --config configs/grpo_method2.yaml --max-steps 128
python -m grpo.status --config configs/grpo_method2.yaml
```

配置中的输出目录为 artifacts/grpo/method2。准备阶段沿用现有任务池生成；训练、OOM 降低 micro-batch、checkpoint 保存和续跑均复用本目录原生入口。不会往已有 metrics.jsonl 无条件追加。

审核模型名可配置，Key 只从 DEEPSEEK_API_KEY 环境变量读取，不写入文件。deepseek-v4-flash 是官方兼容名称，实际返回模型保存在审核记录中。

## 记录与验证边界

轨迹记录保存来源标记、重试结果、审核返回、轨迹统计比较、私有引导和每轮私有来源 prompt ID；公开训练 token 不包含 Gold 补充。保存的 source_action_log_probs 可以复查更新分母。指标区分原学生成功率、引导尝试数量、实际注入数量，以及两种来源各自的裁剪比例。

CPU 测试包含真实 SQLite 工具和执行验证、引导与训练上下文分离、来源 log-prob 数值与位置对齐、SFT/DPO 审核输入、成功即停和三次失败、G+1 优势及各组等权。模型生成和外部审核采用受控替身。

实际 GPU 训练、完整在线审核调用和恢复后的训练效果仍需在训练环境验证；CPU 测试不代表模型效果。
