# SQL Agent RL

基于 Spider 1.0 的多轮 Text-to-SQL 项目。项目支持成功轨迹 SFT、偏好轨迹 DPO 和 Binary GRPO；`sql_agent/` 提供共同的只读 SQLite 环境与执行验证器，`evaluation/` 提供重新评测所需的代码。

原始 Spider 数据集位于 `../datasets/spider/spider_data/`。仓库内的预处理数据、训练轨迹、检查点和历史评测记录均已清理；运行下列步骤会重新生成它们。配置中的模型路径和输出路径沿用原远程环境示例，运行前请改成实际路径。

## 准备数据与环境

```bash
pip install -e '.[train,dev]'
python -m data.preprocess_spider \
  --spider-root ../datasets/spider/spider_data \
  --output-dir data --seed 42
pytest -q
```

Agent 每轮输出一个 JSON action，可调用 `list_tables`、`inspect_table`、`inspect_tables`、`inspect_values`、`execute_sql`。环境只允许只读 SQLite SELECT/WITH；常规环境回合结束时用最近一次 `execute_sql` 的 SQL 进行执行结果验证。SQL-Planner 采集的独立提交规则见下文。

当前默认环境、SFT 轨迹采集、GRPO 和评测配置均最多交互 10 轮。`execute_sql` 最多返回 50 行，`inspect_values` 最多返回 50 个不同值；`list_tables`、`inspect_table` 的表、列、外键上限分别为 128、128、128。单条观察最多 1024 token，单个文本值最多 512 字符，因此较宽的结果可能在达到 50 行之前再次截断；观察中的 `truncation_reasons` 会说明原因。最终验证器重新执行完整 SQL，不使用截断后的观察结果。

环境上下文、SFT 采集、SFT 训练及 GRPO 轨迹的长度上限均为 16384 token；更长的轨迹会增加训练显存占用。默认设置见 `configs/env.yaml`，对应的评测环境设置见 `configs/env_eval_10turn.yaml`。运行前应同时核对环境与训练配置，避免轮数或序列长度不一致。

## SFT

`sft.collect` 使用 Teacher 生成并筛选执行正确的多轮轨迹，`sft.train` 对 Student 做 action-only SFT。运行前在配置中填写可用的 Teacher、Student 和数据路径。

```bash
python -m sft.collect --config configs/sft_cold_start_v2.yaml
python -m sft.train --config configs/sft_train_v2.yaml
python -m sft.train_status --config configs/sft_train_v2.yaml
```

## GRPO

`grpo.prepare` 从预处理后的训练集按难度选择任务；当前配置直接以 SFT 最终检查点初始化 Student 和 Reference，不依赖已删除的 self-training 产物。随后 `grpo.train` 在线生成多轮轨迹并以执行正确性计算奖励。

```bash
python -m grpo.prepare --config configs/grpo_balanced_binary_512_seed42.yaml
python -m grpo.train --config configs/grpo_balanced_binary_512_seed42.yaml --max-steps 128
python -m grpo.status --config configs/grpo_balanced_binary_512_seed42.yaml
```

训练后可通过 `python -m evaluation.run_eval` 重新生成评测记录。旧版轨迹包含已经删除的 `sample_rows`、`submit` 工具，不能直接作为当前接口的训练数据。

## Qwen2.5-Coder-3B `<tool>` 多轮评测

`evaluation.run_tool_xml` 是独立的可恢复评测入口，不改写历史评测结果。它在首轮提供完整 schema；每次助手回复必须是 `<reasoning>...</reasoning>` 加一个 `<tool>{"name":"...","arguments":{...}}</tool>`。`list_tables`、`inspect_tables`、`inspect_values`、`execute_sql` 和最终 `submit_sql` 使用同一格式。每次工具执行后返回 `<observation>`，其中包含 `turns_remaining`；探索最多 10 次，提交另占一次。模型生成在 `</tool>` 截止，防止自行编造工具结果。评测使用贪心解码和本仓库的 `ExecutionVerifier`，不是 SQL-Trail 作者的官方评测器。

有 GPU 后先启动 OpenAI-compatible 的 vLLM 服务，再从仓库根目录运行：

```bash
python -m evaluation.run_tool_xml \
  --endpoint http://127.0.0.1:8004 \
  --model /root/autodl-tmp/Qwen2.5-Coder-3B-Instruct \
  --task-file data/external_dev.jsonl \
  --env-config configs/env_sql_planner_qwen25_coder_3b.yaml \
  --spider-root /root/autodl-tmp/sqlagent/datasets/spider/spider_data
```

默认输出到 `artifacts/sql_planner/qwen25_coder_3b_base_tool_xml_forced_submit_v4_eval/`，逐题保存轨迹、完整对话并定期更新 `summary.json`。同一配置重启时自动跳过已完成题目；换提示词、模型或数据时应使用新的 `--output-dir`。可先用 `--limit 10` 做小规模检查，但全量评测必须使用新的输出目录。

`tool_xml_direct_aligned_v4_forced_submit` 默认每次生成最多 512 tokens，与历史单轮对照的已记录配置一致；也可通过 `--max-tokens` 调整。问题和 schema 的包装复用仓库单轮提示词构造函数，系统提示保留 `<reasoning>`、`<tool>`，强调直接生成候选 SQL、按明确问题修复、空结果不等于错误。每次工具 observation 后的 user 消息都再次附上完整 schema 和原问题，包括复用起始 SQL 后的第一条 observation。轮数到 0 时，增加一条包含完整 schema 和问题的最终提交提醒，只允许 `submit_sql`。若模型仍输出其他动作或格式不正确，评测器从最后一条尝试提交的 SQL、最后成功执行的 SQL 或原始 SQL 依次选择兜底 SQL，写入 `source: evaluator_fallback` 的单独提交步骤及原因。若从未生成任何 SQL，则兜底为 `SELECT NULL WHERE 0`。原生提交与兜底提交分别记为 `submitted_sql` 和 `forced_submit_sql`。历史单轮运行没有归档完整提示词，因此不声称新提示词与历史请求逐字相同。manifest 保存完整环境配置及解码参数。

要真正从历史单轮答案继续交互，在上述命令中追加：

```bash
--initial-sql-dir artifacts/sql_planner/qwen25_coder_3b_base_direct_full_schema_external_dev_eval \
--output-dir artifacts/sql_planner/qwen25_coder_3b_base_direct_seeded_forced_submit_v4_eval
```

该模式按 task_id 读取原 SQL，校验数据集哈希并记录候选 SQL 哈希。第一步由程序包装成 `execute_sql`，标记 `source: baseline_replay`，计入一次探索预算且不消耗模型生成 tokens；随后返回真实执行 observation，由模型决定修复或提交。gold SQL 和基线判分不进入对话。它是固定起始答案的干预实验，不是 SQL-Trail 原始流程。

历史 60.06% 来自离线结果重比较，并非上述入口的原始 `ExecutionVerifier` 分数。全量新运行结束后，可用同一重评分脚本比较新多轮结果和旧单轮结果：

```bash
PYTHONPATH=. python research/sql_trail/audit.py \
  --multi-run artifacts/sql_planner/qwen25_coder_3b_base_direct_seeded_forced_submit_v4_eval \
  --output-dir artifacts/sql_planner/qwen25_coder_3b_base_direct_seeded_forced_submit_v4_eval/rescore
```

该脚本仅使用已保存 SQL，不调用模型，分别报告原评分与官方 `result_eq` 比较函数分数；仍不是完整 test-suite 评测。脚本要求全部 1034 题已完成，且本目录已有 `research/sql_trail/official_exec_eval.py` 来源快照。

## SQL-Planner：自由工具顺序轨迹采集

`sql_planner.collect` 调用 DeepSeek API，初始只提供自然语言问题和五种工具的定义，不预先注入数据库 schema。采集接口提供 `list_tables`、`inspect_tables`、`inspect_values`、`execute_sql` 和 `submit_sql`。`inspect_tables` 接收 1–8 个表名，一次调用返回各表的列、主键和外键；单个表名无效时仅该表返回错误。模型自行决定探索顺序，每次调用后收到真实 SQLite 工具反馈。提示词要求每次模型回复恰好调用一个工具；如果 API 仍返回多个工具调用，采集器将其记为 `multiple_tool_calls` 并停止该轨迹，不执行这一批中的任何调用。最多允许 10 次探索工具调用，另有一次不占探索预算的 `submit_sql`；用满探索预算后，下一次模型回复只能调用 `submit_sql`。模型可以提前提交，提示词要求提前提交前在上一轮成功执行 `execute_sql`，并建议复用该 SQL。采集器不硬性校验提前提交的相邻工具或 SQL 是否相同；调用 `submit_sql` 后执行并用完整结果与标准答案比对。未调用 `submit_sql` 的轨迹不计为正确提交。轨迹保存 `tool_sequence`、探索调用数、每步参数和观察、最终 SQL、正确性、API token 用量及对话消息，方便后续分析调用顺序。此阶段不筛选顺序，也不训练模型。

先运行上面的 Spider 预处理命令生成 `data/train.jsonl`，配置好 `configs/env_sql_planner_qwen25_coder_3b.yaml` 中的 Spider 路径和 Qwen2.5-Coder-3B-Instruct tokenizer 路径，并在环境变量 `DEEPSEEK_API_KEY` 中设置 API 密钥，然后运行；本机可用 `--env-config` 指向本地配置：

```bash
python -m sql_planner.collect --tasks data/train.jsonl --limit 100 --samples-per-task 3
```

默认模型为 `deepseek-v4-flash`，结果逐条保存在 `artifacts/sql_planner/multi_table_v11/trajectories/`；同一任务和样本编号已有文件时会跳过，便于中断后续跑。输出目录的 `run_manifest.json` 固定模型、提示、数据与环境配置，参数变化时需使用新的输出目录，避免混合不同实验。`--limit 0` 表示使用全部训练任务。采集只读取训练集，标准 SQL 不会发给 DeepSeek。模型和运行次数会影响 API 费用；可先用较小的 `--limit` 检查记录格式。

若要按 SQL-Trail 的口径，仅对官方 `train_spider.json` 中的 7,000 条题目采集自由工具顺序轨迹，可运行：

```bash
export DEEPSEEK_API_KEY="<your-api-key>"
python -m sql_planner.collect_spider_train \
  --spider-root ../datasets/spider/spider_data \
  --samples-per-task 1 \
  --workers 16
```

该入口不会读取 `train_others.json`，也不会向模型提供 Gold SQL。默认输出目录为 `artifacts/sql_planner/spider_train_7000_multi_table_v11/`，与旧版采集结果分开。模型可以自由选择探索工具，但每次回复只能提出一个调用；探索次数达到上限后，采集器仅提供 `submit_sql`。每条轨迹独立保存，重复运行会跳过已有文件。建议先加 `--limit 100 --workers 4` 做小规模连通性与费用检查，再运行全部 7,000 条。

## DPO：数学目标与当前偏好数据构造

项目的后训练顺序为 **成功轨迹 SFT → DPO → Binary GRPO**。DPO 使用同一上下文下的优选与劣选续写，让 Qwen2.5-Coder-3B 的 SFT 模型学习更可靠的 SQL 修复行为。当前数据属于 **Gold 引导生成、真实数据库验证、独立模型审核的合成偏好**；尚未逐条人工审核。

### 1. 偏好对与训练范围

每条数据记为 $(x,y_w,y_l)$：

| 符号 | 在本项目中的含义 |
| --- | --- |
| $x$ | 共同前缀：系统提示、原问题、分叉前的助手回复与真实工具观察 |
| $y_w$ | chosen：教师从分叉点生成、数据库验证正确且通过严格审核的修复后缀 |
| $y_l$ | rejected：原始失败轨迹从同一分叉点开始的后缀 |
| $\pi_\theta$ | 从 SFT 检查点初始化、接受 DPO 更新的策略模型 |
| $\pi_{\mathrm{ref}}$ | 冻结的同一 SFT 检查点 |
| $\beta$ | 相对参考模型的偏好分数缩放参数，当前配置为 $0.1$ |

分叉后的两条分支可以产生不同的工具结果。各分支的助手回复均条件化于该分支自己的真实历史；工具观察作为上下文输入。

定义本项目实际使用的分支分数：

$$
s_\theta(y\mid x)=\sum_{i=1}^{N_y}m_i\log\pi_\theta(z_i\mid x,z_{<i}),
\qquad
m_i=\begin{cases}
1,& z_i\text{ 属于分叉点及之后的助手回复},\\
0,& \text{其他位置}.
\end{cases}
$$

这里 $z$ 是按时间顺序序列化的后缀消息。`dpo/prepare.py` 对助手回复内容及模板终止 token 建立 mask，覆盖学生可见的 `<reasoning>` 和 `<tool>` 动作；共同前缀、系统消息、用户消息和工具观察不计入分数。对于“成功执行但答案错误”的主要组，分叉前的错误 SQL 和返回观察完整保留，从**观察返回后的下一条助手回复**开始计算 loss。

`dpo/train.py::branch_logp` 对 mask 内 token 的 log probability **求和**，不按长度取平均。因此后缀长度会影响分支分数，数据筛选和评估应关注长度分布。

### 2. DPO 损失

标准 DPO 通过策略相对参考策略的 log probability 差构造偏好分数，参见 [DPO 原论文](https://arxiv.org/abs/2305.18290)。本项目将多轮助手续写的上述分数代入该目标：

$$
\Delta_\theta(x,y_w,y_l)
=\bigl[s_\theta(y_w\mid x)-s_\theta(y_l\mid x)\bigr]
-\bigl[s_{\mathrm{ref}}(y_w\mid x)-s_{\mathrm{ref}}(y_l\mid x)\bigr].
$$

$$
\mathcal L_{\mathrm{DPO}}(\theta)
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[\log\sigma\left(\beta\Delta_\theta(x,y_w,y_l)\right)\right]
=\mathbb E_{\mathcal D}\left[\operatorname{softplus}(-\beta\Delta_\theta)\right].
$$

其中 $\sigma(u)=1/(1+e^{-u})$。减小损失会提高 chosen 相对 rejected 的优势，并以冻结的 SFT 模型为基准。数据库奖励用于构造和核验偏好标签，不直接作为该损失的乘数。

单对样本的梯度为：

$$
\nabla_\theta\ell
=-\beta\sigma(-\beta\Delta_\theta)
\left[\nabla_\theta s_\theta(y_w\mid x)-\nabla_\theta s_\theta(y_l\mid x)\right].
$$

训练实现预先缓存冻结参考模型的两条分支分数，再加载策略模型进行全参数更新；两条分支分别反向传播，以降低同时保留两份长序列计算图的显存占用。当前 `configs/dpo_reasoning_sql_success.yaml` 设置：1 epoch、有效 batch size 8、学习率 $10^{-6}$、$\beta=0.1$、梯度裁剪范数 1.0，并启用 gradient checkpointing。

### 3. 失败轨迹选择与分叉

来源为 `artifacts/sql_planner/reasoning/trajectories/` 的 7,000 条训练集轨迹：5,093 条正确、1,907 条失败。先在真实 Spider 数据库上检查 Gold SQL；1,904 条可构造，另 3 条因 Gold 无法执行或核验而隔离，记录 `block_reason=gold_not_executable`。原始轨迹保持不变。

`dpo/pairs.py` 按下列优先级选择分叉点：

1. 最后一次成功 `execute_sql`、但结果被核验为错误的观察之后：1,832 条。
2. 若不存在上述位置，优先选择失败的 `execute_sql` 观察之后；当前选择快照中该组为 0 条。
3. 若不存在上述位置，选择其他成功 SQL 观察之后：45 条，包含中间 SQL 正确、最终提交错误的情况。
4. 在未测试的最终提交之前：14 条；或在导致 `invalid_format` 的错误回复之前：13 条。

保存 `fork_turn`、`fork_reason`、源文件哈希和原始 rejected 后缀，核对两条分支的共同前缀一致。选择清单为 `artifacts/sql_planner/reasoning_dpo_all_errors_v1/selection.json`。

### 4. 构造 chosen 修复后缀

当前批次使用 `deepseek-flash`，开启 thinking，`reasoning_effort=high`。流程如下：

1. 在真实工具环境中回放共同前缀，检查观察漂移。
2. 保持普通 agent 的 system prompt；仅在教师续写请求中追加 Gold SQL 和修复要求。
3. 教师每轮输出原始 reasoning 与一个工具动作；工具结果由真实 SQLite 环境产生，继续沿用探索预算和 `submit_sql` 协议。
4. 要求最终提交与 Gold SQL 按实现的精确字符串规则一致，并由 `ExecutionVerifier` 执行完整结果核验；展示观察的截断不影响完整核验。
5. 保存学生可见的原始 reasoning、动作和真实观察；教师提示和 API 内部思考不进入学生消息。当前严格批次设置 `sanitize_hint_reasoning=False`，发现提示引用时交给审核拒绝，保留证据。

这些后缀由 Gold 引导生成。去掉教师提示后，仍需检查 reasoning 是否引用隐藏答案或续写指令，才能进入训练集。

### 5. 独立审核与接受条件

生成通过数据库核验的候选 pair 后，使用独立的 Flash high 请求进行盲审。审核员只收到学生可见的问题、共同前缀、修复后缀和机械检查结果，不收到 Gold SQL、教师提示或 API 内部思考。检查只针对分叉后的 chosen 后缀；原始 rejected 本身允许含有错误。

机械检查和语义审核共同检查：共同前缀一致性、记录一致性、最终 SQL 正确性、回复格式、提示引用、编造观察、错误描述观察、问题不匹配、缺乏依据的 schema 声明，以及 reasoning 与动作的矛盾。语义审核会识别 `The continuation instructs...` 等隐藏指令引用，不局限于 `hint`、`gold` 关键词。

提交前测试规则：最终提交 SQL 必须曾成功 `execute_sql`；共同前缀中的执行也可以计入。比较字符串时只去首尾空白，不规范化 SQL，因此大小写、内部空格或引号差异都可能触发 `untested_final_sql`。提交时探索预算已耗尽可例外，但仍禁止声称观察到不存在的验证结果。

审核返回 `accept`、`reject` 或 `uncertain`，附问题代码、逐字证据与回合位置；程序校验输出结构及证据。**仅 accept 合并进严格数据集**。数据库结果正确仍可能因轨迹质量问题被拒绝；模型审核可能漏检，接受结果不代表已经人工逐条检查。

实现与规则：`dpo/audit.py`、`docs/dpo_repair_audit_prompt.md`。

### 6. 首轮、追加尝试与断点续跑

首轮严格构造已完成：1,904 条可构造任务中，生成 1,885 对，19 条生成失败；审核接受 330 对、拒绝 1,555 对、uncertain 0 对。首轮完整轨迹生成最多尝试 3 次，得到数据库核验通过的 pair 后进行审核。

当前追加批次覆盖此前拒绝的 1,555 条和生成失败的 19 条，共 1,574 条任务。每题**最多追加两次完整轨迹生成并审核**，首次 accept 后停止；原有 330 对原样保留，按 `task_id` 去重后合并。回合格式重试、网络重试以及已保存 pair 的审核重试单独记录，不增加完整轨迹生成名额。

在 API 生成前持久化尝试记录；中断时未保存结果的生成占用本次名额，防止恢复后超过两次。已保存 pair 的审核异常只重试审核。manifest 固定源数据、原有 pair、提示和实现哈希；恢复时跳过已完成任务。

| 数据目录 | 用途 |
| --- | --- |
| `artifacts/sql_planner/reasoning_dpo_flash_high_strict_all_20261003_v1/` | 首轮严格生成、审核及已接受的 330 对 |
| `artifacts/sql_planner/reasoning_dpo_flash_high_strict_retry2_20261006_v1/` | 最多两次追加生成、审核、接受结果及进度 |
| 上述目录的 `attempts/01/`、`attempts/02/` | 开始记录、生成记录、pair 和审核证据 |
| 上述目录的 `report.json` | 当前权威计数与完成状态 |

本次工作交付目录为 `C:/Users/JSWang/Documents/Codex/2026-10-03/yu/outputs/`。首轮数据为 `dpo_pairs_strict.jsonl`；追加批次运行中导出 `dpo_pairs_strict_retry2_merged.partial.jsonl`，全部处理完成且无审核异常后导出 `dpo_pairs_strict_retry2_merged.jsonl`。相关报告为 `dpo_full_generation_report.md` 和 `dpo_retry2_report.md`。

最终结果（2026-10-06）：状态 `completed`，完成 1574/1574 条，新增接受 301 对，与原有 330 对合并为 **631 对不同任务的偏好数据**；待处理审核异常为 0。其余 1,273 条追加尝试后未获得合格数据。

### 7. 训练准备与当前限制

训练前按任务去重；若划分训练与验证集，整道题及其所有派生轨迹进入同一 split。使用目标 SFT 检查点的 tokenizer/chat template，核对 tokenization 和分叉 mask；超出 16,384 token 的分支由当前准备入口报错，不静默截断。

`dpo/prepare.py` 支持严格审核后的 `--pairs-jsonl`，并要求通过 `--audit-roots` 提供两个已完成批次的审核证据。程序逐条检查 JSONL 与 accepted 原件相同、审核为 accept、机械检查无问题、训练集任务不重复，并要求覆盖全部接受子集；原有 `--pairs-dir` 全量准备入口保留。输出 manifest 包含来源和审核哈希、每对长度与 loss token 数、相对 shard 路径及 SHA-256。`dpo/train.py` 加载时校验 shard 哈希；相对路径允许将整个 tokenized 目录搬到 GPU 机器。

在 GPU 机器使用实际 SFT 检查点的 tokenizer 重新准备数据。下例假设合并 JSONL 已复制到仓库的 `data/dpo/`，两个审核目录也已完整复制到 `artifacts/sql_planner/`：

```bash
python -m dpo.prepare \
  --pairs-jsonl data/dpo/dpo_pairs_strict_retry2_merged.jsonl \
  --audit-roots \
    artifacts/sql_planner/reasoning_dpo_flash_high_strict_all_20261003_v1 \
    artifacts/sql_planner/reasoning_dpo_flash_high_strict_retry2_20261006_v1 \
  --model artifacts/sql_planner/reasoning_balanced_2000_500_qwen25_coder_3b_sft/checkpoint_epoch_2 \
  --output-dir artifacts/sql_planner/reasoning_dpo_strict_631_v1/tokenized
python -m dpo.train score-reference --config configs/dpo_reasoning_sql_success.yaml
python -m dpo.train train --config configs/dpo_reasoning_sql_success.yaml
```

`dpo/train.py` 的参考模型和策略模型均从同一 SFT 检查点初始化，关闭 dropout，使分开计算分数和梯度的前向过程一致。DPO 和后续 GRPO 配置已指向 `reasoning_dpo_strict_631_v1`。运行前核对远端 SFT 检查点及配置的绝对路径；本批次 GPU DPO 训练尚未启动。

评估应记录 `fork_reason`、后缀长度、数据库执行正确率和提交前验证情况，并在固定验证集上比较 SFT、DPO 和后续 GRPO 检查点。
