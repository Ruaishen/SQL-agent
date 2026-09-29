# SQL Agent RL

基于 Spider 1.0 的多轮 Text-to-SQL 项目。项目保留两条训练路径：成功轨迹 SFT 和 Binary GRPO；`sql_agent/` 提供共同的只读 SQLite 环境与执行验证器，`evaluation/` 提供重新评测所需的代码。

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
