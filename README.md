# SQL Agent：SFT、DPO、GRPO 与记忆自进化

基于 Spider 1.0 的多轮 Text-to-SQL 项目。当前主线是：构造完整推理与工具轨迹 → SFT → 可选 DPO / GRPO → 冻结最终模型权重 → 只追加经学生重试验证的经验记忆。

## 实测结果

基础模型为 **Qwen2.5-Coder-3B-Instruct**。以下结果来自顺序执行的 SFT → DPO → 标准 Binary GRPO，均为全参数更新；不是随机交替 SFT/DPO 实验，也未启用 GRPO 方法 2 的引导轨迹注入。

评测使用 Spider 官方开发集对应的 `data/external_dev.jsonl`，共 **1034 题**，不是 `internal_validation` 或 `internal_holdout`。三个阶段使用相同的任务文件、system prompt 和 `ExecutionVerifier`，不检索经验记忆：temperature=0、每轮最多生成 512 tokens、上下文上限 16384 tokens，最多 10 次探索，另保留一次提交；探索预算耗尽时可由评测器提交最后执行的 SQL。

| 阶段 | 训练数据与更新次数 | 正确题数 | 执行准确率 | 相对上一阶段 |
|---|---|---:|---:|---:|
| SFT | 3000 条正确轨迹，2 个 epoch | 741/1034 | **71.66%** | — |
| SFT → DPO | 1131 对长度筛选后的完整轨迹，2 个 epoch | 745/1034 | **72.05%** | +0.39 个百分点 |
| SFT → DPO → GRPO | 500 道新训练题，125 步，每题采样 6 条轨迹 | 765/1034 | **73.98%** | +1.93 个百分点 |

分难度正确题数：

| 难度 | 题数 | SFT | SFT → DPO | SFT → DPO → GRPO |
|---|---:|---:|---:|---:|
| easy | 248 | 223 | 227 | 224 |
| medium | 446 | 334 | 326 | 337 |
| hard | 174 | 110 | 109 | 124 |
| extra | 166 | 74 | 83 | 80 |

本次实验配置：

- **SFT**：从原 7000 条轨迹中的正确样本分层抽取 3000 条，easy/medium/hard/extra 分别为 600/1200/600/600；抽样 seed=42，逐条重新执行 SQL 核验。峰值学习率 `1e-5`，5 步 warmup 后余弦衰减，有效 batch 为 64 个 assistant 回合，共 444 步。
- **DPO**：从上述 SFT 权重开始，冻结参考模型同样初始化为该 SFT 权重。学习率 `1e-6`、β=0.1、每次更新累计 4 对，共 566 步；分支分数累加完整轨迹的 assistant token log-probability，不按长度平均。
- **GRPO**：从上述 DPO 权重开始，从原 7000 题中另抽 500 题，easy/medium/hard/extra 分别为 100/200/100/100。题号及同库同问题均与 SFT、DPO 数据不重叠；每步 4 题、每题 6 条轨迹，500 题各使用一次，共 3000 条在线轨迹。学习率 `1e-6`，temperature=1、top_p=0.99，采用 clip-higher：`clip_ratio=0.2`、`clip_ratio_high=0.28`，概率比裁剪区间为 `[0.8, 1.28]`；`kl_beta=0`，不加载 KL 参考模型。未启用 Gold 引导注入。

三个阶段的训练数据库与 `external_dev` 数据库不重叠。训练源使用官方 `train_spider.json` 的 7000 题范围，不能据此推定这些 checkpoint 与预处理得到的内部验证/保留集也没有重叠。本次标准 GRPO 实测使用本地实验脚本将同题 6 条轨迹并行生成，维护 FP32 主权重并导出 BF16 checkpoint；该并行封装不是 `grpo.train` 原生命令的默认采样方式。

结果依据以下本地产物核对，`artifacts/` 不纳入 Git，也不随代码分发：

- `artifacts/experiments/sft3000_dpo1131_2epoch_20261012T060640Z/accuracy.json`
- `artifacts/experiments/grpo500_restart_batched_20261013T004639Z/accuracy.json`

下文配置文件是可编辑的运行模板，并非上述实测配置的完整复现文件。

## 项目结构

| 目录 | 用途 |
|---|---|
| `sql_agent/` | 只读 SQLite 环境、工具、执行验证器、公共提示词与解析器、DeepSeek 客户端 |
| `sft/` | 原始轨迹采集、Gold 辅助完整重生成、筛选、学生 tokenizer 转换和 SFT 训练 |
| `dpo/` | 完整轨迹偏好配对、固定样本生成、特定实验恢复、参考分数缓存与 DPO 训练 |
| `grpo/` | 在线 Binary GRPO、reasoning/tool rollout、可选全错组引导轨迹注入 |
| `experience_memory/` | 通用经验检索、教师反思、学生重试、成功入库和多轮续跑 |
| `evaluation/` | 学生推理与评测、基线对照和历史评测辅助工具 |
| `data/`、`configs/` | 数据划分和运行配置 |
| `artifacts/` | 本地生成数据、记忆库、模型和实验记录，不纳入 Git |

公共环境、协议和执行验证位于 `sql_agent/`；完整轨迹 DPO 位于 `dpo/`。历史 `artifacts/sql_planner/` 仅作为本地数据产物路径使用。DPO 目标公式和详细说明见 [完整轨迹 DPO](docs/dpo_full_trajectory.md)。

## 环境和数据划分

```bash
pip install -e '.[train,dev]'
python -m data.preprocess_spider --config-path configs/env.yaml
```

先修改 `configs/env.yaml` 中的 `spider_root`、`processed_data_root` 和 `tokenizer_path`，指向实际数据库、输出目录与学生 tokenizer。GPU 推理入口另外需要与运行环境匹配的 vLLM。

预处理按数据库划分 `train`、`internal_validation`、`internal_holdout`，外部开发集记为 `external_dev`。SFT 构造和记忆新增只使用权威任务文件中标为 `train` 的题目。验证集用于模型选择，最终保留集用于泛化评测；不从保留集补足训练配额。

历史采集可能覆盖整个官方训练文件，其中部分数据库已被当前划分留作验证或保留集。新构造器以 `--tasks data/train.jsonl` 为准，跳过其他来源；评估已有 checkpoint 时还需核对其实际训练覆盖范围。

## 统一交互协议

每次助手回复为一个非空 `<reasoning>...</reasoning>`，随后一个 `<tool>{"name":"...","arguments":{...}}</tool>`。可调用 `list_tables`、`inspect_tables`、`inspect_values`、`execute_sql`、`submit_sql`；真实工具反馈以 `<observation>` 返回。

默认最多 10 次探索，另保留一次最终提交。初始只提供问题，由模型通过工具发现 schema。数据库只读，执行观察受行数与 token 限制；最终评分重新执行完整 SQL。执行成功不等于回答正确，空结果也不自动代表错误。

公共协议位于 `sql_agent/protocol.py`。从头重生成时保留来源轨迹的原 system 提示词，避免悄悄改变已确认的 SFT 条件。

## SFT 数据构造

构造器支持两类完整轨迹：

1. 原始执行正确轨迹：保留真实 reasoning、工具调用和 observation。
2. 原始错题的 Gold 辅助完整重生成：先验证 Gold 可执行，从原问题重新开始，不复用错误前缀，也不从分叉点续写。

教师重生成通过 DeepSeek 客户端调用 `--model` 指定的模型，默认 thinking enabled、reasoning effort high。生成时保留原 system 提示词，在教师上下文中追加 `sft/prompts/gold_repair_supplement_zh.md`；Gold 与补充提示不进入学生消息。私有 `reasoning_content` 只在教师调用间重放并单独归档。

重生成必须真实执行并提交所提供 Gold，两个工具参数中解码后的 SQL 字符串与 Gold 完全一致，包括空白、别名和分号。成功执行该 SQL 后，下一轮提交相同 SQL。不能靠字符串替换或编造 observation 得到正确轨迹。

独立审核输入包含原问题、学生可见消息及 `trajectory_comparison` 统计对比，不包含教师私有 Gold 补充。审核须仅输出 `accept` 或 `reject`。只保存执行验证通过、审核为 `accept` 的候选；拒绝后重新生成，不能删词或改写 reasoning 来包装合格数据。审核存在漏检可能，`accept` 只代表通过该次审核。

教师补充和审核提示保存在 `sft/prompts/`。以下命令会调用 API；密钥仅从环境变量 `DEEPSEEK_API_KEY` 读取，不写入仓库。

### 采集原始轨迹（已有原始数据可跳过）

```bash
python -m sft.collect_reasoning \
  --tasks data/train.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --model YOUR_DEEPSEEK_MODEL \
  --output-dir artifacts/sft/original --limit 100
```

原始采集不向教师提供 Gold。可通过 `--samples-per-task` 采集多个候选，正确性由程序执行验证。

### 对原始错题完整重生成

```bash
python -m sft.regenerate_reasoning \
  --source artifacts/sft/original \
  --tasks data/train.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --model deepseek-flash \
  --output-dir artifacts/sft/gold_regenerated \
  --attempts 3 --limit 100
```

保存各次尝试和私有教师记录，正式 `trajectories/` 仅写审核通过的学生可见轨迹。同一参数支持复用已有接受结果和尝试记录；失败候选不混入正式源数据。

### 筛选、去重和执行复核

```bash
python -m sft.curate_reasoning \
  --original artifacts/sft/original \
  --regenerated artifacts/sft/gold_regenerated \
  --tasks data/train.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --output artifacts/sft/curated
```

构造器优先选择通过检查的原始正确轨迹，同题只保留一个示例；再用合格的完整重生成补充覆盖。检查学生消息与动作一致、真实工具重放结果一致、最终 SQL 已成功执行、最终结果正确，以及重生成的审核标记。完整重生成另外要求最后执行后立即提交相同 Gold。源数据不改写，拒绝原因写入 `audit.json`。

这里的质量筛选和 SFT 同题去重只作用于训练数据构造。记忆库仍只追加，不做语义合并、修订或停用。

## 转换与 SFT 训练

```bash
python -m sft.reasoning_trajectories \
  --source artifacts/sft/curated --model /path/to/student-base-model \
  --output artifacts/sft/dataset --max-sequence-tokens 16384
python -m sft.train --config configs/sft_reasoning.yaml
python -m sft.train_status --config configs/sft_reasoning.yaml
```

先修改 `configs/sft_reasoning.yaml` 中的学生模型路径。使用学生 tokenizer 和 chat template 转换，完整轨迹的所有 assistant reasoning/action 轮参与监督；system、user、工具 observation 不计算目标 loss。现有训练器按轨迹的监督 token 数归一化，避免长轨迹仅因长度获得更大权重。

旧分叉 Gold 修复的后缀监督规则不进入新数据主线。数据转换拒绝旧 `gold_repair` 后缀记录，并对序列长度、提示词一致性和训练划分做检查。SFT 阶段更新模型权重；训练后可继续 DPO/GRPO，再冻结最终 checkpoint 用于评测或记忆自进化。

## 完整轨迹 DPO（可选阶段）

DPO 使用同一 system prompt 和原问题组成偏好对，训练完整多轮 assistant 轨迹：

- `rejected`（`y_l`）：原始采集时未提供 Gold 的错误轨迹。配对时重新执行最终 SQL，已正确的样本不作为负例；缺少最终 SQL 时，仅接受原始 `invalid_format` 失败。
- `chosen`（`y_w`）：复用 SFT 的 Gold 辅助重生成，从原问题开始，保留真实 reasoning、工具动作和执行反馈；要求最终执行正确且外部答案泄露审核为 `accept`。

两条分支的工具路径、轮数和 observation 可以不同；教师 Gold 提示、私有 `reasoning_content` 和 API 记录不进入训练消息。审核还检查相对原轨迹的 assistant token 数、工具次数和推理风格变化，统计及差值保存在 `trajectory_comparison`。只处理匹配的训练任务，保留来源路径及 SHA-256，并将配对拒绝原因写入 `.report.json`。审核通过表示满足当前审核规则，仍需抽查数据。

### 构造偏好对

以下命令在仓库根目录执行；先替换 Spider、模型和 tokenizer 路径，并配置 `configs/env.yaml`。原始目录须包含 `trajectories/*.json`。生成和审核会调用 DeepSeek API，常规入口从环境变量 `DEEPSEEK_API_KEY` 读取密钥。

```bash
# 1. 从原始错题重新生成 chosen，复用 SFT 验证和审核。
python -m sft.regenerate_reasoning \
  --source artifacts/sft/original --tasks data/train.jsonl \
  --env-config configs/env.yaml --spider-root /path/to/spider_data \
  --model deepseek-flash --output-dir artifacts/dpo/gold_regenerated

# 2. 回放 chosen 的工具轨迹并复核 rejected，保存完整偏好对。
python -m dpo.pairs \
  --original artifacts/sft/original --regenerated artifacts/dpo/gold_regenerated \
  --tasks data/train.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --output artifacts/dpo/pairs.jsonl
```

常规重生成默认 thinking enabled/high，每题最多 3 次尝试；已有接受结果会跳过。配对默认采用严格提交验证：成功执行与 Gold 字符串完全相同的 SQL 后，立即提交相同 SQL，并验证完整结果。

固定随机样本实验可使用 `dpo.pilot`：

```bash
python -m dpo.pilot prepare \
  --source artifacts/sft/original --spider-root /path/to/spider_data \
  --tokenizer /path/to/sft-checkpoint --env-config configs/env.yaml \
  --output artifacts/dpo/pilot_100 --count 100 --seed 42
python -m dpo.pilot run \
  --output artifacts/dpo/pilot_100 --model deepseek-flash \
  --generation-mode disabled --temperature 0.3 \
  --workers 4 --validation-workers 4 --max-tokens 8192
```

`prepare` 固定样本、源文件哈希和运行配置；它按原轨迹的 `split=train` 选题，再从 Spider 原始文件恢复任务，因此应先核对来源划分与评测集隔离。`run` 支持 enabled/disabled 两种生成模式，审核仍使用 thinking/high，每题仅一次生成尝试。已完成结果可复用，已有尝试但缺少结果的任务记为 `interrupted_attempt`，不自动增加尝试次数。结果、私有 API 记录、token 用量和保留比例分别写入运行目录；密钥缺失时交互输入。

### 历史 DPO 恢复脚本

`dpo/recover.py` 是针对固定历史实验路径的恢复脚本，依赖对应的失败复查文件和 API 密钥，不是通用中断续训入口。它回放已有工具前缀，复核 observation 和最终 SQL，并进行独立泄露审核；配对时可使用 `strict_submission=False`，以真实执行结果一致性代替 Gold 字符串及紧邻执行要求。常规 `dpo.pairs` CLI 仍使用严格验证。详见 [完整轨迹 DPO](docs/dpo_full_trajectory.md)。

### Tokenization、参考缓存与训练

```bash
# 3. 使用目标 SFT checkpoint 的 tokenizer 构造两条独立完整分支。
python -m dpo.prepare \
  --pairs artifacts/dpo/pairs.jsonl --model /path/to/sft-checkpoint \
  --output artifacts/dpo/tokenized --max-tokens 16384

# 4. 从同一 SFT checkpoint 缓存冻结参考模型分数。
python -m dpo.train reference \
  --checkpoint /path/to/sft-checkpoint --dataset artifacts/dpo/tokenized \
  --reference-dir artifacts/dpo/reference --device cuda

# 5. 全参数 DPO 训练。
python -m dpo.train train \
  --checkpoint /path/to/sft-checkpoint --dataset artifacts/dpo/tokenized \
  --reference-dir artifacts/dpo/reference --output artifacts/dpo/run1 \
  --beta 0.1 --epochs 1 --batch-size 4 --learning-rate 1e-6 \
  --gradient-checkpointing --device cuda
```

仅 assistant 的 reasoning、工具动作和消息结束 token 参与 DPO 分数；system、问题和工具观察作为上下文。分支分数为所有 assistant token 的 log-probability 之和，不按长度平均；超出上下文限制会报错，不静默截断。

参考模型和策略从同一 SFT checkpoint 初始化，参考分数先缓存，训练时仅加载策略模型。训练器先计算完整分支分数，再逐回合重算并反传 DPO 梯度，关闭 dropout；`--batch-size` 表示每次更新累计的偏好对数。缓存与训练核对 tokenizer、数据分片哈希及 checkpoint 文件清单、大小和修改时间，缓存完成后应保持 checkpoint 不变。

配对、tokenization、参考缓存和训练要求新的输出路径。当前训练支持单设备、梯度检查点和梯度裁剪，不提供多卡或中断续训入口；输出 `metrics.jsonl`、训练 `manifest.json` 和最终 `checkpoint/`。完成 DPO 后再冻结最终权重进入记忆自进化，DPO 训练本身不调用记忆新增流程。

## GRPO（在线强化学习）

```bash
python -m grpo.prepare --config configs/grpo_reasoning_tool_standard.yaml
python -m grpo.train --config configs/grpo_reasoning_tool_standard.yaml --max-steps 128
python -m grpo.status --config configs/grpo_reasoning_tool_standard.yaml
```

运行前修改配置中的 `student_model`、`reference_model`、`train_data`、`output_dir`，并设置 `configs/env.yaml` 的实际数据路径。训练命令必须显式传入 `--max-steps`，不能超过配置中的 `max_steps`。

标准 Binary GRPO 在线采样多轮轨迹，按 SQL 执行正确性产生 0/1 奖励，计算组内标准化优势；一组全部正确或全部错误时，标准二值奖励的优势为零。损失作用于模型生成的回复 token，屏蔽提示、工具观察和特殊 token；使用新旧策略概率比的逐 token 裁剪，可配置非对称上界 `clip_ratio_high`。标准模式不向策略注入 Gold。

`kl_beta=0` 时不计算 KL，也不加载参考模型；配置校验仍要求 `reference_model` 路径存在，可与 `student_model` 相同。支持全参数更新、显存不足时降低 micro-batch、按 `checkpoint_interval` 保存，以及 `--resume-from` 恢复权重和步数。当前恢复会重新创建 AdamW 优化器，不恢复其动量状态。原生采样基于 Transformers `model.generate()`，不依赖 veRL。

GRPO 会更新权重，因此如需使用它，应先完成 GRPO，再冻结最终 checkpoint 进行记忆自进化。当前 GRPO rollout 不检索经验记忆，记忆自进化也不调用 GRPO 训练。原有 JSON action 版配置和代码仍保留。

新增原生方法 2 配置 `configs/grpo_method2.yaml`，继续使用本目录的 Transformers/PyTorch GRPO，不依赖 veRL。全错组由冻结旧学生复用 SFT/DPO 的 Gold 补充提示最多生成 3 次，只追加第一条执行正确且独立审核 accept 的轨迹。生成来源概率进入分母，训练输入排除私有引导。通过 `DEEPSEEK_API_KEY` 提供审核 Key；运行方式与验证边界见 [方法 2 说明](docs/grpo_method2.md)。

## 冻结权重的记忆自进化

每轮使用同一学生 checkpoint 和冻结的记忆快照：

```text
已有记忆检索 → 学生首次推理 → 执行评测
  首次答错 → 教师看错误轨迹与 Gold → 一条候选通用经验
  新 episode：原问题 + 旧记忆 + 候选经验 → 学生重新推理
  首次错、重试对 → 追加记忆；其他情况只留实验记录
下一轮使用追加后的记忆，模型权重保持不变
```

候选经验不提前进入正式库。写入时读取两次记录，要求正确性严格分别为 false 和 true，再重新执行两条 SQL 验证。重试须由学生实际提交，评测器兜底提交不满足入库条件。超时、Gold 异常和没有最终 SQL 的失败跳过。

记忆字段包括：`memory_id`、`question`、`experience`、`sql_before`、`sql_first_execute`、`sql_after`、`source_task_id`、`source_split`、`initial_record_path`、`retry_record_path`。`question` 保存完整来源问题原文，`experience` 只保存标题含义、适用条件和改进建议；默认只编码 `question` 召回，检索上下文仍包含原题和经验；评测入口还支持按经验文本或 SQL 检索，提供 embedding、TF-IDF 和 SQL BM25 选项。不设数据库专属条目。修改后 SQL 来自学生成功重试；`sql_first_execute` 保存首次实际执行的 SQL，无执行记录时为 null，旧记录可缺少此字段。

不在记忆条目中保存正确性标记、模型名、记忆版本、检索 ID、时间、轮次或候选 ID。必要的模型配置和验证证据独立保存在实验记录中。

默认检索使用 `Qwen/Qwen3-Embedding-0.6B` 对待解问题和记忆的 `question` 编码，按归一化向量的余弦相似度召回。命中后提供正文及前后 SQL 示例，提醒映射当前 schema。默认最多 5 条、仅取正相似度，记忆上下文预算默认 8192 tokens。Embedding 权重冻结，默认 CPU、batch size 8、编码长度上限 2048 tokens；超长文本编码时截断。首次非空检索下载模型，离线可用 `--embedding-model /path/to/Qwen3-Embedding-0.6B`，GPU 可用 `--embedding-device cuda:0`。记忆不自动修订、停用或合并；同一证据对的重复处理仅作续跑幂等保护。

```bash
python -m experience_memory.evolve \
  --model /path/to/frozen-checkpoint --dataset artifacts/sft/dataset \
  --tasks data/train.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --teacher-model YOUR_DEEPSEEK_MODEL \
  --memory-db artifacts/experience_memory/memories.sqlite \
  --output-dir artifacts/experience_memory/run_001 \
  --rounds 1 --limit 100 --memory-top-k 5 --memory-max-tokens 8192
```

整轮结束后追加成功经验，下一轮再使用。详细记录、续跑方式和限制见 `docs/experience_memory.md`。

## 评测与效果边界

```bash
python -m evaluation.run_reasoning_sft \
  --model /path/to/frozen-checkpoint --dataset artifacts/sft/dataset \
  --tasks data/internal_holdout.jsonl --env-config configs/env.yaml \
  --spider-root /path/to/spider_data --output-dir artifacts/evaluation/with_memory \
  --memory-db artifacts/experience_memory/memories.sqlite
```

不传 `--memory-db` 即无记忆评测，使用另一个输出目录保存对照。保留集和外部开发集只评分，不用其 Gold 生成记忆。模型选择使用 validation，不反复用最终保留集挑 checkpoint。

分别报告首次正确率、错题重试修复率、新增记忆数、未见题执行准确率，以及有/无记忆的错→对、对→错和成本。原题重试成功仅证明满足入库规则，不能单独证明经验的因果作用或泛化收益。单库执行结果一致也可能包含偶然正确；更强语义验证需额外 test-suite。

## 验证

```bash
pytest -q
```

CPU 测试包含实际 SQLite 工具与执行验证；模型生成和教师请求使用脚本化替身。GPU SFT、GRPO、vLLM 和真实 DeepSeek 调用需在对应运行环境验证，本仓库不以 CPU 单元测试代表真实模型效果。
