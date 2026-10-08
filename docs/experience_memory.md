# 冻结权重的 SQL 经验记忆

模型权重始终固定。学生先使用当前记忆解题；答错后，DeepSeek 查看问题、schema、
错误轨迹和 Gold SQL，生成一条通用经验。学生新开 episode，使用原问题、旧记忆和
候选经验重新推理，只有首次失败且重试成功才追加记忆。

不进行停用、自动修订或合并。相同经验可以来自不同案例；同一对证据文件重复处理
不会重复入库。这是续跑幂等保护，不是语义去重。

## 记忆字段

SQLite 的 `memories` 表存放 JSON 记录，仅包含：

| 字段 | 含义 |
|---|---|
| `memory_id` | 从来源题目和证据路径生成的稳定唯一编号 |
| `question` | 完整来源问题原文，与经验独立保存 |
| `experience` | 只包含标题含义、适用条件、建议和检查方法组成的通用经验 |
| `sql_before` | 学生首次失败时的最终 SQL |
| `sql_first_execute` | 原始失败轨迹中第一次实际 `execute_sql` 的 SQL，包含执行报错；没有执行则为 `null` |
| `sql_after` | 学生重试成功时实际提交的 SQL |
| `source_task_id` | 来源题目编号 |
| `source_split` | 来源划分，生成记忆必须为 `train` |
| `initial_record_path` | 首次推理和评测证据的绝对路径 |
| `retry_record_path` | 重试推理和评测证据的绝对路径 |

不保存正确性标记、学生/教师模型名、记忆版本、检索 ID、时间、轮次和候选 ID。
模型配置和评测正确性仍保存在独立实验记录中，用于续跑和验证，不属于记忆条目。

## 入库规则

入库时读取首次和重试证据，要求 `correct` 严格分别为 JSON `false`、`true`。
随后重新执行 Gold、首次 SQL 和重试 SQL，要求 Gold 可评测、首次 SQL 不正确、
重试 SQL 正确。`sql_after` 来自学生重试，不能用 Gold 替换。

重试必须由学生实际调用 `submit_sql`；评测器强制提交不算入库成功。
缺少最终 SQL 的失败没有修改前示例，本版跳过。超时、结果超限或 Gold 异常也跳过。

通过原题重试只是入库资格，不能证明因果收益或其他题目的泛化收益。

## 检索与推理

检索只使用冻结的 `Qwen/Qwen3-Embedding-0.6B`（0.6B 参数、1024 维、中英文支持），
通过现有 `[train]` 依赖中的 Transformers / PyTorch 加载，不需要额外检索库。
默认仅编码记忆的 `question`，不编码 `experience` 或前后 SQL；待解问题加英文检索任务指令后编码。问题向量与经验文本分离，召回后仍向学生提供完整原题、经验和前后 SQL。
使用左侧 padding、最后 token 的 hidden state 和 L2 归一化，向量点积即余弦相似度。
每轮记忆快照编码一次，问题分批编码，相同分数按记忆写入顺序排序。
默认最多 5 条，记忆上下文预算默认 8192 tokens，只取正相似度；未设置经过验证的相关性阈值，语义向量的正值不保证适用。
不会在推理过程中再次检索。模型加载或编码失败会报错，不回退到词汇匹配。

默认 `--embedding-model Qwen/Qwen3-Embedding-0.6B --embedding-device cpu`，
`--embedding-batch-size 8 --embedding-max-length 2048`。
首次非空检索自动下载到 Hugging Face 缓存；离线可传模型本地目录。
CPU 使用 FP32，GPU 使用 FP16，可通过 `--embedding-device cuda:0` 显式启用。
0.6B 权重在 FP32 下约 2.4 GB、FP16 下约 1.2 GB，运行还需激活及其他内存。
编码超过 2048 tokens 的文本会截断，注入的原始经验及 SQL 不因此改变。
模型和编码配置写入运行清单，变更配置需使用新输出目录；运行清单使用 `embedding_question_v1` 标识题目向量检索。
默认题目检索要求独立的非空 `question` 字段；旧记忆需先拆分字段。旧实验可用 `--memory-retriever embedding_experience` 继续按旧 `experience` 文本编码。向量只在运行内缓存。
模型官方说明：https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

检索命中后注入经验以及前后 SQL 示例，提醒学生将历史表名、字段名映射到当前 schema。
预算按学生 tokenizer 计算；整条经验及例子无法放入预算时跳过，不截断 SQL。
候选经验重试时优先保留；如与旧记忆一起超出预算，本版省略旧记忆。

学生接收已有经验和候选建议，教师的 Gold、分析输入及完整输出不直接注入学生。
教师提示要求不输出具体题目答案或完整 SQL；通用性依赖教师生成质量，需抽查。

## 运行一轮或多轮

在安装 vLLM、Transformers 且能加载模型的环境执行。教师模型名必须显式指定。
API key 仅通过运行时环境变量 `DEEPSEEK_API_KEY` 提供，不写入配置或代码。

```bash
python -m experience_memory.evolve \
  --model /path/to/fixed-checkpoint \
  --dataset artifacts/sft/dataset \
  --tasks data/train.jsonl \
  --env-config configs/env.yaml \
  --spider-root /path/to/spider_data \
  --teacher-model YOUR_DEEPSEEK_MODEL \
  --memory-db artifacts/experience_memory/memories.sqlite \
  --output-dir artifacts/experience_memory/run_001 \
  --rounds 1 --limit 100 --batch-size 16 \
  --memory-top-k 5 --memory-max-tokens 8192
```

也可使用安装后的 `sql-agent-memory-evolve` 命令。

每轮先写 `memory_snapshot.json`，整个轮次使用同一快照；所有重试结束后才追加成功经验。
下一轮读取更新后的库。相同输出目录支持续跑，增加 `--rounds` 可以继续后续轮次。
运行参数不匹配会拒绝复用已有记录。轮次存在错误时停止进入后续轮次，修复后续跑。

实验目录结构：

```text
run_001/
  run_manifest.json
  round_001/
    round_manifest.json
    memory_snapshot.json
    initial/       # 学生首次轨迹，含实际初始消息和工具返回
    teacher/       # 候选经验、教师输入输出、token 用量
    retry/         # 学生重试轨迹
    errors/        # 可重试的 API、解析、入库等错误
    summary.json
```

证据文件名使用 task ID 的 SHA-256，避免路径字符问题。
命令使用独立 SQLite 运行锁阻止同目录并发写入，进程退出后锁自动释放。
保留实验目录，记忆中的证据路径需要这些文件。

## 在已有评测入口使用记忆

```bash
python -m evaluation.run_reasoning_sft \
  --model /path/to/fixed-checkpoint \
  --dataset artifacts/sft/dataset \
  --tasks data/internal_holdout.jsonl \
  --env-config configs/env.yaml \
  --spider-root /path/to/spider_data \
  --output-dir artifacts/experience_memory/holdout_eval \
  --memory-db artifacts/experience_memory/memories.sqlite \
  --memory-top-k 5 --memory-max-tokens 8192
```

不传 `--memory-db` 即原来的无记忆推理。该评测命令只读取记忆，不反思或新增条目。
正式泛化评测使用未参与模型训练和记忆构建的题目，Gold 仅供评分。

## 验证

CPU 测试使用脚本化学生和模拟教师，但 SQL 入库验证实际执行 SQLite。
覆盖精简字段、严格真假检查、伪造成功 SQL 拒绝、训练划分限制、强制提交拒绝、
检索预算、Gold 输入边界、重试成功/失败、整轮快照和续跑幂等。
真实模型质量及 DeepSeek 端到端调用需在模型运行环境另行验证。

## TF-IDF 评测对照

`evaluation.run_reasoning_sft` 支持 `--memory-retriever tfidf`，默认仍为 embedding。TF-IDF 只对冻结记忆的 `experience` 建立词表和 IDF（历史版本该字段包含原题，分离后的新版本为纯经验），不加载 embedding 模型。英文按小写单词、中文按单字切分，使用 unigram、原始词频、平滑 IDF `1 + log((1+N)/(1+df))` 和 L2 归一化余弦相似度。并列分数保持记忆写入顺序，零分不召回；仍遵守前 5 和记忆 token 预算。前后 SQL 仅在召回后提供给学生，不参与打分。

## 首次成功执行 SQL 后检索

当前评测使用 `--memory-retriever embedding_second_sql_every_turn`：第一次 `execute_sql` 不召回；从第二次实际执行开始，每次执行后均以本次 SQL 检索记忆 `sql_before`，报错也触发，取前 5 条正分记忆供下一轮使用。其他工具调用不增加 SQL 执行序号。每次更新当前记忆上下文，移除之前直接注入的记忆文本，保留历史 SQL、工具结果和模型回答，避免重复堆积超出上下文窗口。逐次召回记录保存在 turns 和 memory_retrievals 中，包含 SQL、记忆 ID、相似度、上下文与执行序号。

单次首次执行召回仍可用 `--memory-retriever embedding_first_sql`，包括 SQL 报错的情况。

`--memory-retriever tfidf_first_sql` 使用同样的首次实际 `execute_sql` 触发时机，每题只召回一次，初始输入不带记忆，SQL 报错也触发。TF-IDF 词表和 IDF 只根据冻结记忆中的 `sql_before` 拟合，查询是该次 SQL 原文；不使用原题、经验或修改后 SQL 打分，也不加载 embedding 模型。仍使用小写单词/CJK 单字 unigram、原始词频、平滑 IDF 和 L2 余弦相似度，只保留正分，并列按写入顺序。两种首次执行召回方式均可用 `--memory-top-k 5` 或 `10` 控制数量，召回后向学生提供完整记忆内容，遵守相同 token 预算。每轮评测保存独立的记忆快照、SQL、召回 ID、分数和注入上下文。

评测可用 `--memory-retriever embedding_first_successful_sql`。开始推理时不注入记忆；当 `execute_sql` 第一次返回 `status=success` 后，用其原始 SQL 与记忆 `sql_before` 的向量计算余弦相似度，召回前 5 条正分记忆，将其附在本次执行观察后，供下一轮使用。执行报错不触发，后续成功执行也不重复检索；未成功执行 SQL 的题不检索。触发依据是工具执行成功，不依赖 Gold 或答题正确性。查询使用 SQL 专用检索指令，记忆 SQL 原文编码；模型需在推理过程中与学生同时驻留，启动脚本为其预留显存。逐题记录包含 query_sql、memory_ids、scores 和实际注入上下文。


## BM25 首次执行 SQL 召回

`--memory-retriever bm25_first_sql --memory-top-k 10` 在第一次实际执行
`execute_sql` 后召回一次，包括执行报错。查询为该次 SQL 原文，索引仅包含冻结记忆
的 `sql_before`，不使用问题、经验或修改后 SQL 打分，也不加载 embedding 模型。
分词与 SQL TF-IDF 相同：小写英文单词/数字/下划线、中文单字 unigram。
采用 Okapi BM25，`k1=1.5`、`b=0.75`，IDF 为
`log(1 + (N-df+0.5)/(df+0.5))`；每个不同查询词只贡献一次。
只保留正分，并列按记忆写入顺序。BM25 分数未归一化，不直接与余弦分数比较。
召回后的上下文格式和 token 预算与其他首次执行 SQL 召回实验相同。


评测的 vLLM `max_model_len` 跟随环境 YAML 的 `max_context_tokens`，并将两者记录在运行 manifest 中。设置 `max_context_tokens: 32768` 可使用当前 SFT 模型配置支持的 32K 上下文；记忆预算仍由 `--memory-max-tokens` 独立控制。Top 15/20 的 32K 实验均使用 16384 的记忆预算、相同冻结记忆和首次执行 SQL 召回规则。


`sql_first_execute` 由原始失败轨迹提取；未实际执行的工具请求不算，缺失时不以最终 SQL 替代。新入库自动保存该字段。旧格式仍可读取且不会在读取时自动修改；显式补充当前记忆库可运行 `python -m experience_memory.backfill_first_sql --memory-db /path/to/memories.sqlite --jsonl /path/to/memories.jsonl`。评测冻结快照保持原样；补充字段本身不改变召回索引或学生提示。


使用 `--memory-retriever tfidf_first_sql --memory-sql-field sql_first_execute --memory-top-k 10`，可在当前首次实际执行 SQL 后，检索历史失败轨迹的首次执行 SQL。缺少该字段或字段为 `null` 的记忆不参加召回，不回退到最终错误 SQL。manifest 记录实际索引数量与排除数量；召回后展示的原题、经验和前后最终 SQL 格式保持一致。
