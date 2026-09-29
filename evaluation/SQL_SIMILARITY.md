# SQL AST 相似度（C-TSED-SQL v3.1.0）

本模块只计算相似度，不构造 SFT、不调用模型、不执行预测或标准 SQL、不修改轨迹。

## 指标

给定预测 SQL、GOLD SQL 和同一数据库 schema，使用 SQLGlot 30.17.0 解析及绑定列，成对规范化可安全交换的内连接，运行其默认完整 `optimize()`，再构建保留值的比较树，用 APTED 1.0.3 计算单位代价的最小树编辑距离 d。完整顺序为：`parse(SQLite) → DQS 回退 → schema 校验/列绑定 → 成对规范化内连接 → optimize() → 比较树 → APTED`。优化规则以当前固定版本为准；输出中记录实际规则列表。

`similarity = max(0, 1 - d / max(pred_node_count, gold_node_count))`

insert、delete、replace 各为 1；操作针对一个节点。删除节点会把其子节点提升到父节点，整棵子树不按一次操作计费。距离对给定的规范化有序树精确；它不覆盖所有 SQL 语义等价关系。分数不是正确概率。

## 项目位置与依赖

主实现：`evaluation/sql_similarity.py`；测试：`tests/test_sql_similarity.py`。

在项目原有 Python 环境安装更新后的依赖：

```powershell
python -m pip install -e .
```

交付目录也包含相同源码的独立脚本 `sql_similarity.py`，可单独安装两个依赖运行：

```powershell
python -m pip install -r requirements_similarity.txt
```

## Python 调用

```python
from evaluation.sql_similarity import sql_similarity, schema_from_sqlite

schema = {"student": {"id": "INT", "name": "TEXT", "age": "INT"}}
result = sql_similarity(
    "SELECT name FROM student WHERE age > 18",
    "SELECT name FROM student WHERE age >= 18",
    schema,
)
assert result["status"] == "ok"
print(result["similarity"], result["ted"], result["edit_script"])

# 也可只读提取 SQLite 表、视图及列元数据：
# schema = schema_from_sqlite("database.sqlite")
```

schema 使用 `{表名: {列名: 类型字符串}}`。列顺序必须与真实数据库一致，因为 `SELECT *` 按它展开。元数据读取入口使用 UNKNOWN 类型；规范化不进行依赖静态类型的代数优化。

## 命令行

在项目根目录运行单对评分：

```powershell
python -m evaluation.sql_similarity --schema schema.json --pred "SELECT name FROM student WHERE age > 18" --gold "SELECT name FROM student WHERE age >= 18"
```

使用数据库元数据：

```powershell
python -m evaluation.sql_similarity --db database.sqlite --pred "SELECT name FROM student" --gold "SELECT age FROM student"
```

批量输入 JSONL，每行包含 `pred_sql` 和 `gold_sql`；可选 `task_id`、`attempt_id`、`db_id` 会原样传到输出。一个文件使用同一份 schema；跨数据库请按数据库分组，或在 Python 中逐对传入对应 schema。

```json
{"task_id":"example_1","pred_sql":"SELECT name FROM student","gold_sql":"SELECT age FROM student"}
```

```powershell
python -m evaluation.sql_similarity --schema schema.json --input pairs.jsonl --output scores.jsonl
```

独立脚本示例（在交付目录运行）：

```powershell
python sql_similarity.py --schema example_schema.json --input example_pairs.jsonl --output my_scores.jsonl
```

输出文件必须不存在，防止覆盖已有数据。`--include-trees` 导出比较树，`--max-nodes` 设置每棵比较树节点上限（默认 600）。不支持的输入或超限输入返回 `status=unscorable`，`similarity` 和 `ted` 为 null，包含错误阶段与原因。程序错误不会伪装成零分。

## 规范化范围

- 忽略格式、注释、无意义括号，按 SQLite ASCII 规则处理标识符大小写；字符串大小写保留。
- 通过作用域和关系实例解析别名，支持自连接、相关子查询、普通 CTE；递归 CTE 暂不支持。
- 两侧对应查询块都由同一组、互不重复的物理表构成，且仅含普通 INNER/CROSS JOIN 时，在展开星号后统一表顺序，并把简单 `ON` 谓词归入 `WHERE`，交由 `optimize()` 统一放置。表集合不同、外连接、`USING`/`NATURAL`、自连接、复杂 ON 和查询块带 `LIMIT/OFFSET` 时跳过该规则。
- 列是一个原子节点，保留其所属作用域、关系实例、列名；派生表列绑定到输出位置。
- 最终输出列的名字忽略，输出列顺序保留；ORDER BY 中的输出别名解析到其投影表达式。
- 确定性普通布尔谓词中的同类 AND/OR 展平、排序；含函数、子查询等结构时保留顺序。
- 普通常量 IN 列表排序，不删除重复项；保留字符串、数值的类别和值。
- 仅把“常量 比较符 普通列”反向为“普通列 反向比较符 常量”；不交换两列或显式 COLLATE 表达式。
- 展开星号；比较树保留优化后仍存在的 DISTINCT、聚合、JOIN 类型、WHERE/ON/HAVING 角色、排序和窗口属性、LIMIT/OFFSET、集合操作。
- SQLGlot 解析器的标量属性（例如排序方向和 CAST 的类型枚举）也计入比较树。

在建树前的 `optimize()` 会额外尝试谓词下推、JOIN/子查询/CTE 改写、类型标注、常量与布尔表达式简化等完整默认流程。例如 `age > 18 AND 1 = 1` 可简化为 `age > 18`，`age > 18 AND age > 18` 可去重。优化结果是比较输入，不能作为任意 SQL 的语义等价证明。

Spider 标准 SQL 中使用双引号字符串。默认 `sqlite_dqs=True` 模拟 SQLite 的传统 DQS 行为：原始双引号名称在可确定的可见 schema 中无法绑定时，按字符串处理，保留原始大小写；方括号及反引号名称不采用此回退。可用 `--strict-quotes` 或 `sqlite_dqs=False` 关闭，需与实际数据库连接设置一致。不确定的派生列绑定不猜测。

## 结果字段

- `status`、`similarity`、`ted`：可评分状态、相似度、最小编辑次数。
- `pred_node_count`、`gold_node_count`：比较树节点数，包括语义角色节点。
- `edit_counts`、`edit_script`：各类编辑次数和最优映射差异。每个编辑包含两侧节点标签、路径及区域。
- `changed_regions`：两侧发生编辑的查询块/子句区域，供审查使用。
- `pred_qualified_sql`、`gold_qualified_sql`：绑定完成、成对内连接规范化及完整优化器运行前的 SQL。
- `pred_optimized_sql`、`gold_optimized_sql`：运行 SQLGlot 完整 `optimize()` 后的 SQL。条件排序等额外处理发生在比较树构造阶段，所以此文本仍不等于比较树的规范字符串。
- `pred_tree_hash`、`gold_tree_hash`、`schema_hash`：比较树和 schema 指纹；schema 指纹包括列顺序。
- `metric_version`、`dependencies`、`optimizer.rules`、`sqlite_dqs`、`warnings`、`limitations`：复现信息和边界。
- `preoptimizer_comparison_orientation_conflict`：优化前两侧列与列比较的操作数顺序不同，需检查 SQLite collation。若优化后距离为 0，此类配对标记 `unscorable`，避免将可能不等价的 SQL 错判为零距离。

编辑明细给出一组最优节点映射，路径对应原始比较树/目标比较树，不能按输出顺序当作逐步修改 SQL 的补丁。最优映射可能不唯一；角色区域也受映射选择影响。

## 明确边界

完整 `optimize()` 会运行当前固定版本的默认重写规则；本模块不声明覆盖所有等价 SQL，也不忽略字面量，不把单库执行结果偶然相同视为零距离。GROUP BY 顺序、一般数值等价写法等未做额外归并。按当前解析器成功解析，不代表数据库一定能执行。

SQLite 中 `a=b` 和 `b=a` 在两列采用不同 collation 时可能产生不同结果，而 SQLGlot 当前优化器会把两式变成相同 AST。本模块在优化前保留列与列的有序比较指纹；若两侧指纹不同且优化后距离为 0，返回不可评分。指纹不同但距离大于 0 的记录保留分数并附警示。此检查覆盖所述操作数顺序问题，不能证明全部优化规则在所有 SQLite 边界上安全。

AND/OR 排序可消除纯顺序差异，但局部修改改变排序位置时会放大距离；增加查询块也可能改变后续查询块编号。比较树、规则或依赖改变后，应升级版本并重新校准阈值。该实现不输出自动收录 SFT 的结论。

## 验证与样例

`example_scores_v3.jsonl` 是 v3.0.0 的历史样例评分；旧 `example_scores.jsonl` 属于 v2 指标，都不用于 v3.1.0 阈值。`spider_similarity_sample.jsonl` 和 `optimizer_sample_report.json` 也是此前版本的兼容性记录。v3.1.0 的 200 对随机样本重新评分文件为 `spider_200_sql_similarity_v3_1.jsonl`，并保留 v3.0.0 文件供同批比较；没有修改原始轨迹，也没有重新评判正确性。

`similarity_sample_report.json` 记录样本数量、状态、耗时及源文件哈希复核。

## 依据

- TSED 归一化：https://aclanthology.org/2024.acl-short.3/
- APTED 最小距离及映射：https://github.com/JoaoFelipe/apted
- SQLite 双引号行为：https://www.sqlite.org/quirks.html#double_quoted_string_literals_are_accepted
- SQLGlot 默认规则：https://github.com/tobymao/sqlglot/blob/main/sqlglot/optimizer/optimizer.py

本实现采用 SQLGlot 比较树和项目定义的规范化规则，不能把所得数值视为原始 Tree-sitter TSED 实验的直接复现。
