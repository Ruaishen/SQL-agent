# 完整轨迹 DPO：无 Gold 原始轨迹与 Gold 引导轨迹

## 数据定义

每题取同一 system prompt 和原问题 $x$：

- $y_l$：原始采集中没有 Gold SQL 提示的错误完整轨迹。
- $y_w$：DeepSeek API 看到教师专用 Gold SQL 后，从原问题重新生成的完整轨迹；最终提交通过真实执行核验，公开轨迹通过外部答案泄露审核。

两条轨迹只要求 system 和问题相同，工具路径、轮数及 observation 可以不同。没有共同错误前缀，也没有分叉后的局部损失。

## 生成、验证和审核直接复用 SFT

`sft/regenerate_reasoning.py` 的 `regenerate()` 已实现整个流程：

1. 保留原始 system prompt，从问题重新开始。教师请求额外包含 `sft/prompts/gold_repair_supplement_zh.md` 和 Gold SQL；学生消息不保存这些额外内容。
2. 教师调用 DeepSeek thinking/high，生成公开 reasoning 与工具动作，真实工具环境返回观察。
3. 当前 SFT 约束要求成功执行与 Gold 精确字符串一致的 SQL，紧接着提交相同 SQL；最终提交还必须通过数据库完整结果验证。
4. 检查公开 reasoning 中明显的外部答案来源引用，再以独立 API 请求审核学生可见消息。审核使用 `sft/prompts/external_answer_leak_audit_zh.md`，判断外部答案来源引用或依赖，并检查相对原始轨迹的长度、工具次数和风格差异是否明显且无任务必要性，输出 `accept/reject`。生成侧提供原始统计与风格片段；审核侧提供两条轨迹的统计、差值和比值。总长度按同一 tokenizer 对全部公开 assistant 回复内容计数，包含 reasoning、工具动作及其格式，不含工具观察、system、问题、教师提示和私有思考。
5. 仅保存审核为 accept 的公开轨迹；Gold 提示和私有 `reasoning_content` 保存在私有尝试记录中，不进入学生训练消息。审核拒绝后重新生成，原始 reasoning 不作删改。

`dpo/pairs.py` 将 SFT 接受轨迹与同任务的原始错误轨迹配对，并复用 `sft.curate_reasoning.validate_record()` 再次回放工具与核验 chosen。原始最终 SQL 也会重新核验，若已经正确则不标成 loser。原始格式失败轨迹可作为 loser。只处理权威 `train` 任务，同任务重复来源会报错。

“无 Gold”依据原始采集来源及公开消息检查；仅凭消息内容无法证明上游模型从未获得隐藏信息，应使用普通 `sft.collect_reasoning` 采集目录作为 original。

## DPO 目标

对于每条分支，只有 assistant 产生的 reasoning、工具调用和消息结束 token 计入分数；system、问题、工具观察参与上下文，但不计分。

$$
S_\theta(y\mid x)=\sum_{t\in\text{assistant tokens}}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

$$
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma\left(\beta\left[
S_\theta(y_w\mid x)-S_\theta(y_l\mid x)
-S_{\mathrm{ref}}(y_w\mid x)+S_{\mathrm{ref}}(y_l\mid x)
\right]\right).
$$

`dpo/prepare.py` 使用与 SFT 相同的按 assistant 回合构造上下文和 action mask 的方式。每条分支的观察历史独立；分数是所有 assistant token 的对数概率之和，不按长度平均，不静默截断超长数据。

冻结参考模型和策略模型均从指定 SFT checkpoint 开始。先缓存参考分数，之后只加载策略模型训练。`dpo/train.py` 用完整分支分数计算 DPO 梯度系数，再逐回合重算并反传，避免同时保留两条完整轨迹的计算图。关闭 dropout，确保两次前向一致。reference checkpoint 在缓存与训练之间须保持不变；文件清单、大小和修改时间会核对。

## 运行步骤

在仓库根目录运行，先配置环境、Spider 数据和 SFT checkpoint。以下占位路径需替换为实际目录，API 密钥从 `DEEPSEEK_API_KEY` 环境变量读取。

```powershell
# 1. 直接使用现有 SFT 入口构造并审核 y_w。
python -m sft.regenerate_reasoning --source artifacts/sft/original --tasks data/train.jsonl --env-config configs/env.yaml --spider-root "<Spider目录>" --model "<DeepSeek模型名>" --output-dir artifacts/dpo/gold_regenerated

# 2. 将已接受 y_w 与原始 y_l 配对；已有 SFT 重生成目录也可直接使用。
python -m dpo.pairs --original artifacts/sft/original --regenerated artifacts/dpo/gold_regenerated --spider-root "<Spider目录>" --output artifacts/dpo/pairs.jsonl

# 3. 用目标 SFT checkpoint 的 tokenizer 准备两条完整分支。
python -m dpo.prepare --pairs artifacts/dpo/pairs.jsonl --model "<SFT checkpoint>" --output artifacts/dpo/tokenized

# 4. 缓存冻结参考分数。
python -m dpo.train reference --checkpoint "<SFT checkpoint>" --dataset artifacts/dpo/tokenized --reference-dir artifacts/dpo/reference

# 5. DPO 全参数训练。
python -m dpo.train train --checkpoint "<SFT checkpoint>" --dataset artifacts/dpo/tokenized --reference-dir artifacts/dpo/reference --output artifacts/dpo/run1 --beta 0.1 --epochs 1 --batch-size 4 --learning-rate 1e-6 --gradient-checkpointing
```

`--batch-size` 是累计偏好对数，每次优化后才更新模型。生成默认每题最多 3 次尝试，已保存接受结果会跳过；配对、准备、缓存和训练要求新输出路径以避免覆盖。训练当前支持单设备，不含多卡和中断续训入口。

代码与测试：`dpo/pairs.py`、`dpo/prepare.py`、`dpo/train.py`、`tests/test_dpo_full_trajectory.py`。本次实现未调用付费 API，也未训练实际 checkpoint。
