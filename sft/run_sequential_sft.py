"""Train FP32 SFT and evaluate after each epoch, retaining the best checkpoint."""
from __future__ import annotations

import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from sft.train import train
from sft.train_config import SftTrainConfig

CONFIG = Path('configs/sft_sequential_base_fp32_2epoch.yaml')
SFT_DATA = Path('artifacts/sql_planner/reasoning_correct_only_2500_sft')


def main() -> None:
    config = SftTrainConfig.load(CONFIG)
    output = config.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'SFT output already exists: {output}')
    results = {}

    def evaluate(epoch, checkpoint, student, optimizer):
        gc.collect()
        torch.cuda.empty_cache()
        dest = output / f'eval_epoch_{epoch}'
        command = [sys.executable, '-u', '-m', 'evaluation.run_reasoning_sft',
                   '--model', str(checkpoint), '--dataset', str(SFT_DATA),
                   '--tasks', 'data/external_dev.jsonl',
                   '--env-config', 'configs/env_sql_planner_qwen25_coder_3b.yaml',
                   '--spider-root', '/root/autodl-tmp/sqlagent/datasets/spider/spider_data',
                   '--output-dir', str(dest), '--batch-size', '8', '--max-tokens', '512',
                   '--gpu-memory-utilization', '0.42', '--max-num-seqs', '8']
        print(f'SFT_EPOCH_{epoch}_EVAL_STARTED', flush=True)
        with (output / f'eval_epoch_{epoch}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((dest / 'summary.json').read_text())
        if result['completed'] != result['total'] or result['total'] != 1034:
            raise ValueError(f'SFT epoch {epoch} evaluation incomplete')
        results[epoch] = result['correct']
        print(f"SFT_EPOCH_{epoch}_ACCURACY={result['correct']}/1034={result['execution_accuracy']:.4%}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    summary = train(config, max_steps=None, save=True, epoch_callback=evaluate)
    if summary['status'] != 'completed' or set(results) != {1, 2}:
        raise RuntimeError('SFT training or evaluation incomplete')
    keep = max(results, key=lambda epoch: (results[epoch], -epoch))
    drop = 3 - keep
    shutil.rmtree(output / f'checkpoint_epoch_{drop}')
    record = {'epoch_1_correct': results[1], 'epoch_2_correct': results[2],
              'kept_epoch': keep, 'checkpoint': str((output / f'checkpoint_epoch_{keep}').resolve())}
    (output / 'best_checkpoint.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
