"""Measure each execute_sql call's full-vocabulary SQL entropy and Gold correctness."""
import argparse
import json
import math
import time
from pathlib import Path

from evaluation.sql_token_entropy import score_sql_turn
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.verifier import ExecutionVerifier


def aggregate(rows):
    tokens = sum(r['sql_token_count'] for r in rows)
    entropy = sum(r['entropy_sum_nats'] for r in rows)
    return {'calls': len(rows), 'tasks': len({r['task_id'] for r in rows}),
            'sql_token_count': tokens,
            'mean_per_token_entropy_nats': entropy / tokens if tokens else None,
            'mean_per_token_entropy_bits': entropy / tokens / math.log(2) if tokens else None,
            'mean_per_call_token_entropy_nats': sum(r['mean_token_entropy_nats'] for r in rows) / len(rows) if rows else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.perf_counter()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    assert 'memory' not in manifest and manifest['record_token_ids']
    tasks = {t.task_id: t for t in load_tasks(Path(manifest['dataset']))}
    config = EnvConfig.from_yaml(args.run_dir / 'env.yaml')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda').eval()
    path = args.run_dir / 'execute_sql_token_entropy.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['task_id'], r['turn_index']) for r in rows}
    records = [json.loads(p.read_text()) for p in sorted((args.run_dir / 'trajectories').glob('*.json'))]
    assert len(records) == 1034
    executed_keys = {(r['task_id'], i) for r in records for i, t in enumerate(r['turns'])
                     if t.get('tool') == 'execute_sql' and 'observation' in t}
    rows = [r for r in rows if (r['task_id'], r['turn_index']) in executed_keys]
    for row in rows:
        record = next(r for r in records if r['task_id'] == row['task_id'])
        row['execute_index'] = sum(t.get('tool') == 'execute_sql' and 'observation' in t
                                   for t in record['turns'][:row['turn_index'] + 1])
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    done = {(r['task_id'], r['turn_index']) for r in rows}
    expected, excluded = 0, []
    cache = {}
    with torch.inference_mode(), path.open('a') as output:
        for record in records:
            task = tasks[record['task_id']]
            verifier = ExecutionVerifier(task.resolve_db_path(config.spider_root), task.reference_sql, config)
            execution_index = 0
            for index, turn in enumerate(record['turns']):
                if turn.get('tool') != 'execute_sql':
                    continue
                if 'observation' not in turn:
                    excluded.append({'task_id': task.task_id, 'turn_index': index,
                                     'reason': 'Generated execute_sql request was not executed'})
                    continue
                expected += 1
                execution_index += 1
                if (task.task_id, index) in done:
                    continue
                assert 'generated_token_ids' in turn and 'prompt_token_ids' in turn
                sql = turn['arguments']['sql']
                key = (task.db_id, task.reference_sql, sql)
                if key not in cache:
                    cache[key] = verifier.verify(sql).to_dict()
                verification = cache[key]
                if str(verification.get('error') or '').startswith('verifier_reference_'):
                    raise RuntimeError(f'Invalid Gold SQL for {task.task_id}: {verification}')
                row = {'task_id': task.task_id, 'db_id': task.db_id, 'difficulty': task.difficulty,
                       'turn_index': index, 'execute_index': execution_index, 'sql': sql,
                       'correct': verification['correct'], 'verification': verification,
                       'execution_status': turn.get('observation', {}).get('status'),
                       'final_task_correct': record['correct']}
                row.update(score_sql_turn(model, tokenizer, turn, sql))
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(json.dumps({'execute_sql_scored': len(rows), 'elapsed_seconds': time.perf_counter()-started}), flush=True)
    assert len(rows) == expected and len({(r['task_id'], r['turn_index']) for r in rows}) == expected
    summary = {'method': 'Full vocabulary Shannon entropy, original unscaled logits (temperature=1), exact recorded prompt and generated IDs, BF16 forward/FP32 entropy',
               'scope': 'SQL content tokens of EVERY execute_sql call; each repeated call counted separately; reasoning and tool wrapper excluded',
               'correctness': 'ExecutionVerifier result compared to Gold SQL of THIS task, independent of final submission correctness; execution errors included as incorrect',
               'total_tasks': len(records), 'execute_sql_calls': expected,
               'excluded_unexecuted_requests': excluded,
               'tasks_with_execute_sql': len({r['task_id'] for r in rows}),
               'groups': {}, 'incorrect_breakdown': {}, 'first_execute_groups': {},
               'wall_seconds': time.perf_counter() - started}
    classifiers = {
        'correct_sql': lambda r: r['correct'],
        'executable_wrong_result': lambda r: not r['correct'] and r['verification']['agent_sql_executable'],
        'execution_error': lambda r: not r['correct'] and not r['verification']['agent_sql_executable'],
    }
    for name, predicate in classifiers.items():
        group = [r for r in rows if predicate(r)]
        summary['groups'][name] = aggregate(group)
        summary['first_execute_groups'][name] = aggregate([r for r in group if r['execute_index'] == 1])
    summary['incorrect_breakdown']['executable_wrong_result'] = aggregate([r for r in rows if not r['correct'] and r['verification']['agent_sql_executable']])
    summary['incorrect_breakdown']['invalid_or_execution_error'] = aggregate([r for r in rows if not r['correct'] and not r['verification']['agent_sql_executable']])
    (args.run_dir / 'execute_entropy_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
