"""Recover the explicitly selected 87 correct submissions and 60 interrupted continuations."""
from __future__ import annotations

import concurrent.futures
import copy
import getpass
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from dpo.pairs import build
from dpo.pilot import RecordingClient
from experience_memory.evolve import read_json, write_json
from sft.curate_reasoning import EXTERNAL_REFERENCE
from sft.regenerate_reasoning import PROMPTS, trajectory_comparison, trajectory_profile
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.protocol import parse_tagged_response
from sql_agent.truncation import TokenCounter, canonical_json


def complete_candidate(task, source, checkpoint, config, client, group):
    """Replay saved evidence and execute only actual model responses."""
    student = copy.deepcopy(checkpoint['student_messages'])
    teacher = copy.deepcopy(checkpoint['private_teacher_messages'])
    turns = copy.deepcopy(checkpoint['turns'])
    response = checkpoint['private_api_calls'][-1]['message']['content']
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=student[0]['content'])
    try:
        env.reset(task)
        for i, turn in enumerate(turns):
            _, action = parse_tagged_response(turn['response'])
            actual, done = env.step(action)
            if done or canonical_json(actual) != canonical_json(turn['observation']):
                raise ValueError('Saved prefix observation differs from live replay')
            if student[2 + 2*i]['content'] != turn['response'] or student[3 + 2*i]['content'] != f"<observation>{canonical_json(actual)}</observation>":
                raise ValueError('Saved prefix messages differ from recorded tools')
        if len(student) != 2*len(turns)+2 or student[-1]['role'] != 'user':
            raise ValueError('Expected complete observed prefix before interrupted reply')
        while True:
            reasoning, action = parse_tagged_response(response)
            if action['tool'] == 'submit_sql':
                observation, _ = env.submit_sql(action['arguments'])
                if not observation.get('verification', {}).get('correct'):
                    raise ValueError('Recovered final submission does not match Gold results')
            else:
                if env.turn >= config.max_turns:
                    raise ValueError('Recovery exploration budget exhausted')
                observation, done = env.step(action)
                if done:
                    raise ValueError('Recovery terminated before final submission')
            student.append({'role': 'assistant', 'content': response})
            teacher.append({'role': 'assistant', 'content': response})
            turns.append({'turn': len(turns)+1, 'response': response, 'reasoning': reasoning,
                          'tool': action['tool'], 'arguments': action['arguments'], 'observation': observation})
            if action['tool'] == 'submit_sql':
                final_sql = action['arguments']['sql']
                break
            message = {'role': 'user', 'content': f'<observation>{canonical_json(observation)}</observation>'}
            student.append(message)
            teacher.append(dict(message))
            if group == 'correct_submission':
                raise ValueError('Expected existing correct submission, not a new generation')
            completion = client.complete_text(teacher, temperature=.3, max_tokens=8192)
            if completion.finish_reason != 'stop':
                raise ValueError('Recovery generation did not finish')
            response = completion.message.get('content', '')
    finally:
        env.close()
    counter = TokenCounter(config.tokenizer_path)
    comparison = trajectory_comparison(trajectory_profile(source['messages'], counter), trajectory_profile(student, counter))
    verdict = 'reject'
    static_rejection = any(EXTERNAL_REFERENCE.search(t['reasoning']) for t in turns)
    if not static_rejection:
        audit = client.complete_thinking_text([
            {'role': 'system', 'content': (PROMPTS/'external_answer_leak_audit_zh.md').read_text(encoding='utf-8')},
            {'role': 'user', 'content': json.dumps({'question': task.question, 'messages': student,
                                                   'trajectory_comparison': comparison}, ensure_ascii=False)}],
            max_tokens=32768)
        if audit.finish_reason != 'stop' or audit.message.get('content') not in ('accept', 'reject'):
            raise ValueError('Recovery audit must return exactly accept or reject')
        verdict = audit.message['content']
    return {'task_id': task.task_id, 'db_id': task.db_id, 'question': task.question, 'difficulty': task.difficulty,
            'split': task.split, 'correct': True, 'sft_eligible': verdict == 'accept', 'leak_audit': verdict,
            'trajectory_comparison': comparison, 'generation_mode': 'full_regeneration_from_question',
            'generation_thinking': False, 'messages': student, 'student_messages': student, 'turns': turns,
            'final_sql': final_sql, 'trainable_turn_numbers': list(range(1, len(turns)+1)),
            'recovery_group': group, 'submission_validation': 'live_execution_result_equivalence',
            'static_external_reference_rejection': static_rejection,
            'private_teacher_messages': teacher}


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root/'artifacts/dpo/nonthinking_full1907_20261012'
    output = root/'artifacts/dpo/recovery147_20261012'
    if output.exists():
        raise FileExistsError('Use a fresh recovery output; no automatic repeated attempts')
    analysis = read_json(parent/'失败原因复查.json')
    selected = {r['task_id']: 'correct_submission' for r in analysis['submission_cases'] if r['result_matches_gold']}
    selected.update({r['task_id']: 'continue_after_gold' for r in analysis['next_action_cases']})
    assert Counter(selected.values()) == {'correct_submission': 87, 'continue_after_gold': 60}
    tasks = {t.task_id: t for t in (TaskRecord.from_dict(json.loads(l)) for l in (parent/'tasks.jsonl').read_text(encoding='utf-8').splitlines())}
    entries = []
    for source_run in read_json(parent/'report.json')['source_runs']:
        run = Path(source_run['path'])
        manifest = read_json(run/'manifest.json')
        for e in manifest['sample']:
            if e['task_id'] not in selected:
                continue
            result = read_json(run/'results'/f"{e['task_id']}.json")
            if result.get('error') not in ('Submit the exact previously tested Gold SQL', 'Successful exact Gold execution must be followed by submission'):
                continue
            entries.append({**e, 'source_run': str(run), 'config': manifest['config'], 'group': selected[e['task_id']]})
    assert len(entries) == len({e['task_id'] for e in entries}) == 147
    output.mkdir()
    write_json(output/'manifest.json', {'count': 147, 'groups': dict(Counter(selected.values())), 'sample': entries,
               'generation_mode': 'nonthinking_existing_submission_or_prefix_continuation',
               'audit_thinking': 'high', 'audit_max_tokens': 32768, 'workers': 12,
               'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'audit_prompt_sha256': hashlib.sha256((PROMPTS/'external_answer_leak_audit_zh.md').read_bytes()).hexdigest()})
    key = getpass.getpass('API key (hidden): ')
    outcomes = []
    def one(entry):
        task_id = entry['task_id']
        run = Path(entry['source_run'])
        client = RecordingClient(key, 'deepseek-flash')
        client.timeout_seconds = 600
        source = read_json(run/'original/trajectories'/entry['file'])
        checkpoint_path = run/'attempts'/f'{task_id}.json'
        config_values = entry['config'].copy()
        for name in ('spider_root', 'processed_data_root', 'tokenizer_path'):
            config_values[name] = Path(config_values[name])
        config = replace(EnvConfig(**config_values), max_turns=entry['max_turns'])
        target = output/'attempts'/f'{task_id}.json'
        write_json(target, {'status': 'started', 'task_id': task_id})
        record = {}
        error = None
        try:
            assert hashlib.sha256((run/'original/trajectories'/entry['file']).read_bytes()).hexdigest() == entry['source_sha256']
            record = complete_candidate(tasks[task_id], source, read_json(checkpoint_path)['record'], config, client, entry['group'])
            record['recovery_source'] = {'path': str(checkpoint_path), 'sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
            outcome = 'accepted' if record['sft_eligible'] else 'audit_rejected'
            if record['sft_eligible']:
                write_json(output/'regenerated/trajectories'/entry['file'], {k:v for k,v in record.items() if not k.startswith('private_')})
        except Exception as exc:
            outcome = 'recovery_failed'
            error = str(exc).replace(key, '[redacted]')
        write_json(output/'original/trajectories'/entry['file'], source)
        write_json(target, {'status': 'completed', 'task_id': task_id, 'outcome': outcome, 'error': error,
                           'record': record, 'private_api_calls': client.calls})
        usage = Counter()
        for call in client.calls:
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens'):
                usage[k] += int(call['usage'].get(k, 0))
        result = {'task_id': task_id, 'group': entry['group'], 'outcome': outcome, 'error': error,
                  'trajectory_comparison': record.get('trajectory_comparison'), 'usage': dict(usage)}
        write_json(output/'results'/f'{task_id}.json', result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, e) for e in entries]):
            outcomes.append(future.result())
            progress = {'completed': len(outcomes), 'total': 147, 'counts': dict(Counter(r['outcome'] for r in outcomes))}
            write_json(output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    config_values = entries[0]['config'].copy()
    for name in ('spider_root', 'processed_data_root', 'tokenizer_path'):
        config_values[name] = Path(config_values[name])
    config = EnvConfig(**config_values)
    paired = {'pairs': 0, 'skipped': []}
    if any(r['outcome'] == 'accepted' for r in outcomes):
        paired = build(output/'original', output/'regenerated', [tasks[e['task_id']] for e in entries], config,
                       output/'pairs.jsonl', workers=4, strict_submission=False)
    write_json(output/'report.json', {'status': 'completed', 'sample_size': 147,
               'groups': dict(Counter(selected.values())), 'counts': dict(Counter(r['outcome'] for r in outcomes)),
               'retained_dpo_pairs': paired['pairs'], 'pairing_skipped': paired['skipped'],
               'errors': dict(Counter(r['error'] for r in outcomes if r['error'])),
               'usage': dict(sum((Counter(r['usage']) for r in outcomes), Counter()))})
    print(json.dumps(read_json(output/'report.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
