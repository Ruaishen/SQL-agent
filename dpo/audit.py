"""Audit raw paired repair suffixes, preserving originals and accepted pairs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from sql_planner.collect import _atomic_json
from sql_planner.collect_tagged import parse_response
from sql_planner.deepseek import DeepSeekClient

CODES = {'hint_reference', 'invented_observation', 'observation_mismatch',
         'question_mismatch', 'unsupported_schema', 'reasoning_action_mismatch',
         'format_error', 'untested_final_sql', 'invalid_pair'}


def audit_input(pair: dict) -> dict:
    """Reconstruct precisely the student's visible ordered observations."""
    messages = pair['chosen_messages']
    turns = []
    for index in range(2, len(messages), 2):
        message = messages[index]
        reasoning, action = parse_response(message['content'])
        observation = None
        if index + 1 < len(messages):
            content = messages[index + 1]['content']
            if messages[index + 1]['role'] != 'user' or not content.startswith('<observation>') or not content.endswith('</observation>'):
                raise ValueError('Unexpected student observation format')
            observation = json.loads(content[len('<observation>'):-len('</observation>')])
        turns.append({'turn': len(turns) + 1, 'reasoning': reasoning,
                      'response': message['content'], 'action': action,
                      'observation_after_action': observation})
    final = turns[-1]
    if final['action']['tool'] != 'submit_sql':
        raise ValueError('No final submission')
    sql = final['action']['arguments']['sql']
    tested = any(t['action']['tool'] == 'execute_sql'
                 and t['action']['arguments']['sql'].strip() == sql.strip()
                 and (t['observation_after_action'] or {}).get('status') == 'success'
                 for t in turns[:-1])
    last_observation = next((t['observation_after_action'] for t in reversed(turns[:-1])
                             if t['observation_after_action'] is not None), {})
    budget_exhausted = last_observation.get('turns_remaining', 10) <= 0
    fork = pair['fork_turn']
    records_match = all(t['response'] == s['response']
                        and t['action'] == {'tool': s['tool'], 'arguments': s['arguments']}
                        for t, s in zip(turns[fork - 1:], pair['chosen_turns']))
    records_match = records_match and len(turns[fork - 1:]) == len(pair['chosen_turns'])
    prefix_match = messages[:2 * fork] == pair['rejected_messages'][:2 * fork] == pair['prompt_messages']
    verified = bool(pair['chosen_verification'].get('correct'))
    issues = []
    if not (records_match and prefix_match and verified):
        issues.append('invalid_pair')
    if not tested and not budget_exhausted:
        issues.append('untested_final_sql')
    return {'question': messages[1]['content'], 'fork_turn': fork, 'turns': turns,
            'mechanical_checks': {'exact_final_sql_tested_before_submit': tested,
                                  'budget_exhausted_at_submit': budget_exhausted,
                                  'chosen_sql_database_verified': verified,
                                  'shared_prefix_matches': prefix_match,
                                  'chosen_record_matches': records_match,
                                  'issues': issues}}


def validate_review(value: dict, data: dict) -> dict:
    if set(value) != {'verdict', 'issues', 'summary'} or value['verdict'] not in {'accept', 'reject', 'uncertain'}:
        raise ValueError('Invalid review schema')
    issues = value['issues']
    if not isinstance(issues, list) or bool(issues) != (value['verdict'] != 'accept'):
        raise ValueError('Verdict and issues disagree')
    turns = {t['turn']: t for t in data['turns']}
    for issue in issues:
        if set(issue) != {'code', 'turn', 'quote', 'explanation', 'evidence_turns'} or issue['code'] not in CODES:
            raise ValueError('Invalid issue schema')
        turn = turns.get(issue['turn'])
        if turn is None or issue['turn'] < data['fork_turn']:
            raise ValueError('Issue must refer to the repair suffix')
        if not isinstance(issue['quote'], str) or not issue['quote'] or issue['quote'] not in turn['response']:
            raise ValueError('Issue quote is not verbatim')
        if not isinstance(issue['evidence_turns'], list) or any(not isinstance(n, int) or n not in turns or n >= issue['turn'] for n in issue['evidence_turns']):
            raise ValueError('Evidence must precede the reviewed turn')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/sql_planner/dpo_repair_audit_200_v1'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    groups = {'old_prompt': Path('artifacts/sql_planner/dpo_chosen_hint_compare_100_v2'),
              'new_prompt': Path('artifacts/sql_planner/dpo_flash_high_base_system_100_v3')}
    prompt_path = Path('docs/dpo_repair_audit_prompt.md')
    prompt = prompt_path.read_text(encoding='utf-8')
    jobs = []
    inputs = []
    for group, root in groups.items():
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['sample']:
            path = root / 'flash_high/pairs' / entry['file']
            raw = path.read_bytes()
            inputs.append({'group': group, 'task_id': entry['task_id'], 'file': entry['file'],
                           'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()})
            jobs.append((group, entry, json.loads(raw)))
    assert len(jobs) == 200
    identity = {'protocol': 'blind_flash_high_audit_v1', 'model': 'deepseek-flash',
                'thinking': 'enabled', 'reasoning_effort': 'high', 'max_tokens': 8192,
                'prompt_sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs': inputs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / 'manifest.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Audit identity changed')
    _atomic_json(identity_path, identity)
    key = os.environ['DEEPSEEK_API_KEY']

    def run(group, entry, pair):
        dest = args.output_dir / 'reviews' / group / entry['file']
        if dest.exists():
            return
        data = audit_input(pair)
        client = DeepSeekClient(key, model='deepseek-flash', timeout_seconds=240, max_retries=2)
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]
        calls = []
        review = None
        started = time.monotonic()
        for attempt in range(3):
            result = client._request({'model': client.model, 'messages': messages,
                                      'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
                                      'max_tokens': 8192, 'response_format': {'type': 'json_object'}, 'stream': False})
            calls.append({'message': result.message, 'usage': result.usage,
                          'request_id': result.request_id, 'api_model': result.model,
                          'finish_reason': result.finish_reason})
            try:
                if result.finish_reason == 'length':
                    raise ValueError('Truncated review')
                review = validate_review(json.loads(result.message['content']), data)
                break
            except (ValueError, TypeError, KeyError) as exc:
                messages.append({'role': 'user', 'content': f'上次 JSON 输出验证失败：{exc}。重新审查并输出符合规定的 JSON；不要修改待审轨迹。'})
        if review is None:
            raise ValueError(f'No valid review for {group}/{entry["task_id"]}')
        codes = set(data['mechanical_checks']['issues']) | {i['code'] for i in review['issues']}
        decision = ('uncertain' if review['verdict'] == 'uncertain' and not data['mechanical_checks']['issues']
                    else 'reject' if codes else 'accept')
        record = {'task_id': entry['task_id'], 'group': group, 'file': entry['file'],
                  'review': review, 'decision': decision, 'issue_codes': sorted(codes),
                  'mechanical_checks': data['mechanical_checks'], 'api_calls': calls,
                  'elapsed_seconds': time.monotonic() - started}
        _atomic_json(dest, record)
        if decision == 'accept':
            _atomic_json(args.output_dir / 'accepted' / group / 'pairs' / entry['file'], pair)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index % 10 == 0:
                print(f'audited={index}/200', flush=True)
    rows = []
    summary = {}
    usage = Counter()
    for group in groups:
        records = [json.loads(p.read_text(encoding='utf-8'))
                   for p in (args.output_dir / 'reviews' / group).glob('*.json')]
        assert len(records) == 100
        rows.extend(records)
        summary[group] = {'total': len(records), 'decisions': dict(Counter(r['decision'] for r in records)),
                          'issue_counts': dict(Counter(code for r in records for code in r['issue_codes'])),
                          'only_untested_final_sql': sum(r['issue_codes'] == ['untested_final_sql'] and r['review']['verdict'] != 'uncertain' for r in records)}
        for record in records:
            for call in record['api_calls']:
                usage.update({k: int(call['usage'].get(k, 0)) for k in ('prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens')})
    accepted_ids = {r['task_id'] for r in rows if r['decision'] == 'accept'}
    report = {'status': 'completed', 'total': 200, 'groups': summary,
              'accepted_pairs': sum(r['decision'] == 'accept' for r in rows),
              'accepted_unique_task_ids': len(accepted_ids), 'usage': dict(usage),
              'note': 'LLM judgment plus mechanical checks; uncertain is excluded. Rejection issue counts overlap. Original pairs are unchanged.'}
    _atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
