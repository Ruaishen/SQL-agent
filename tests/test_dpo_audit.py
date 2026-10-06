import json

import pytest

from dpo.audit import audit_input, validate_review


def action(name, sql='SELECT 1'):
    return '<reasoning>I will check the requested count.</reasoning><tool>' + json.dumps(
        {'name': name, 'arguments': {'sql': sql}}) + '</tool>'


def pair(sql='SELECT 1', remaining=8):
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'Count entries.'},
                {'role': 'assistant', 'content': action('execute_sql')},
                {'role': 'user', 'content': '<observation>' + json.dumps(
                    {'status': 'success', 'rows': [[1]], 'turns_remaining': remaining}) + '</observation>'},
                {'role': 'assistant', 'content': action('submit_sql', sql)}]
    return {'chosen_messages': messages, 'rejected_messages': messages,
            'prompt_messages': messages[:4], 'fork_turn': 2,
            'chosen_verification': {'correct': True},
            'chosen_turns': [{'response': messages[4]['content'], 'tool': 'submit_sql', 'arguments': {'sql': sql}}]}


def test_prefix_test_counts_and_future_results_are_not_exposed():
    data = audit_input(pair())
    assert data['mechanical_checks']['issues'] == []
    assert data['mechanical_checks']['exact_final_sql_tested_before_submit']
    assert data['turns'][1]['observation_after_action'] is None
    assert 'gold' not in json.dumps(data)


def test_untested_sql_and_exhausted_budget_exception():
    assert audit_input(pair('SELECT 2'))['mechanical_checks']['issues'] == ['untested_final_sql']
    assert audit_input(pair('SELECT 2', remaining=0))['mechanical_checks']['issues'] == []


def test_reviewer_cannot_invent_quote_or_use_future_evidence():
    data = audit_input(pair())
    review = {'verdict': 'reject', 'summary': '说明不足', 'issues': [{
        'code': 'reasoning_action_mismatch', 'turn': 2,
        'quote': 'I will check the requested count.', 'explanation': 'test', 'evidence_turns': [1]}]}
    assert validate_review(review, data) == review
    review['issues'][0]['quote'] = 'invented quote'
    with pytest.raises(ValueError, match='verbatim'):
        validate_review(review, data)
    review['issues'][0]['quote'] = 'I will check the requested count.'
    review['issues'][0]['evidence_turns'] = [2]
    with pytest.raises(ValueError, match='precede'):
        validate_review(review, data)
