import math
import torch
from evaluation.sql_token_entropy import entropy_from_logits, final_sql_turn, sql_token_indices


def test_full_entropy_uniform_and_concentrated():
    assert torch.allclose(entropy_from_logits(torch.zeros(2, 4)), torch.full((2,), math.log(4)))
    assert entropy_from_logits(torch.tensor([[100., 0., 0.]])).item() < 1e-30


def test_sql_span_excludes_reasoning_and_handles_json_escapes():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(chr(i) for i in ids)
    text = '<reasoning>SELECT is uncertain.</reasoning><tool>{"name":"submit_sql","arguments":{"sql":"SELECT \\"x\\""}}</tool>'
    indices = sql_token_indices(Tokenizer(), list(map(ord, text)), 'SELECT "x"')
    assert ''.join(text[i] for i in indices) == 'SELECT \\"x\\"'


def test_fallback_uses_actual_model_sql_not_evaluator_text():
    generated = {'arguments': {'sql': 'SELECT 1'}, 'generated_token_ids': [1]}
    fallback = {'arguments': {'sql': 'SELECT 1'}, 'source': 'evaluator_fallback'}
    assert final_sql_turn({'final_sql': 'SELECT 1', 'turns': [generated, fallback]}) is generated
    assert final_sql_turn({'final_sql': None, 'turns': []}) is None


def test_execute_entropy_is_token_weighted_and_counts_repeated_calls():
    from evaluation.execute_sql_entropy import aggregate
    rows = [dict(task_id='same', sql_token_count=1, entropy_sum_nats=1., mean_token_entropy_nats=1.),
            dict(task_id='same', sql_token_count=3, entropy_sum_nats=6., mean_token_entropy_nats=2.)]
    result = aggregate(rows)
    assert result['calls'] == 2 and result['tasks'] == 1
    assert result['mean_per_token_entropy_nats'] == 1.75
    assert result['mean_per_call_token_entropy_nats'] == 1.5
