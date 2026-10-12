"""Offline DPO tests; no API, database or pretrained checkpoint required."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dpo.pairs import make_pair
from dpo.prepare import tokenize_branch
from dpo.train import branch_score, dataset_manifest, dpo_loss, turn_logp


def examples():
    base = [{"role": "system", "content": "agent"}, {"role": "user", "content": "count"}]
    loser = {"task_id": "t", "question": "count", "db_id": "d", "split": "train", "correct": False,
             "messages": base + [{"role": "assistant", "content": "wrong"}], "final_sql": "SELECT 0"}
    winner = {**loser, "correct": True, "messages": copy.deepcopy(base) + [{"role": "assistant", "content": "correct"}],
              "generation_mode": "full_regeneration_from_question", "leak_audit": "accept", "sft_eligible": True, "final_sql": "SELECT 1"}
    return loser, winner


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}!" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) % 128 for c in text])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.linear = torch.nn.Linear(8, 128)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.linear(self.embedding(input_ids)))


class DpoTests(unittest.TestCase):
    def test_pair_is_full_trajectory(self):
        loser, winner = examples()
        pair = make_pair(loser, winner)
        self.assertEqual(pair["prompt_messages"], loser["messages"][:2])
        self.assertEqual(pair["chosen_messages"], winner["messages"])
        self.assertNotIn("fork_turn", pair)

    def test_bad_pairs_rejected(self):
        for mutation in (lambda l, w: w.update(leak_audit="reject"),
                         lambda l, w: l.update(teacher_used_gold_hint=True),
                         lambda l, w: l.update(correct=True),
                         lambda l, w: w["messages"][0].update(content="different"),
                         lambda l, w: w["messages"][-1].update(reasoning_content="private")):
            loser, winner = copy.deepcopy(examples())
            mutation(loser, winner)
            with self.assertRaises(ValueError):
                make_pair(loser, winner)

    def test_observations_condition_but_are_not_scored(self):
        messages = examples()[0]["messages"] + [{"role": "user", "content": "observation"}, {"role": "assistant", "content": "final"}]
        turns = tokenize_branch(CharacterTokenizer(), messages, 1000)
        self.assertEqual(len(turns), 2)
        self.assertEqual([int(t["action_mask"].sum()) for t in turns], [len("wrong!"), len("final!")])
        self.assertGreater(turns[1]["input_ids"].numel(), turns[0]["input_ids"].numel())

    def test_context_overflow_rejected(self):
        with self.assertRaises(ValueError):
            tokenize_branch(CharacterTokenizer(), examples()[0]["messages"], 3)

    def test_dpo_direction_and_initial_loss(self):
        initial = dpo_loss(torch.tensor(-20.), torch.tensor(-30.), -20., -30., 0.1)
        better = dpo_loss(torch.tensor(-19.), torch.tensor(-30.), -20., -30., 0.1)
        self.assertAlmostEqual(float(initial), 0.69314718, places=6)
        self.assertLess(float(better), float(initial))

    def test_recomputed_turn_gradients_equal_direct_dpo(self):
        torch.manual_seed(7)
        model = TinyModel()
        chosen = tokenize_branch(CharacterTokenizer(), examples()[1]["messages"], 1000)
        rejected = tokenize_branch(CharacterTokenizer(), examples()[0]["messages"], 1000)
        win = branch_score(model, chosen, "cpu")
        lose = branch_score(model, rejected, "cpu")
        ref_w, ref_l, beta = win - 2, lose + 1, 0.1
        direct = dpo_loss(sum(turn_logp(model, t, "cpu") for t in chosen),
                          sum(turn_logp(model, t, "cpu") for t in rejected), ref_w, ref_l, beta)
        direct.backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        coefficient = beta * torch.sigmoid(torch.tensor(-beta * (win - lose - ref_w + ref_l))).item()
        for sign, turns in ((-1, chosen), (1, rejected)):
            for turn in turns:
                (sign * coefficient * turn_logp(model, turn, "cpu")).backward()
        for p, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, gradient, atol=1e-6, rtol=1e-5)

    def test_modified_shard_rejected(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            shard = root / "00000.pt"
            shard.write_bytes(b"original")
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "score_mode": "sum_all_assistant_tokens", "pair_count": 1, "shards": {shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()}}))
            dataset_manifest(root)
            shard.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                dataset_manifest(root)


if __name__ == "__main__":
    unittest.main()
