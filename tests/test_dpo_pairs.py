from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from dpo.pairs import fork_info, make_pair, select


def _source():
    prompt = [{"role": "system", "content": "p"}, {"role": "user", "content": "q"}]
    wrong = "<reasoning>try</reasoning><tool>{}</tool>"
    submit = "<reasoning>done</reasoning><tool>{}</tool>"
    return {
        "task_id": "spider_train_00001", "split": "train", "db_id": "db",
        "question": "q",
        "prompt_version": "reasoning_tool_observation_v4",
        "status": "submitted_sql", "correct": False, "final_sql": "SELECT wrong",
        "verification": {"correct": False, "agent_sql_executable": True},
        "turns": [
            {"tool": "list_tables", "arguments": {}, "observation": {"status": "success"}},
            {"tool": "execute_sql", "arguments": {"sql": "SELECT wrong"},
             "observation": {"status": "success", "rows": [[1]]}},
            {"tool": "submit_sql", "arguments": {"sql": "SELECT wrong"},
             "observation": {"status": "success"}},
        ],
        "messages": prompt + [
            {"role": "assistant", "content": "list"},
            {"role": "user", "content": "<observation>tables</observation>"},
            {"role": "assistant", "content": wrong},
            {"role": "user", "content": "<observation>ok</observation>"},
            {"role": "assistant", "content": submit},
        ],
    }


class DpoPairTests(unittest.TestCase):
    def test_fork_after_verified_wrong_success(self):
        source = _source()
        judge = lambda sql: sql == "SELECT gold"
        self.assertEqual(fork_info(source, judge)["fork_turn"], 3)
        self.assertEqual(fork_info(source, judge)["fork_reason"], "after_wrong_sql_success")
        unrelated = copy.deepcopy(source)
        unrelated["turns"][1]["arguments"]["sql"] = "SELECT unrelated"
        self.assertEqual(fork_info(unrelated, judge)["fork_turn"], 3)
        failed = copy.deepcopy(source)
        failed["turns"][1]["observation"]["status"] = "error"
        self.assertEqual(fork_info(failed, judge)["fork_reason"], "after_sql_error")
        correct_execution = copy.deepcopy(source)
        correct_execution["turns"][1]["arguments"]["sql"] = "SELECT gold"
        self.assertEqual(fork_info(correct_execution, judge)["fork_reason"],
                         "after_other_sql_success")
        no_execution = copy.deepcopy(source)
        no_execution["turns"].pop(1)
        self.assertEqual(fork_info(no_execution, judge)["fork_reason"],
                         "before_untested_submission")
        invalid = copy.deepcopy(no_execution)
        invalid["status"] = "invalid_format"
        invalid["turns"].pop()
        self.assertEqual(fork_info(invalid, judge)["fork_reason"],
                         "before_invalid_response")

    def test_pair_prefix_and_fork(self):
        source = _source()
        chosen = source["messages"][:6] + [
            {"role": "assistant", "content": "execute gold"},
            {"role": "user", "content": "<observation>gold result</observation>"},
            {"role": "assistant", "content": "submit gold"},
        ]
        repair = {"status": "repaired", "cutoff_turn": 3,
                  "student_messages": chosen, "turns": source["turns"],
                  "final_sql": "SELECT gold", "verification": {"correct": True},
                  "gold_hint_sha256": "abc", "teacher_model": "teacher"}
        pair = make_pair(source, repair, 3, "after_wrong_sql_success")
        self.assertEqual(pair["prompt_messages"], source["messages"][:6])
        self.assertEqual(pair["prompt_messages"][-1]["content"], "<observation>ok</observation>")
        self.assertEqual(pair["loss_from_turn"], 3)
        changed = copy.deepcopy(repair)
        changed["student_messages"][5]["content"] = "different observation"
        with self.assertRaisesRegex(ValueError, "different prefixes"):
            make_pair(source, changed, 3, "after_wrong_sql_success")

    def test_selection_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "source" / "trajectories"
            directory.mkdir(parents=True)
            path = directory / "one.json"
            original = json.dumps(_source())
            path.write_text(original, encoding="utf-8")
            result = select(directory.parent, judge_factory=lambda _: (
                True, lambda sql: sql == "SELECT gold"))
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["eligible_count"], 1)
            self.assertEqual(result["entries"][0]["fork_turn"], 3)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
