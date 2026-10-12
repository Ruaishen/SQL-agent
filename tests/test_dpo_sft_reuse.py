"""Real SQLite integration with scripted API replies and existing SFT code."""
import copy
import json
from collections import deque
from dataclasses import replace

from dpo.pairs import build
from sft.collect_reasoning import collect_trajectory
from sft.regenerate_reasoning import regenerate
from sql_agent.deepseek import Completion


def response(tool, args):
    return '<reasoning>Check the requested count using the database.</reasoning><tool>' + json.dumps({"name": tool, "arguments": args}) + '</tool>'


class ScriptedClient:
    model = "offline-scripted"

    def __init__(self, replies):
        self.replies = deque(replies)
        self.requests = []

    def complete_thinking_text(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return Completion({"content": self.replies.popleft(), "reasoning_content": "PRIVATE"}, "stop", None, None, {})

    complete_text = complete_thinking_text


def test_sft_regeneration_to_complete_dpo_pair(tmp_path, sample_db, config, task):
    task = replace(task, split="train")
    loser = collect_trajectory(task, config, ScriptedClient([
        response("list_tables", {}), response("execute_sql", {"sql": "SELECT 0"}),
        response("submit_sql", {"sql": "SELECT 0"})]))
    assert loser["correct"] is False
    teacher = ScriptedClient([response("list_tables", {}),
        response("execute_sql", {"sql": task.reference_sql}),
        response("submit_sql", {"sql": task.reference_sql}), "accept"])
    winner = regenerate(task, loser, config, teacher)
    for name, record in (("original", loser), ("regenerated", winner)):
        folder = tmp_path / name / "trajectories"
        folder.mkdir(parents=True)
        public = {k: v for k, v in record.items() if not k.startswith("private_")}
        (folder / "task.json").write_text(json.dumps(public), encoding="utf-8")
    output = tmp_path / "pairs.jsonl"
    report = build(tmp_path / "original", tmp_path / "regenerated", [task], config, output, workers=2)
    pair = json.loads(output.read_text(encoding="utf-8"))
    assert report["pairs"] == 1 and report["skipped"] == []
    assert pair["rejected_messages"] == loser["messages"]
    assert pair["chosen_messages"] == winner["student_messages"]
    assert pair["chosen_final_sql"] == task.reference_sql
    assert "PRIVATE" not in json.dumps(pair)
    assert "teacher_only_sql" not in json.dumps(pair)
    assert pair["prompt_messages"] == loser["messages"][:2]
    assert "teacher_only_sql" in teacher.requests[0][1]["content"]
