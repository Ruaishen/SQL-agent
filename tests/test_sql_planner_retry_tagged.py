from __future__ import annotations

import json
from collections import deque

from sql_planner.collect import _record_path
from sql_planner.deepseek import Completion
from sql_planner.retry_tagged import retry_task


class FakeClient:
    model = "fake-deepseek"

    def __init__(self, *responses):
        self.responses = deque(responses)

    def complete_text(self, messages, *, temperature, max_tokens):
        return Completion(
            {"role": "assistant", "content": self.responses.popleft()},
            "stop", "fake-request", self.model, {},
        )


def tagged(reasoning, name, arguments):
    return f'<reasoning>{reasoning}</reasoning><tool>{json.dumps({"name": name, "arguments": arguments})}</tool>'


def test_retry_stops_on_first_correct_and_resumes_existing_files(sample_db, config, task, tmp_path):
    del sample_db
    client = FakeClient(
        tagged("List tables.", "list_tables", {}),
        tagged("Try a query.", "submit_sql", {"sql": "SELECT 1"}),
        tagged("List tables.", "list_tables", {}),
        tagged("Count employees.", "submit_sql", {"sql": "SELECT count(*) FROM employees"}),
    )
    result = retry_task(task, config, client, tmp_path, max_attempts=2, temperature=0.7, max_tokens=256)
    assert result == {"task_id": task.task_id, "attempted": 2, "created": 2, "rescued": True}
    assert json.loads(_record_path(tmp_path, task, 1).read_text())["correct"] is False
    assert json.loads(_record_path(tmp_path, task, 2).read_text())["correct"] is True
    resumed = retry_task(task, config, FakeClient(), tmp_path, max_attempts=2, temperature=0.7, max_tokens=256)
    assert resumed["created"] == 0 and resumed["rescued"]
