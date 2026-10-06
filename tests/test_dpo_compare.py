from dpo.compare import metrics, sample_entries


def test_hint_metric_only_audits_saved_chosen_reasoning():
    record = {
        "api_calls": [{"request_id": "test", "api_model": "test", "finish_reason": "stop",
            "message": {"content": "<reasoning>The hint suggests a query.</reasoning>",
                        "reasoning_content": "Use the gold SQL."},
            "usage": {}, "elapsed_seconds": 1}],
        "pair": {"chosen_turns": [{"reasoning": "I will inspect the result."}]},
        "repair": {"status": "repaired", "verification": {"correct": True}},
        "exact_gold_submission": True, "elapsed_seconds": 2,
    }
    result = metrics(record)
    assert result["constructed_pair"]
    assert not result["visible_hint_leak"]
    assert result["raw_call_hint_leak"]
    assert result["private_thinking_leaking_calls"] == 1
    record["pair"]["chosen_turns"][0]["reasoning"] = "The target SQL selects names."
    assert metrics(record)["visible_hint_leak"]


def test_sample_is_fixed_and_has_unique_tasks():
    selection = {"entries": [
        {"file": f"{i:04}.json", "task_id": str(i), "eligible": True,
         "fork_reason": "large" if i < 180 else "small"}
        for i in range(200)
    ]}
    sample = sample_entries(selection, 100, 42)
    assert sample == sample_entries(selection, 100, 42)
    assert len({x["task_id"] for x in sample}) == 100
    assert sum(x["fork_reason"] == "small" for x in sample) == 10
