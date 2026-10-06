from __future__ import annotations

import unittest
from types import SimpleNamespace

from dpo.prepare import prepare, tokenize_branch


class _CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        result = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return result + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, value, *, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(char) for char in value])


class DpoMaskTests(unittest.TestCase):
    def test_partial_generation_is_rejected_before_tokenization(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            pairs = directory / "pairs"
            pairs.mkdir()
            (directory / "selection.json").write_text(
                json.dumps({"eligible_count": 1, "entries": [{"file": "one.json", "eligible": True}]}),
                encoding="utf-8")
            (directory / "generation_report.json").write_text(
                json.dumps({"status": "partial", "eligible_pairs": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prepare(pairs, directory / "missing-model", directory / "tokens", 1000)

    def test_only_assistant_turns_from_fork_are_scored(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "discover"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "wrong sql"},
            {"role": "user", "content": "success result"},
            {"role": "assistant", "content": "submit"},
        ]
        branch = tokenize_branch(_CharacterTokenizer(), messages, 2, 1000)
        rendered = _CharacterTokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        scored = "".join(rendered[index + 1] for index, flag in enumerate(branch["loss_mask"])
                         if flag)
        self.assertEqual(scored, "wrong sql</assistant>submit</assistant>")
        self.assertEqual(branch["assistant_turns"], 3)


if __name__ == "__main__":
    unittest.main()
