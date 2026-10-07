from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from sql_agent.deepseek import DeepSeekClient


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DeepSeekTextTests(unittest.TestCase):
    def test_text_completion_uses_no_native_tools(self):
        body = {"choices": [{"message": {"content": "<reasoning>x</reasoning>"},
                             "finish_reason": "stop"}], "id": "request-1", "usage": {}}
        with patch("urllib.request.urlopen", return_value=_Response(json.dumps(body).encode())) as open_url:
            completion = DeepSeekClient("fake-key").complete_text(
                [{"role": "user", "content": "question"}], temperature=0.3, max_tokens=32)
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(completion.message["content"], "<reasoning>x</reasoning>")


if __name__ == "__main__":
    unittest.main()
