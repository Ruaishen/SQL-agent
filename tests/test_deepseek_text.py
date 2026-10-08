from __future__ import annotations

import io
import http.client
import json
import unittest
from unittest.mock import patch

from sql_agent.deepseek import DeepSeekAPIError, DeepSeekClient


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DeepSeekTextTests(unittest.TestCase):
    def test_incomplete_response_body_retries(self):
        class BrokenResponse(_Response):
            def read(self, *args, **kwargs):
                raise http.client.IncompleteRead(b'partial')

        body = {'choices': [{'message': {'content': '{"experience":"Check grouping."}'},
                             'finish_reason': 'stop'}]}
        responses = [BrokenResponse(), _Response(json.dumps(body).encode())]
        with patch('urllib.request.urlopen', side_effect=responses) as open_url, patch('time.sleep'):
            completion = DeepSeekClient('fake-key', max_retries=1).complete_reflection(
                [{'role': 'user', 'content': 'question'}])
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(json.loads(completion.message['content'])['experience'], 'Check grouping.')

    def test_repeated_incomplete_response_becomes_api_error(self):
        with patch('urllib.request.urlopen', side_effect=http.client.IncompleteRead(b'partial')) as open_url, patch('time.sleep'):
            with self.assertRaisesRegex(DeepSeekAPIError, 'request failed: IncompleteRead'):
                DeepSeekClient('fake-key', max_retries=2).complete_reflection(
                    [{'role': 'user', 'content': 'question'}])
        self.assertEqual(open_url.call_count, 3)

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
