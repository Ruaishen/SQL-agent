from concurrent.futures import ThreadPoolExecutor

from tokenizers import Tokenizer

from sql_agent.truncation import TokenCounter


def test_shared_counter_matches_independent_encoding_and_refreshes_changed_file(config):
    reference = Tokenizer.from_file(str(config.tokenizer_path / "tokenizer.json"))
    counters = [TokenCounter(config.tokenizer_path) for _ in range(4)]
    assert all(c._tokenizer is counters[0]._tokenizer for c in counters)
    texts = ['<reasoning>Check schema.</reasoning>', 'SELECT count(*) FROM items', '中文证据'] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda item: counters[item[0] % 4].count_text(item[1]), enumerate(texts)))
    assert actual == [len(reference.encode(text, add_special_tokens=False).ids) for text in texts]
    path = config.tokenizer_path / "tokenizer.json"
    path.write_bytes(path.read_bytes() + b"\n")
    refreshed = TokenCounter(config.tokenizer_path)
    assert refreshed._tokenizer is not counters[0]._tokenizer
    assert refreshed.count_text(texts[0]) == actual[0]
