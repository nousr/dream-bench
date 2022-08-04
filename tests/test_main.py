import pytest
from dream_bench import tokenizer

@pytest.mark.parametrize("message", ["hello", "world"])
def test_tokenizer(message):
    tokenizer.tokenize(texts=message, context_length=77, truncate_text=True)
