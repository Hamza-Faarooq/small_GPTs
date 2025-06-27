import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = re.compile(pattern)

    def split(self, text):
        return self.pattern.findall(text)

    def train(self, text, vocab_size, verbose=False):
        words = self.split(text)
        super().train(" ".join(words), vocab_size, verbose)

    def encode(self, text):
        parts = self.split(text)
        ids = []
        for part in parts:
            ids.extend(super().encode(part))
        return ids
