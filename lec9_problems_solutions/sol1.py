class BasicTokenizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def get_stats(self, tokens):
        pairs = {}
        for word in tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
              
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def merge_vocab(self, tokens, pair_to_merge):
        new_tokens = []
        bigram = ''.join(pair_to_merge)
        for word in tokens:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair_to_merge:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_tokens.append(new_word)
        return new_tokens

    def train(self, text, vocab_size, verbose=False):
        self.vocab_size = vocab_size
        tokens = [list(word) for word in text.split()]
        while len(self.vocab) < vocab_size:
            pairs = self.get_stats(tokens)
            if not pairs: break
            most_common = max(pairs, key=pairs.get)
            tokens = self.merge_vocab(tokens, most_common)
            self.vocab[''.join(most_common)] = len(self.vocab)
            if verbose:
                print(f"Merged: {most_common}")
        # Add all remaining characters
        for word in tokens:
            for token in word:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        tokens = list(text)
        i = 0
        while i < len(tokens) - 1:
            pair = tokens[i] + tokens[i+1]
            if pair in self.vocab:
                tokens[i:i+2] = [pair]
                i = max(i-1, 0)
            else:
                i += 1
        return [self.vocab[tok] for tok in tokens]

    def decode(self, ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([reverse_vocab[i] for i in ids])
