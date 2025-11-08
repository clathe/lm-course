import torch

from ttlm.tokenizer.base import Tokenizer


class ASCIIPairEncoding(Tokenizer):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, str] = {}

    @property
    def bos_token_id(self) -> int:
        return 129

    @property
    def bos_token(self) -> str:
        return "<BOS>"

    @property
    def eos_token_id(self) -> int:
        return 130

    @property
    def eos_token(self) -> str:
        return "<EOS>"

    @property
    def pad_token_id(self) -> int:
        return 131

    @property
    def pad_token(self) -> str:
        return "<PAD>"

    @property
    def unk_token_id(self) -> int:
        return 132

    @property
    def unk_token(self) -> str:
        return "<UNK>"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        return {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
        }

    @staticmethod
    def preprocess_text(text: str) -> list[int]:
        token_sequence = []
        for char in text:
            ascii_val = ord(char)
            if ascii_val < 128:
                token_sequence.append(ascii_val)
            else:
                raise ValueError(f"Character '{char}' is not in ASCII range.")
        return token_sequence

    def train(self, texts: list[str]) -> None:
        vocab = {i: chr(i) for i in range(128)}
        merges = {}

        tokenised_texts = [self.preprocess_text(text) for text in texts]

        num_merges = self.vocab_size - len(vocab) - len(self.special_tokens)
        for i in range(len(vocab) + len(self.special_tokens), len(vocab) + len(self.special_tokens) + num_merges):
            frequencies: dict[tuple[int, int], int] = {}
            for tokenised_text in tokenised_texts:
                frequencies = self._get_token_pair_frequencies(tokenised_text, frequencies)

            if not frequencies:
                break

            most_frequent_pair = max(frequencies, key=frequencies.get)
            merges[most_frequent_pair] = i
            new_token = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]
            vocab[i] = new_token
            tokenised_texts = [
                self._merge_token_pair(tokenised_text, most_frequent_pair, i)
                for tokenised_text in tokenised_texts
            ]

        self.merges = merges
        self.vocab = vocab

    @staticmethod
    def _get_token_pair_frequencies(tokenised_text: list[int], frequencies: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
        if frequencies is None:
            frequencies = {}
        for i in range(len(tokenised_text) - 1):
            pair = (tokenised_text[i], tokenised_text[i + 1])
            frequencies[pair] = frequencies.get(pair, 0) + 1
        return frequencies

    @staticmethod
    def _merge_token_pair(tokenised_text: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        merged_text = []
        i = 0
        while i < len(tokenised_text):
            if i < len(tokenised_text) - 1 and (tokenised_text[i], tokenised_text[i + 1]) == pair:
                merged_text.append(new_token)
                i += 2
            else:
                merged_text.append(tokenised_text[i])
                i += 1
        return merged_text

    def decode(self, tokens: list[list[int]], special_tokens: bool = False) -> list[str]:
        decoded = []
        special_token_ids = self.special_tokens.values()
        inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        for token_list in tokens:
            if not special_tokens:
                token_list = [t for t in token_list if t not in special_token_ids]
            chars = []
            for token in token_list:
                if token in self.vocab:
                    chars.append(self.vocab[token])
                elif token in special_token_ids:
                    chars.append(inverse_special_tokens[token])
                else:
                    chars.append(self.unk_token)
            decoded.append("".join(chars))
        return decoded

    def encode(
        self, strings: list[str], bos: bool = True, eos: bool = True
    ) -> list[torch.LongTensor]:
        encoded = []

        for s in strings:
            tokens = self.preprocess_text(s)
            while len(tokens) > 1:
                merge_canidates = self._get_token_pair_frequencies(tokens).keys()
                pair_to_merge = min(
                    merge_canidates,
                    key=lambda pair: self.merges.get(pair, float("inf")),
                )
                if pair_to_merge not in self.merges:
                    break

                tokens = self._merge_token_pair(
                    tokens, pair_to_merge, self.merges[pair_to_merge]
                )
            if bos:
                tokens = [self.bos_token_id] + tokens
            if eos:
                tokens.append(self.eos_token_id)
            encoded.append(torch.LongTensor(tokens))
        return encoded
