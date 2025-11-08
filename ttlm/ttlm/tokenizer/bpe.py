import regex as re
import torch

from ttlm.tokenizer.base import Tokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPETokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 6000, pattern: str = None) -> None:
        super().__init__()
        self.vocab_size_limit = vocab_size
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.num_special_tokens = 4
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def bos_token(self) -> str:
        return "<BOS>"

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def eos_token(self) -> str:
        return "<EOS>"

    @property
    def pad_token_id(self) -> int:
        return 2

    @property
    def pad_token(self) -> str:
        return "<PAD>"

    @property
    def unk_token_id(self) -> int:
        return 3

    @property
    def unk_token(self) -> str:
        return "<UNK>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + self.num_special_tokens

    def _get_token_pair_frequencies(
        self, tokenised_text: list[bytes], frequencies: dict[tuple[int, int], int] | None = None
    ) -> dict[tuple[int, int], int]:
        frequencies = frequencies or {}
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

    def train(self, texts: list[str]) -> None:
        tokenised_texts: list[list[int]] = []
        for text in texts:
            chunks = self.compiled_pattern.findall(text)
            tokenised_texts.extend([list(chunk.encode("utf-8")) for chunk in chunks])

        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        num_merges = self.vocab_size_limit - len(vocab) - self.num_special_tokens
        for i in range(len(vocab) + self.num_special_tokens, len(vocab) + self.num_special_tokens + num_merges):
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

    def decode(self, tokens: list[list[int]], special_tokens: bool = False) -> list[str]:
        decoded = []
        special_tokens_to_remove = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        for token_list in tokens:
            if not special_tokens:
                token_list = [t for t in token_list if t not in special_tokens_to_remove]
            part_bytes = []
            for token in token_list:
                if token == self.unk_token_id:
                    part_bytes.append(self.unk_token.encode("utf-8"))
                elif token == self.bos_token_id:
                    part_bytes.append(self.bos_token.encode("utf-8"))
                elif token == self.eos_token_id:
                    part_bytes.append(self.eos_token.encode("utf-8"))
                elif token == self.pad_token_id:
                    part_bytes.append(self.pad_token.encode("utf-8"))
                elif token in self.vocab:
                    part_bytes.append(self.vocab[token])
                else:
                    part_bytes.append(self.unk_token.encode("utf-8"))
            decoded.append(b"".join(part_bytes).decode("utf-8", errors="replace"))
        return decoded

    def encode(self, strings: list[str], bos: bool = True, eos: bool = True) -> list[torch.LongTensor]:
        encoded = []

        for s in strings:
            chunks = self.compiled_pattern.findall(s)
            byte_chunks = [list(chunk.encode("utf-8")) for chunk in chunks]
            tokens: list[int] = []
            for byte_chunk in byte_chunks:
                while len(byte_chunk) > 1:
                    merge_candidates = self._get_token_pair_frequencies(byte_chunk).keys()
                    pair_to_merge = min(
                        merge_candidates,
                        key=lambda pair: self.merges.get(pair, float("inf")),
                    )
                    if pair_to_merge not in self.merges:
                        break

                    byte_chunk = self._merge_token_pair(
                        byte_chunk, pair_to_merge, self.merges[pair_to_merge]
                    )
                tokens.extend(byte_chunk)
            if bos:
                tokens = [self.bos_token_id] + tokens
            if eos:
                tokens.append(self.eos_token_id)
            encoded.append(torch.LongTensor(tokens))
        return encoded
