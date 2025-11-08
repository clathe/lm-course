import torch

from ttlm.tokenizer.base import Tokenizer


class CharacterPairEncoding(Tokenizer):
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self._merges = {}
        self._vocabulary = {}

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
        return self._vocab_size

    def _get_token_pair_frequencies(self, tokens: list[str]) -> dict[tuple[str, str], int]:
        frequencies = {}
        for i in range(len(tokens) - 1):
            if tokens[i] == "<EOD>" or tokens[i + 1] == "<EOD>":
                continue

            pair = (tokens[i], tokens[i + 1])
            frequencies[pair] = frequencies.get(pair, 0) + 1
        return frequencies

    def train(self, texts: list[str]) -> None:
        # texts is list[str]
        tokenised_texts = []
        for text in texts:
            tokenised_texts += list(text) + ["<EOD>"]
        # vocab int -> char/str
        vocab = {
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token,
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
            **{i + 4: token for i, token in enumerate(set(tokenised_texts)) if token != "<EOD>"}
        }

        # char/str -> char/str
        merges = {}

        for i in range(len(vocab), self.vocab_size):
            frequencies = self._get_token_pair_frequencies(tokenised_texts)
            if not frequencies:
                break

            most_frequent_pair = max(frequencies, key=frequencies.get)

            new_token = "".join(most_frequent_pair)
            tokenised_texts = self._merge_most_frequent_token_pair(tokenised_texts, most_frequent_pair)
            merges[most_frequent_pair] = i
            vocab[i] = new_token
        self._merges = merges
        self._vocabulary = vocab

    def _merge_most_frequent_token_pair(self, tokenised_texts: list[str], most_frequent_pair: tuple[str, str]) -> list[str]:
        new_tokenised_texts = []
        i = 0
        while i < len(tokenised_texts):
            if (
                i < len(tokenised_texts) - 1
                and (tokenised_texts[i], tokenised_texts[i + 1]) == most_frequent_pair
            ):
                new_tokenised_texts.append("".join(most_frequent_pair))
                i += 2
            else:
                new_tokenised_texts.append(tokenised_texts[i])
                i += 1
        return new_tokenised_texts

    def encode(
        self, strings: list[str], bos: bool = True, eos: bool = True
    ) -> list[torch.LongTensor]:
        encoded = []
        reversed_vocab = {v: k for k, v in self._vocabulary.items()}
        for s in strings:
            tokenised_str = list(s)
            while len(tokenised_str) > 1:
                merge_candidates = self._get_token_pair_frequencies(tokenised_str).keys()
                # get lowest rank merge candidate
                pair_to_merge = min(merge_candidates, key=lambda pair: self._merges.get(pair, float('inf')))
                # if no more merge candidate is found, all values are "inf" and we can stop
                if pair_to_merge not in self._merges:
                    break
                tokenised_str = self._merge_most_frequent_token_pair(tokenised_str, pair_to_merge)
            if bos:
                tokenised_str = [self.bos_token] + tokenised_str
            if eos:
                tokenised_str.append(self.eos_token)
            encoded.append(torch.LongTensor([
                reversed_vocab.get(token, self.unk_token_id) for token in tokenised_str
            ]))
        return encoded

    def decode(
        self, tokens: list[list[int]], special_tokens: bool = False
    ) -> list[str]:
        decoded = []
        special_tokens_to_remove = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
        }
        for token_list in tokens:
            if not special_tokens:
                token_list = [t for t in token_list if t not in special_tokens_to_remove]
            decoded_tokens = []
            for token in token_list:
                decoded_tokens.append(self._vocabulary.get(token, self.unk_token))
            decoded.append("".join(decoded_tokens))
        return decoded
