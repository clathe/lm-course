"""Default pretraining experiment configuration."""

import ttlm.tokenizer.ascii_pair_encoding
from ttlm.config import PreTrainingConfig, TokenizerConfig

CFG = PreTrainingConfig(
    tokenizer=TokenizerConfig(
        module=ttlm.tokenizer.ascii_pair_encoding.ASCIIPairEncoding,
    ),
    device="cpu",
)