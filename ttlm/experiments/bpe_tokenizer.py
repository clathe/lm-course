"""Default pretraining experiment configuration."""

import ttlm.tokenizer.bpe
from ttlm.config import PreTrainingConfig, TokenizerConfig

CFG = PreTrainingConfig(
    tokenizer=TokenizerConfig(
        module=ttlm.tokenizer.bpe.BPETokenizer,
    ),
)