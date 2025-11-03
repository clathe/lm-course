"""Cowboy's transformer-based next-token prediction language model."""

from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ttlm.tokenizer.base import Tokenizer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMS normalization to the input tensor."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """Feedforward block with SwiGLU activation."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Applies SwiGLU transformation to the input tensor."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) layer."""

    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        """Initializes the RotaryEmbedding layer."""
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be an even number, but got {head_dim}")

        half_dim = head_dim // 2
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, half_dim, dtype=torch.float32) * 2 / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cos_cache: Optional[Tensor] = None
        self._sin_cache: Optional[Tensor] = None

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotates half the hidden dimensions of the input tensor."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _update_cache(self, seq_len: int) -> None:
        """Updates the cache if the sequence length has changed."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            positions = torch.arange(
                seq_len, device=self.inv_freq.device, dtype=torch.float32
            )
            freqs = torch.outer(positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cache = emb.cos()
            self._sin_cache = emb.sin()

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Applies RoPE to the query and key tensors."""
        _, _, seq_len, _ = q.shape
        self._update_cache(seq_len)
        cos = (
            self._cos_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        )  # Shape: [1, 1, seq_len, head_dim]
        sin = (
            self._sin_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        )  # Shape: [1, 1, seq_len, head_dim]
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


class Attention(nn.Module):
    """Multi-head self-attention with rotary embeddings for autoregressive models."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        attn_bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.rotary_emb = RotaryEmbedding(self.head_dim, rope_theta=rope_theta)
        self.dropout = nn.Dropout(dropout)
        self.scale_attention = RMSNorm(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Applies rotary self-attention with causal masking."""
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary_emb(q, k)
        q, k = self.scale_attention(q), self.scale_attention(k)  # QK norm
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, self.hidden_dim)
        return self.dropout(self.o_proj(attn_out))


class TransformerBlock(nn.Module):
    """Transformer block: RMSNorm, rotary attention, RMSNorm, SwiGLU."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attn_bias: bool = False,
        ff_bias: bool = False,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_dim)
        self.self_attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.post_norm = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, ff_dim, bias=ff_bias)

    def forward(self, x: Tensor) -> Tensor:
        """Applies pre-norm rotary attention and SwiGLU MLP."""
        residual = x
        x = self.pre_norm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        return residual + x


class Model(nn.Module):
    """Transformer-based autoregressive language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        softcap: float = 15.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.softcap = softcap
        self.flash_attention = False
        self.init_std = 0.02
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def to_ckpt(self, path: str, tokenizer: Tokenizer) -> None:
        """Saves model state and config to a checkpoint file."""
        checkpoint = {
            "state_dict": self.cpu().state_dict(),
            "tokenizer": tokenizer,
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "softcap": self.softcap,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_ckpt(cls, path: str) -> "Model":
        """Loads a model from a checkpoint file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
            softcap=config["softcap"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = checkpoint["tokenizer"]
        return model, tokenizer

    def forward(self, input_ids: Tensor) -> dict[str, Tensor]:
        """Forward pass returning logits."""
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits
