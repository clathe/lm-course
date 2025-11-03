import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from ttlm.tokenizer.base import Tokenizer

@torch.inference_mode()
def generate(
        model,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Naive autoregressive generation."""
        model.eval()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
