import argparse

import torch
from ttlm.engine import generate
from ttlm.model import Model


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a checkpoint")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint file", default="logs/default.ckpt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}")
    model, tokenizer = Model.from_ckpt(args.ckpt)
    model = model.to(args.device)
    model.eval()

    print(f"Model parameters: {model.num_parameters:,}")
    print(f"Generating {args.num_samples} samples...")

    for i in range(args.num_samples):
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=args.device)

        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        generated_tokens = output_ids[0].tolist()
        print(f"\nSample {i + 1}:")
        print(f"Tokens: {generated_tokens}")
        print(f"Text: {tokenizer.decode(generated_tokens)}")


if __name__ == "__main__":
    main()
