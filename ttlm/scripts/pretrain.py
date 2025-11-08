"""Main pre-training script."""

import argparse
import logging
import token

logging.basicConfig(level=logging.INFO)
import os

import torch
from experiments.loader import load as load_experiment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ttlm.config import PreTrainingConfig
from ttlm.dataset.tinystories import TinyStories
from ttlm.dist import World
from ttlm.scheduler import get_cos_with_warmup


def pretrain(config: PreTrainingConfig) -> None:
    """Main pre-training loop."""
    with World(device=config.device) as world:
        tokenizer = config.tokenizer.module()
        dataset = TinyStories()
        tokenizer.train(dataset.data)
        sampler = (
            DistributedSampler(dataset, drop_last=True) if world.distributed else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size // world.world_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            shuffle=False if sampler else config.data.shuffle,
            sampler=sampler,
        )
        model = config.model.module(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            ff_dim=config.model.ff_dim,
            dropout=config.model.dropout,
        )
        model.to(world.device, dtype=config.dtype)
        if world.distributed:
            model = DDP(model, device_ids=[world.local_rank])
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        lr_scheduler = get_cos_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(config.scheduler.warmup_steps_ratio * len(dataloader)),
            num_training_steps=config.epochs * len(dataloader),
            min_lr_ratio=config.scheduler.min_lr_ratio,
            num_cycles=config.scheduler.num_cycles,
        )
        for epoch in range(config.epochs):
            if world.distributed:
                sampler.set_epoch(epoch)
            for i, batch in enumerate(dataloader):
                model.train()
                input_ids = tokenizer.encode(batch)
                tensor_ids = pad_sequence(
                    input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(world.device)
                base_model = model.module if world.distributed else model
                with torch.autocast(device_type=world.device.type, dtype=config.dtype):
                    logits = base_model(input_ids=tensor_ids)
                pred_logits = logits[..., :-1, :].reshape(-1, tokenizer.vocab_size)
                labels = tensor_ids[..., 1:].reshape(-1).to(world.device)
                loss = torch.nn.functional.cross_entropy(
                    pred_logits, labels, ignore_index=tokenizer.pad_token_id
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if world.is_main_process:
                    logging.info(f"Epoch {epoch + 1}, last step loss: {loss.item()}")
        if world.is_main_process:
            logging.info("Pre-training completed successfully, saving model...")
            os.makedirs(f"logs/{args.experiment}", exist_ok=True)
            model.to_ckpt(f"logs/{args.experiment}.ckpt", tokenizer=tokenizer)
        world.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--experiment_id", type=int, default=0)
    args = parser.parse_args()
    config = load_experiment(args.experiment, args.experiment_id)
    pretrain(config)
