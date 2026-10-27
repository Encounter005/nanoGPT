#!/usr/bin/env python3
from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    vocab_size: int | None = None
    dataset_path: str = './data/chinese/input.txt'
    block_size: int = 256
    num_embed: int = 512
    num_heads: int = 8
    num_layers: int = 6
    batch_size: int = 64
    eval_interval: int = 250
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters: int = 200
    epoch: int = 5000
    warmup_iters: int = 100
    min_lr: float = 1e-4
    dropout: float = 0.2

    def __post_init__(self):
        if self.num_embed % self.num_heads:
            raise ValueError(
                f"num_embed({self.num_embed}) 必须能被 num_heads({self.num_heads}) 整除"
            )

    @property
    def head_size(self) -> int:
        return self.num_embed // self.num_heads
