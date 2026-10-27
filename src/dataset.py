#!/usr/bin/env python3
from pathlib import Path
from config import GPTConfig
import torch


class CharDataset:
    def __init__(self, config: GPTConfig):
        data = Path(config.dataset_path)
        text = data.read_text(encoding="utf-8")
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char : index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # index : char

        config.vocab_size = len(chars)
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        # split data
        data = torch.tensor(self.encode(text), dtype=torch.long)
        ratio = int(0.9 * len(data))
        self.train_data = data[:ratio]
        self.val_data = data[ratio:]

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]

    def decode(self, indicator: list) -> str:
        return "".join([self.itos[i] for i in indicator])

    def get_batch(self, split: str):
        block_size = self.block_size
        batch_size = self.batch_size
        if split not in ("train", "val"):
            raise ValueError(f"split 必须是 'train' 或 'val'，收到 {split!r}")

        tokens = self.train_data if split == "train" else self.val_data

        if len(tokens) <= self.block_size:  # ← 3
            raise ValueError(
                f"{split} 只有 {len(tokens)} token，不足 block_size={self.block_size}"
            )
        ix = torch.randint(len(tokens) - block_size, (batch_size,))
        x = torch.stack([tokens[i : i + block_size] for i in ix])
        y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
        return x, y


class BatchSequence:
    """把 CharDataset.get_batch 包装成可重复迭代的 batch 序列，供 RunnerV3.fit 使用。

    语料是连续流，没有"一轮"的概念：每次迭代随机采 steps 个 batch，
    所以一个 epoch 就是 steps 步。
    """

    def __init__(self, dataset: CharDataset, split: str, steps: int):
        if steps < 1:
            raise ValueError(f"steps 必须为正整数，收到 {steps}")
        self.dataset = dataset
        self.split = split
        self.steps = steps

    def __iter__(self):
        for _ in range(self.steps):
            yield self.dataset.get_batch(self.split)
