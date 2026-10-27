"""默认数据源：nanoGPT 的 .bin + meta.pkl（np.memmap 切片取 batch）。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from ..configs import DataConfig
from ..registry import register_dataset
from ..utils import DataError


class MemmapDataset:
    def __init__(self, cfg: DataConfig):
        root = Path(cfg.root) / cfg.name
        train_path, val_path = root / "train.bin", root / "val.bin"
        if not (train_path.exists() and val_path.exists()):
            raise DataError(
                f"缺少 {train_path} 或 {val_path}：先准备数据，例如 python data/shakespeare_char/prepare.py"
            )
        self.train = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val = np.memmap(val_path, dtype=np.uint16, mode="r")
        meta_path = root / "meta.pkl"
        self.stoi: dict[str, int] | None = None
        self.itos: dict[int, str] | None = None
        if meta_path.exists():
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
            self.stoi, self.itos = meta.get("stoi"), meta.get("itos")
        if self.stoi is None:  # 没有 meta 就按 GPT-2 分词处理
            import tiktoken

            self._enc = tiktoken.get_encoding("gpt2")
            self.vocab_size = self._enc.n_vocab
        else:
            self._enc = None
            self.vocab_size = len(self.stoi)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = {"train": self.train, "val": self.val}.get(split)
        if data is None:
            raise DataError(f"未知的 split：{split!r}（只能是 train / val）")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def encode(self, text: str) -> list[int]:
        if self.stoi is None:
            return self._enc.encode_ordinary(text)
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        if self.itos is None:
            return self._enc.decode(ids)
        return "".join(self.itos[i] for i in ids)


@register_dataset("memmap")
def build_memmap(cfg: DataConfig) -> MemmapDataset:
    return MemmapDataset(cfg)
