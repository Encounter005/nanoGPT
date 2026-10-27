#!/usr/bin/env python3

from lm import LM
import math

import torch
import torch.nn as nn
import torch.optim as optim

from config import GPTConfig
from dataset import BatchSequence, CharDataset
from gpt import GPTLanguageModel
from runner import RunnerV3
from utils import accuracy


@torch.no_grad()
def estimate_loss(model: nn.Module, data: CharDataset, config: GPTConfig):
    out = {}
    model.eval()
    eval_iters = config.eval_iters
    device = next(model.parameters()).device
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class LMAdapter(nn.Module):
    """RunnerV3 要 (N, C, ...) 的 logits，语言模型给 (B, T, V)。

    loss 由 Runner 自己算，所以丢掉模型返回的 loss；转置成 (B, V, T) 之后
    CrossEntropyLoss 和 accuracy（类别维都在 dim=1）可以直接用。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits.transpose(1, 2)


def make_lr_lambda(cfg: GPTConfig, total_steps: int):
    """线性 warmup + 余弦衰减到 min_lr，返回 LambdaLR 需要的比例函数。

    LambdaLR 会用 base_lr（optimizer 里的 learning_rate）乘以返回值，
    所以这里只返回相对比例，峰值 1.0、谷值 min_lr/learning_rate。
    """
    floor = cfg.min_lr / cfg.learning_rate

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_iters:
            return (step + 1) / cfg.warmup_iters
        if step >= total_steps:
            return floor
        ratio = (step - cfg.warmup_iters) / max(1, total_steps - cfg.warmup_iters)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * ratio))

    return lr_lambda


def main():
    cfg = GPTConfig()
    dataset = CharDataset(cfg)
    model = LM(cfg)
    # 每 eval_interval 步评估一次，所以一个 epoch = eval_interval 步，
    # 总步数仍是 cfg.epoch。LambdaLR 按 batch 步进，T_max 用这里的总步数。
    steps_per_epoch = cfg.eval_interval
    num_epochs = cfg.epoch // steps_per_epoch
    total_steps = steps_per_epoch * num_epochs
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    runner = RunnerV3(
        model=LMAdapter(model),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        metric_fn=accuracy,
        higher_is_better=True,
        device=cfg.device,  # 由 Runner 负责搬模型和每个 batch
    )

    runner.fit(
        train_loader=BatchSequence(dataset, "train", steps=steps_per_epoch),
        dev_loader=BatchSequence(dataset, "val", steps=cfg.eval_iters),
        num_epochs=num_epochs,
        log_every=1,
        seed=1337,
        best_path="gpt-1",
        lr_scheduler=optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(cfg, total_steps)
        ),
        lr_scheduler_step_per_batch=True,
    )


if __name__ == "__main__":
    main()
