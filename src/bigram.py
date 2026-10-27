#!/usr/bin/env python3
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from lm import LM
from config import GPTConfig
from dataset import CharDataset


def train(model: nn.Module, data: CharDataset, config: GPTConfig):
    device = next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    for iter in range(config.epoch):
        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, data, config)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        src, label = data.get_batch("train")
        src = src.to(device)
        label = label.to(device)

        _, loss = model(src, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())


class BigramLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super(BigramLanguageModel, self).__init__()
        self.num_embed = config.num_embed
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.num_embed)
        self.lm_head = nn.Linear(config.num_embed, config.vocab_size)
        self.device = config.device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd: Tensor = self.token_embedding_table(idx)  # [B, T, C]

        logits = self.lm_head(token_embd)  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # 需要预测数量一致，位置一一对应
            # targets = targets.view(-1)  # [B * T]
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int):
        # idx: [B, T]
        idx = idx.to(next(self.parameters()).device)
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus on last time step
            logits = logits[:, -1, :]  # becomes[B, C]
            # get proabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T + 1]
        return idx


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


def main():
    config = GPTConfig()
    data = CharDataset(config)
    device = config.device
    model = BigramLanguageModel(config).to(device)
    train(model, data, config)
    model.eval()
    device = config.device

    # idx = torch.zeros((1, 1), dtype=torch.long)
    idx = torch.tensor(
        data.encode("hello there"), dtype=torch.long, device=device
    ).unsqueeze(0)
    res = model.generate(idx, max_new_tokens=500)
    # decode() 需要一维编号列表，所以先用 [0] 去掉外层的批次结构。
    # 如果不取 [0]，.tolist() 得到的是嵌套列表 [[0, 23, 8, ...]]，而不是 decode() 所需的 [0, 23, 8, ...]。
    res = data.decode(res[0].tolist())
    print(res)


if __name__ == "__main__":
    main()
