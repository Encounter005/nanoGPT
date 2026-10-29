#!/usr/bin/env python3
from my_trainer.configs import ModelConfig
from my_trainer.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Head(nn.Module):
    """
    one head of self-attention
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        num_embed = cfg.n_embd
        head_size = cfg.n_embd // cfg.n_head
        block_size = cfg.block_size
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)), persistent=False
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_size = head_size

    def forward(self, inputs: Tensor) -> Tensor:
        """
        input: [batch, time_step, channels]
        output: [batch, time_step, head_size]
        """
        B, T, C = inputs.shape
        k: Tensor = self.key(inputs) # [B, T, HS]
        q: Tensor = self.query(inputs) # [B, T, HS]
        v: Tensor = self.value(inputs) # [B, T, HS]
        wei = q @ k.transpose(-2, -1) # [B, T, HS] @ [B, HS, T] = [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei / (self.head_size**0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        outputs = wei @ v # [B, T, T] @ [B, T, HS] = [B, T, HS]
        return outputs

class MultiHeadAttention(nn.Module):
    '''multiple heads of self-attention in parallel'''
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        num_heads = cfg.n_head
        num_embed = cfg.n_embd
        head_size = cfg.n_embd // cfg.n_head
        dropout = cfg.dropout
        self.heads = nn.ModuleList([Head(cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        num_embed = cfg.n_embd
        dropout = cfg.dropout
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.GELU(),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer block: communication followed by computation'''

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        num_embed = cfg.n_embd

        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        num_embed = cfg.n_embd
        block_size = cfg.block_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.position_embedding_table = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(num_embed) # final layer norm
        self.lm_head = nn.Linear(num_embed, vocab_size)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embd: Tensor = self.token_embedding_table(idx)  # [B, T, C]
        pos_embd: Tensor = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # [T, C]

        x = token_embd + pos_embd #[B, T, C]
        x = self.blocks(x) #[B, T, C]
        x = self.ln_f(x) #[B, T, C]
        

        logits = self.lm_head(x)  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            # 保持 (B, T, V)：模板的 accuracy 和 generate 都按三维 logits 取最后一步
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        # idx: [B, T]
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -self.block_size :])
            # focus on last time step
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # [B, C]
            if top_k:
                top, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top[:, [-1]]] = -float("inf")
            # get proabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T + 1]
        return idx


@register_model("gpt_v0")
def build_gpt_v0(cfg: ModelConfig, vocab_size: int) -> GPTLanguageModel:
    return GPTLanguageModel(cfg, vocab_size)
