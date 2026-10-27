#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

torch.manual_seed(1337)


B, T, C = 4, 8, 32
# src
x = torch.randn(B, T, C)  # [ batch, time, channels ]
# single head self attention
head_size = 16
key = nn.Linear(C, head_size, bias=True)
query = nn.Linear(C, head_size, bias=True)
value = nn.Linear(C, head_size, bias=True)
k: Tensor = key(x)
q: Tensor = query(x)
v: Tensor = value(x)
scores = q @ k.transpose(-2, -1)  # [B, T, 16] @ [B, 16, T] = [B, T, T]
scores = scores / (head_size**0.5)
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = F.softmax(scores, dim=-1)
attn = attn @ v
print(attn.shape)
print(attn[0])
