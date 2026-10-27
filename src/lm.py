import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig

from my_transformer import Encoder, get_subsequent_mask


class LM(nn.Module):
    """用已有的自注意力堆叠预测下一个 token，不使用交叉注意力。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        block_size = cfg.block_size
        vocab_size = cfg.vocab_size

        if block_size < 1:
            raise ValueError('block_size 必须为正整数')
        self.block_size = block_size
        # 类名虽为 Encoder，传入因果 mask 后只能读取当前位置及之前。
        self.backbone = Encoder(
            n_src_vocab=vocab_size, d_word_vec=cfg.num_embed, d_model=cfg.num_embed,
            d_inner=256, n_layers=cfg.num_layers, n_head=cfg.num_heads, d_k=cfg.head_size, d_v=cfg.head_size,
            pad_idx=None, dropout=cfg.dropout, n_position=block_size,
        )
        self.lm_head = nn.Linear(cfg.num_embed, vocab_size)

    def forward(self, idx, targets=None):
        """idx/targets 为 [B,T]；始终返回 [B,T,V] logits 和可选 loss。"""
        if idx.ndim != 2 or idx.size(0) == 0 or not 1 <= idx.size(1) <= self.block_size:
            raise ValueError('idx 必须为非空 [B,T]，且 T 不超过 block_size')
        if targets is not None and targets.shape != idx.shape:
            raise ValueError('targets 必须与 idx 的形状一致')
        hidden, = self.backbone(idx, get_subsequent_mask(idx))
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """保留完整输出，每次预测只使用最近 block_size 个 token。"""
        if idx.ndim != 2 or idx.size(0) == 0 or idx.size(1) == 0:
            raise ValueError('生成提示必须为非空 [B,T]')
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens 不能为负数')
        idx = idx.to(next(self.parameters()).device)
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
