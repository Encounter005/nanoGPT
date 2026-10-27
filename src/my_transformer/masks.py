"""Transformer 的 padding 和因果 mask；True 表示允许关注。"""

import torch

def get_pad_mask(seq, pad_idx):
    """
    seq：[B,L] 的 token 编号；pad_idx：当前词表的 PAD 编号。
    返回：[B,1,L] 的布尔 mask，非 PAD 为 True，PAD 为 False。
    中间的 1 用于广播到所有查询位置；上层注意力还会增加 head 维。
    只屏蔽作为 Key 的 PAD 位置，不会自动把 PAD 查询的输出清零。
    """
    # [B,L] → [B,1,L]，避免每个查询重复存储相同的 padding 规则。
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    seq：[B,L]，这里只使用形状和设备，不读取 token 内容。
    返回：[1,L,L] 的下三角布尔 mask，允许当前及之前位置，屏蔽未来。
    第一维为 1，广播给所有 batch 样本；本函数不单独处理 PAD。
    与 get_pad_mask(seq, pad_idx) 按位与后得到 [B,L,L] 的目标 mask。
    """
    batch_size, sentence_length = seq.size()
    # 1. triu(diagonal=1) 在主对角线上方填 1，代表未来位置。
    # 2. 用 1 减去它得到含主对角线的下三角矩阵，再转成布尔值。
    subsequent_mask = (
        1
        - torch.triu(
            torch.ones((1, sentence_length, sentence_length), device=seq.device),
            diagonal=1,
        )
    ).bool()
    # 保持 seq 的设备；与当前注意力实现一致，True 表示允许而非屏蔽。
    return subsequent_mask
