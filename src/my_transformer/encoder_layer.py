"""编码器的单个 Pre-LN Block：源自注意力和 FFN 各有独立归一化与残差。"""

import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .positionwise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """单个编码器层：源序列自注意力 → 逐位置前馈网络，可堆叠成完整 Encoder。"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        d_model：输入/输出特征维度 D；d_inner：FFN 中间层维度。
        n_head：注意力头数 H；d_k/d_v：每个头的 Q、K / V 维度。
        dropout：传给注意力输出和 FFN 的随机丢弃概率。
        """
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)  # 融合源句各位置的信息
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)  # D → d_inner → D，逐位置加工特征
        self.norm_attn = nn.LayerNorm(d_model, eps=1e-6)  # 自注意力前归一化 D 维特征
        self.norm_ffn = nn.LayerNorm(d_model, eps=1e-6)  # FFN 前的独立归一化

    def forward(self, enc_input, slf_attn_mask=None):
        """
        enc_input：[B, Lsrc, D]，当前层的源序列表示；B 为 batch 大小。
        slf_attn_mask：通常为源序列 padding mask [B, 1, Lsrc]，0/False 表示屏蔽。
        本层不生成 mask；编码时可看完整源句，通常不需要因果 mask。
        返回 x：[B,Lsrc,D]，attn：[B,H,Lsrc,Lsrc]。
        """
        # 1. 先归一化输入，Q/K/V 都取同一个 z，因此是源序列自注意力。
        # 输出：[B,Lsrc,D]；各头注意力权重：[B,H,Lsrc,Lsrc]。
        z = self.norm_attn(enc_input)
        attn_output, attn= self.slf_attn(
            z, z, z, mask=slf_attn_mask
        )
        # 2. 注意力分支加回原始未归一化输入，保留残差直通路径。
        x = enc_input + attn_output
        # 3. FFN 前归一化，再加回当前 x；各位置共享 FFN 参数，形状不变。
        x  = x + self.pos_ffn(self.norm_ffn(x))

        # 本层不再对相加结果做 Post-LN；最终归一化由完整 Encoder 负责。
        return x, attn
