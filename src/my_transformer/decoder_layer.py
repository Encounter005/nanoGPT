"""解码器的单个 Pre-LN Block：目标自注意力、交叉注意力和 FFN。"""

import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """单个解码器层：目标自注意力 → 源/目标交叉注意力 → 逐位置前馈网络。"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        d_model：输入/输出特征维度 D；d_inner：FFN 中间层维度。
        n_head：注意力头数 H；d_k/d_v：每个头的 Q、K / V 维度。
        dropout：传给注意力输出和 FFN 的随机丢弃概率。
        """
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 目标序列内部交流
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 根据目标表示检索源句信息
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)  # 加工融合后的特征

        self.norm_ffn = nn.LayerNorm(d_model, eps=1e-6)  # FFN 前归一化
        self.norm_cross = nn.LayerNorm(d_model, eps=1e-6)  # 交叉注意力前只归一化目标查询
        self.norm_self = nn.LayerNorm(d_model, eps=1e-6)  # 目标自注意力前归一化

    def forward(
        self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None
    ):
        """
        dec_input：[B, Ltgt, D]，当前层的目标序列表示。
        enc_output：[B, Lsrc, D]，完整 Encoder 的输出；Lsrc 可以不同于 Ltgt。
        slf_attn_mask：通常为 [B, Ltgt, Ltgt]，合并目标 padding 与因果屏蔽。
        dec_enc_attn_mask：通常为 [B, 1, Lsrc]，屏蔽源序列 padding。
        mask 中 0/False 表示禁止关注；本层只使用 mask，不负责生成。
        返回 x：[B,Ltgt,D]；self_attn：[B,H,Ltgt,Ltgt]；cross_attn：[B,H,Ltgt,Lsrc]。
        """
        # 1. 目标自注意力：Q/K/V 均来自目标表示。
        # 必须由上层传入因果 mask，避免训练时偷看未来答案。
        # 输出：[B,Ltgt,D]；权重：[B,H,Ltgt,Ltgt]。

        z = self.norm_self(dec_input)
        a, self_attn = self.slf_attn(z, z, z, mask=slf_attn_mask)
        x = dec_input + a  # 残差使用原始 dec_input，不是归一化后的 z

        # 2. 交叉注意力：Q 来自上一步目标表示，K/V 来自 Encoder 输出。
        # 每个目标位置可关注完整源句的有效位置，不对源句做因果屏蔽。
        # 权重：[B,H,Ltgt,Lsrc]；输出长度跟随 Q，仍为 [B,Ltgt,D]。
        z = self.norm_cross(x)
        a, cross_attn = self.enc_attn(z, enc_output, enc_output, mask=dec_enc_attn_mask)
        x = x + a  # 交叉注意力调整目标表示，源 memory 不在此处做残差相加
        
        # 3. FFN 逐位置加工，形状保持 [B,Ltgt,D]。
        # 本层负责 Pre-LN 与残差连接；底层模块只计算 Attention/FFN 分支。

        x = x + self.pos_ffn(self.norm_ffn(x))
        
        # 返回特征表示、目标自注意力权重、目标对源句的交叉注意力权重。
        # 特征还需经上层词表投影才能得到 logits；本层不直接输出词编号。
        return x, self_attn, cross_attn
