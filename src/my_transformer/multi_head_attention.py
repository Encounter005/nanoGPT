"""多头注意力分支：投影 → 分头 → 注意力 → 拼接 → 输出投影。"""

import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """多头注意力：分别投影 Q/K/V，分头计算注意力，再拼接并融合各头结果。"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        n_head：注意力头数，下面用 H 表示。
        d_model：输入和最终输出的特征维度，下面用 D 表示。
        d_k：每个头的 Query、Key 维度，两者相同才能计算点积。
        d_v：每个头的 Value 维度，也是每个头的注意力输出维度。
        dropout：注意力权重和输出特征的随机丢弃概率，作用位置不同。
        常见配置：D=512，H=8，d_k=d_v=64。
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 先做可学习的投影，再分头；不是直接切分原始词向量。
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # D → H*d_k
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # D → H*d_k
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # D → H*d_v，可与 d_k 不同
        self.func = nn.Linear(n_head * d_v, d_model, bias=False)  # 融合各头：H*d_v → D

        # 各头并行计算 softmax(QKᵀ / sqrt(d_k))V。
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)  

        # Pre-LN 和残差连接由 EncoderLayer/DecoderLayer 负责，本模块只计算注意力分支。

    def forward(self, q, k, v, mask=None):
        """
        B：batch 大小；Lq：查询长度；Lk：键长度；Lv：值长度（必须等于 Lk）。
        输入 q：[B, Lq, D]，k：[B, Lk, D]，v：[B, Lv, D]。
        自注意力：q/k/v 来自同一序列；交叉注意力：q 来自 Decoder，k/v 来自 Encoder。
        mask：可广播到 [B, H, Lq, Lk] 的允许关注标记，0/False 表示屏蔽。
        此处 unsqueeze 前常用形状为 [B, 1, Lk]（padding）或 [B, Lq, Lk]。
        返回：输出 [B, Lq, D]，注意力权重 [B, H, Lq, Lk]。
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        batch_size, query_length, key_length ,value_length = q.size(0), q.size(1), k.size(1), v.size(1)

        # 1. 线性投影 + view 分头；view 只改变形状，不学习新特征。
        q = self.w_qs(q).view(batch_size, query_length, n_head, d_k)  # [B,Lq,D] → [B,Lq,H*d_k] → [B,Lq,H,d_k]
        k = self.w_ks(k).view(batch_size, key_length, n_head, d_k)  # [B,Lk,D] → [B,Lk,H*d_k] → [B,Lk,H,d_k]
        v = self.w_vs(v).view(batch_size, value_length, n_head, d_v)  # [B,Lv,D] → [B,Lv,H*d_v] → [B,Lv,H,d_v]

        # 2. 交换序列维和头维，使 matmul 对每个 batch、每个头独立计算。
        # q：[B,H,Lq,d_k]；k：[B,H,Lk,d_k]；v：[B,H,Lv,d_v]。
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 3. 给 mask 增加头维，使所有头共享同一屏蔽规则。
        if mask is not None:
            # 例如 [B,Lq,Lk] → [B,1,Lq,Lk]，新增的头维广播到全部 H 个头。
            mask = mask.unsqueeze(1)

        # 4. QKᵀ 得到 [B,H,Lq,Lk] 的匹配分数，mask + softmax + dropout 得到 attn。
        # attn @ V 得到 [B,H,Lq,d_v]；从这里开始 q 表示注意力输出，不再是 Query。
        q, attn = self.attention(q, k, v, mask=mask)

        # 5. [B,H,Lq,d_v] → [B,Lq,H,d_v] → [B,Lq,H*d_v]，拼接各头而非求和。
        # transpose 后内存通常不连续，contiguous 使其可用 view 合并维度。
        q = q.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        q = self.dropout(self.func(q))  # 输出投影融合各头：[B,Lq,H*d_v] → [B,Lq,D]
        # 返回计算分支与权重；归一化和残差相加由上层负责。
        # 训练时权重已经过 dropout，每行之和不保证为 1。
        return q, attn
