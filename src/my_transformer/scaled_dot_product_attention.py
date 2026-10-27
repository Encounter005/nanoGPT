"""缩放点积注意力：softmax(QKᵀ / temperature)V，处理屏蔽与半精度边界。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """根据 Q/K 的匹配程度，对 V 加权求和；不是用 argmax 只选一个位置。"""

    def __init__(self, temperature, attn_dropout=0.1):
        """
        temperature：点积缩放因子，调用方通常设为 sqrt(d_k)，应为正数。
        attn_dropout：归一化后的注意力权重丢弃概率，不是输出特征 dropout。
        """
        super().__init__()
        self.temperature = temperature  # 减小大维度点积，使 softmax 不易过度尖锐
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q：[B,H,Lq,d_k]；k：[B,H,Lk,d_k]；v：[B,H,Lk,d_v]。
        B：batch 大小；H：头数；Lq/Lk：查询/键长度。
        mask：可广播到 [B,H,Lq,Lk]，0/False 表示禁止关注。
        返回 output：[B,H,Lq,d_v]，attn：[B,H,Lq,Lk]，dtype 与 v 一致。
        全屏蔽行返回零权重、零输出；训练时 dropout 后权重和不保证为 1。
        """
        # 1. q / temperature：[B,H,Lq,d_k]；k 转置：[B,H,d_k,Lk]。
        # 相乘得到每个查询对所有键的分数：[B,H,Lq,Lk]。
        # 半精度输入在 float32 中计算分数；局部关闭 autocast，避免点积再次降精度。
        with torch.autocast(device_type=q.device.type, enabled=False):
            if q.dtype in (torch.float16, torch.bfloat16):
                q, k = q.float(), k.float()
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 2. 将禁止关注的位置设为极小分数；mask 只屏蔽，不提供可训练参数。
        if mask is not None:
            # 使用当前 dtype 的最小有限值；全屏蔽行先得到有限的 softmax。
            blocked = mask == 0
            attn = attn.masked_fill(blocked, torch.finfo(attn.dtype).min)

        # 3. 沿键长度 Lk 归一化，每个查询独立得到关注各键的权重。
        attn = F.softmax(attn, dim=-1)
        if mask is not None:
            # 强制被屏蔽的权重为 0，包括全屏蔽行，避免均匀关注或 NaN。
            attn = attn.masked_fill(blocked, 0)
        # 4. 训练时丢弃部分注意力连接，再恢复 V 的 dtype 以便矩阵乘法。
        attn = self.dropout(attn).to(v.dtype)
        # 5. [B,H,Lq,Lk] @ [B,H,Lk,d_v] → [B,H,Lq,d_v]，汇总 V 的内容。
        output = torch.matmul(attn, v)

        return output, attn
