"""逐位置前馈分支：每个位置共享同一套 Linear → ReLU → Linear。"""

import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """加工每个词的特征，不直接混合不同位置的信息；输入/输出维度相同。"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        d_in：输入/输出特征维度 D；d_hid：中间特征维度，通常大于 D。
        dropout：分支输出的丢弃概率；本类不包含 LayerNorm 或残差相加。
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # D → d_hid，学习隐藏特征组合
        self.w_2 = nn.Linear(d_hid, d_in)  # d_hid → D，恢复与残差分支一致的维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x：[B,L,D]，由上层 Pre-LN 处理后的表示；返回仍为 [B,L,D]。
        B：batch 大小；L：当前序列长度；Linear 只变换最后一个特征维度。
        """

        # 1. [B,L,D] → [B,L,d_hid] → ReLU → [B,L,D]。
        # ReLU 使两层线性变换不再等价于一层，增强非线性表达能力。
        x = self.w_2(F.gelu(self.w_1(x), approximate='tanh'))
        # 2. 输出 dropout；EncoderLayer/DecoderLayer 再将结果加到原始残差上。
        return self.dropout(x)
