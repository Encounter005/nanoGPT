"""固定正弦/余弦位置编码：补充序列顺序，不引入可训练的位置参数。"""

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """固定的正弦/余弦位置编码：将位置信息加到词向量中，输入输出形状不变。"""

    def __init__(self, d_hid, n_position=200):
        """
        d_hid：位置向量维度 D，必须与输入词向量的最后一维一致。
        n_position：预先生成的位置数，即支持的最大序列长度，默认 200。
        """
        super(PositionalEncoding, self).__init__()

        # pos_table 形状为 [1, n_position, D]，前面的 1 用于广播到整个 batch。
        # 注册为 buffer 而不是可训练参数：不参与梯度更新，但随模型迁移设备，
        # 并保存在 state_dict 中；例如 model.to('cuda') 会同时迁移这张表。
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """生成不同位置、不同频率的编码，返回 [1, n_position, d_hid]。"""
        # 初始化时用 NumPy 生成一次，forward 直接使用注册的 PyTorch buffer。

        def get_position_angle_vec(position):
            # position：位置编号，从 0 开始；hid_j：特征维编号，从 0 开始。
            # hid_j // 2 让相邻的偶数维 2i 和奇数维 2i+1 共用同一个频率。
            # 角度 = position / 10000^(2i / d_hid)，不同维度对使用不同频率。
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # 每行对应一个位置，每列对应一个特征维，形状为 [n_position, d_hid]。
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 偶数维：PE(pos, 2i) = sin(pos / 10000^(2i / d_hid))。
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        # 奇数维：PE(pos, 2i+1) = cos(pos / 10000^(2i / d_hid))。
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        # 转成 float32 张量，再增加 batch 维：[n_position, D] → [1, n_position, D]。
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        x：[B,L,D] 的词向量表示；L 不超过 n_position，D 必须等于 d_hid。
        返回：[B,L,D]，原始词表示与位置向量逐元素相加，不拼接或扩大维度。
        """
        # x 为词向量表示，形状 [B, L, D]；L 必须不超过 n_position。
        # 截取前 L 个位置：[1, n_position, D] → [1, L, D]。
        # 与 x 相加时广播到 B 个样本，每个样本共享同一套位置编码。
        # 输出仍为 [B, L, D]：每个词的表示 = 词向量 + 对应的位置向量。
        # clone 复制位置表切片，detach 断开该切片的梯度，不会切断 x 的梯度。
        # 此处 buffer 本身不需要梯度且加法不修改原表，因此 clone/detach 可以省略。
        return x + self.pos_table[:, :x.size(1)].clone().detach()
