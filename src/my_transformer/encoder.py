"""完整编码器：词向量和位置编码 → 多层 EncoderLayer → 最终归一化。"""

import torch.nn as nn

from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    """读取完整源序列，输出可供 Decoder 交叉注意力检索的 memory。"""

    def __init__(
        self,
        n_src_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        n_position=200,
        scale_emb=False,
    ):
        """
        n_src_vocab：源词表大小；pad_idx：源 PAD 编号。
        d_word_vec：词向量维度，调用方应保证与 d_model（D）一致。
        n_layers：编码层数量；n_head：头数 H；d_k/d_v：每头 Q、K / V 维度。
        d_inner：FFN 中间维度；dropout：输入和各子层的丢弃概率。
        n_position：支持的最大源序列长度；scale_emb：是否把词向量乘 sqrt(D)。
        """
        super().__init__()
        # 按整数编号查词向量表：[B,Lsrc] → [B,Lsrc,d_word_vec]。
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # 为每个位置加入固定的位置向量，使注意力能够区分词的先后顺序。
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        # ModuleList 会注册每一层的参数；各层结构相同，但参数彼此独立。
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        # Pre-LN 堆叠末尾的最终归一化，不是对每个子层做 Post-LN。
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        src_seq：[B,Lsrc] 的词编号；src_mask：通常为 [B,1,Lsrc]，0/False 屏蔽。
        return_attns：是否额外收集每层 [B,H,Lsrc,Lsrc] 的注意力权重。
        默认返回 (enc_output,)；开启时返回 (enc_output, enc_slf_attn_list)。
        enc_output：[B,Lsrc,D]；通常作为 Decoder 的 Key/Value 来源。
        """

        enc_slf_attn_list = []  # 每个元素对应一层，不对不同层的权重求和

        # 1. 词编号查表，可选缩放，再加入位置编码并执行 dropout。
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(enc_output))

        # 2. 逐层融合上下文；每层输入/输出保持 [B,Lsrc,D]。
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        # 3. 对最终表示的 D 维特征归一化，作为完整 Encoder 的输出。
        enc_output = self.layer_norm(enc_output)

        # 4. 仅按开关改变返回内容；不收集权重也照常计算注意力。
        if return_attns:
            return enc_output, enc_slf_attn_list
        return (enc_output,)
