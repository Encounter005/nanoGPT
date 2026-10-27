"""完整解码器：目标前缀表示 → 多层 DecoderLayer → 最终归一化。"""

import torch.nn as nn

from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    """融合目标上下文和源 memory，返回目标位置的特征，不直接选择词编号。"""

    def __init__(
        self,
        n_target_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        n_position=200,
        dropout=0.1,
        scale_emb=False,
    ):
        """
        n_target_vocab：目标词表大小；pad_idx：目标 PAD 编号。
        d_word_vec：Embedding 维度，调用方应保证与 d_model（D）一致。
        n_layers：解码层数量；n_head：头数 H；d_k/d_v：每头 Q、K / V 维度。
        d_inner：FFN 中间维度；n_position：支持的最大目标输入长度。
        dropout：输入和各子层的丢弃概率；scale_emb：是否把词向量乘 sqrt(D)。
        """
        super().__init__()

        # 目标词编号 → 词向量；与源 Embedding 默认是独立的表。
        self.target_word_emb = nn.Embedding(
            n_target_vocab, d_word_vec, padding_idx=pad_idx
        )
        # 固定位置编码与 dropout：维度不变，补充目标序列的位置信息。
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        # 注册多个参数独立的解码层，每层都读取同一个 Encoder memory。
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        # Pre-LN 堆叠末尾再做一次最终归一化。
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(
        self, target_seq, target_mask, enc_output, src_mask, return_attns=False
    ):
        """
        target_seq：[B,Ltgt]，训练时为错位目标输入，推理时为已生成的前缀。
        target_mask：通常为 [B,Ltgt,Ltgt]，同时屏蔽目标 PAD 与未来位置。
        enc_output：[B,Lsrc,D]，源 memory；src_mask：通常为 [B,1,Lsrc]。
        return_attns：是否返回各层的目标自注意力、交叉注意力权重列表。
        默认返回 (dec_output,)；开启时返回输出和上述两个列表。
        dec_output：[B,Ltgt,D]，还需经顶层词表投影才能得到 logits。
        """

        # 两个列表分别按层保存 [B,H,Ltgt,Ltgt]、[B,H,Ltgt,Lsrc] 的权重。
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # 1. 查目标词向量，再加入位置编码；训练/推理共用这个步骤。
        dec_output = self.target_word_emb(
            target_seq
        )  # 本层只接收输入编号；是否使用 teacher forcing 由调用方决定。
        if self.scale_emb:
            dec_output *= self.d_model**0.5
        dec_output = self.dropout(self.position_enc(dec_output))

        # 2. 逐层执行目标自注意力、交叉注意力、FFN，形状保持 [B,Ltgt,D]。
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=target_mask,
                dec_enc_attn_mask=src_mask,
            )
            if return_attns:
                dec_slf_attn_list.append(dec_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)
        
        # 3. 堆叠结束后归一化；不改变 batch、目标长度或特征维度。
        dec_output = self.layer_norm(dec_output)

        # 4. 返回特征与可选权重；这两个列表用于分析各层关注了哪些位置。
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return (dec_output,)
