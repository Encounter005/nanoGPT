"""顶层模型：构造 mask，串联 Encoder/Decoder，再投影到目标词表。"""

import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .masks import get_pad_mask, get_subsequent_mask

class Transformer(nn.Module):
    """Pre-LN 编码器—解码器模型；返回词表 logits，不在本类中执行逐词生成。"""

    def __init__(
        self,
        n_src_vocab,
        n_target_vocab,
        src_pad_idx,
        target_pad_idx,
        d_word_vec=512,
        d_model=512,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_position=200,
        target_emb_project_weight_sharing=True,
        emb_src_target_weight_sharing=False,
        scale_emb_or_project="project",
    ):
        """
        n_src_vocab/n_target_vocab：源/目标词表大小；src_pad_idx/target_pad_idx：PAD 编号。
        d_word_vec：Embedding 维度；d_model：各层输入/输出维度 D，两者必须一致。
        d_inner：FFN 中间维度；n_layers：Encoder 和 Decoder 各自堆叠的层数。
        n_head：头数 H；d_k/d_v：每个头的 Q、K / V 维度。
        dropout：传给 Embedding 输出、注意力和 FFN 的丢弃概率。
        n_position：位置编码支持的最大序列长度。
        target_emb_project_weight_sharing：是否共享目标 Embedding 与词表投影权重。
        emb_src_target_weight_sharing：是否共享源/目标 Embedding，需同一套 token 编号。
        scale_emb_or_project：选择缩放 Embedding、logits 或均不缩放。
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx

        # 共享目标 Embedding/输出投影时，可选择缩放位置：
        # 'emb'：词向量乘 sqrt(D)；'project'：logits 除 sqrt(D)；'none'：不缩放。
        # 若不共享目标 Embedding/输出投影，下方会关闭这两种缩放。

        assert scale_emb_or_project in ["emb", "project", "none"]
        scale_emb = (
            (scale_emb_or_project == "emb")
            if target_emb_project_weight_sharing
            else False
        )
        self.scale_project = (
            (scale_emb_or_project == "project")
            if target_emb_project_weight_sharing
            else False
        )
        self.d_model = d_model

        # 两个独立堆叠：Encoder 提供源句 memory，Decoder 根据目标前缀读取它。
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            d_inner=d_inner,
            pad_idx=src_pad_idx,
            scale_emb=scale_emb,
            d_word_vec=d_word_vec,
            d_k=d_k,
            d_v=d_v
        )
        self.decoder = Decoder(
            n_target_vocab=n_target_vocab,
            n_position=n_position,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            d_inner=d_inner,
            pad_idx=target_pad_idx,
            scale_emb=scale_emb,
            d_word_vec=d_word_vec,
            d_k=d_k,
            d_v=d_v
        )

        # 每个目标位置的 D 维表示 → 目标词表分数；这些 logits 还不是概率。
        self.target_word_project = nn.Linear(d_model, n_target_vocab, bias=False)

        # 对矩阵参数使用 Xavier 初始化；LayerNorm 的一维参数保留默认值。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        assert d_model == d_word_vec
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if target_emb_project_weight_sharing:
            # 两处指向同一个 Parameter，而不是复制数值，训练更新也会共享。
            self.target_word_project.weight = self.decoder.target_word_emb.weight

        if emb_src_target_weight_sharing:
            # 仅适用于同一套 token→编号映射；词表大小相同不保证语义相同。
            # 当前下方校验被注释，调用方必须保证共享配置合法。
            # if n_src_vocab != n_target_vocab or src_pad_idx != target_pad_idx:
            #     raise ValueError("共享源/目标 Embedding 要求词表大小和 PAD 编号相同")
            self.encoder.src_word_emb.weight = self.decoder.target_word_emb.weight


    def forward(self, src_seq, target_seq):
        """
        src_seq：[B, Lsrc] 的整数 token 编号。
        target_seq：[B, Ltgt] 的 Decoder 输入；训练时由调用方右移标签并添加 BOS。
        返回：[B*Ltgt, n_target_vocab]，便于与展平的监督标签计算交叉熵。
        本函数不计算损失、不生成完整句子；target_seq 不是未经错位的完整答案。
        """
        # 1. 源 padding mask：[B,1,Lsrc]；目标 mask：[B,Ltgt,Ltgt]。
        # 目标同时屏蔽 PAD 和未来位置，True 表示允许关注。
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        target_mask = get_pad_mask(target_seq, self.target_pad_idx) & get_subsequent_mask(target_seq)

        # 2. 默认返回单元素元组，逗号用于解包；源 memory：[B,Lsrc,D]。
        enc_output, = self.encoder(src_seq, src_mask)
        # 3. Decoder 读取目标前缀和源 memory，输出：[B,Ltgt,D]。
        dec_output, = self.decoder(target_seq, target_mask, enc_output, src_mask)
        # 4. 词表投影：[B,Ltgt,D] → [B,Ltgt,n_target_vocab]。
        seq_logit = self.target_word_project(dec_output)
        if self.scale_project:
            seq_logit *= self.d_model ** -0.5

        # 5. 合并 batch 与目标长度；不做 softmax，交叉熵直接接收 logits。
        return seq_logit.view(-1, seq_logit.size(2))
