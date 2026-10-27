"""小型断言回归检查：Pre-LN、词表共享、mask 和半精度注意力。

运行：python -m src.NLP.c_transformer.my_transformer.test_transformer
当前模型的共享配置校验被注释，因此非法共享的预期拒绝检查会先失败；
保留该断言用于暴露这项既有问题，不在注释修改中放宽测试要求。
"""

import torch

from . import (
    Transformer, EncoderLayer, DecoderLayer, ScaledDotProductAttention,
    get_pad_mask, get_subsequent_mask,
)


def main():
    """运行固定种子的最小检查，断言失败即停止，不依赖额外测试框架。"""
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # 1. 用不同源/目标词表大小、d_k != d_v 检查维度与共享配置边界。
    config = dict(n_src_vocab=30, n_target_vocab=20, src_pad_idx=0, target_pad_idx=0,
                  d_word_vec=8, d_model=8, d_inner=16, n_layers=2, n_head=2,
                  d_k=4, d_v=3, dropout=0, n_position=8)
    model = Transformer(**config).eval()
    assert model.encoder.src_word_emb.weight is not model.decoder.target_word_emb.weight
    assert model.target_word_project.weight is model.decoder.target_word_emb.weight
    # 显式共享时应拒绝词表大小或 PAD 编号不一致；当前实现未启用该校验。
    for changes in ({}, {'n_src_vocab':20, 'src_pad_idx':1}):
        try:
            Transformer(**(config | changes), emb_src_target_weight_sharing=True)
        except ValueError:
            pass
        else:
            raise AssertionError('非法词表共享应被拒绝')
    shared = Transformer(**(config | {'n_src_vocab':20}), emb_src_target_weight_sharing=True)
    assert shared.encoder.src_word_emb.weight is shared.decoder.target_word_emb.weight

    # 2. 端到端 logits：[B*Ltgt,V]；交叉熵忽略 PAD，反向后所有参数应有有限梯度。
    src = torch.tensor([[1,29,2,0], [1,5,6,2]])
    tgt = torch.tensor([[1,7,2], [1,8,2]])
    logits = model(src,tgt)
    assert logits.shape == (6,20)
    torch.nn.functional.cross_entropy(logits, torch.tensor([7,2,0,8,2,0]), ignore_index=0).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any('.slf_attn.layer_norm.' in n or '.enc_attn.layer_norm.' in n or '.pos_ffn.layer_norm.' in n
                   for n, _ in model.named_parameters())

    # 3. 权重形状分别为源自注意力、目标自注意力、目标对源的交叉注意力。
    # 检查 PAD/未来位置权重为零，并改变最后的目标词以排查未来信息泄漏。
    sm = get_pad_mask(src,0)
    tm = get_pad_mask(tgt,0) & get_subsequent_mask(tgt)
    with torch.no_grad():
        enc, ea = model.encoder(src,sm,return_attns=True)
        dec, da, ca = model.decoder(tgt,tm,enc,sm,return_attns=True)
        assert all(a[0,:,:,3].abs().max() == 0 for a in ea + ca)
        assert all(torch.triu(a,diagonal=1).abs().max() == 0 for a in da)
        changed = tgt.clone()
        changed[:,-1] = 9
        other, = model.decoder(changed,tm,enc,sm)
        torch.testing.assert_close(other[:,:-1],dec[:,:-1],rtol=0,atol=1e-6)

    # 4. 关闭 dropout 并将计算分支置零：单层应保留原始残差，输出等于输入。
    # 若残差后仍存在 Post-LN，这个恒等映射检查就会失败。
    for layer in (EncoderLayer(8,16,2,4,3,0), DecoderLayer(8,16,2,4,3,0)):
        layer.eval()
        with torch.no_grad():
            for p in layer.parameters():
                p.zero_()
            x = torch.randn(2,3,8)
            result = layer(x) if isinstance(layer,EncoderLayer) else layer(x,torch.randn(2,4,8))
            torch.testing.assert_close(result[0],x,rtol=0,atol=0)

    # 5. 在 FP32/FP16/BF16 下检查屏蔽、dtype、有限输出及有限梯度。
    attention = ScaledDotProductAttention(2,attn_dropout=0)
    # 包括一整行全屏蔽和一行部分屏蔽；全屏蔽行输出和权重应为零。
    mask = torch.tensor([[[[False,False,False], [True,False,True]]]])
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        q = torch.randn(1,2,2,4,dtype=dtype,requires_grad=True)
        k = torch.randn(1,2,3,4,dtype=dtype,requires_grad=True)
        v = torch.randn(1,2,3,2,dtype=dtype,requires_grad=True)
        output, weights = attention(q,k,v,mask)
        assert output.dtype == weights.dtype == dtype
        assert torch.isfinite(output).all() and torch.isfinite(weights).all()
        assert output[:,:,0].count_nonzero() == weights[:,:,0].count_nonzero() == 0
        assert weights[:,:,:,1].count_nonzero() == 0
        torch.testing.assert_close(weights[:,:,1].sum(-1),torch.ones(1,2,dtype=dtype))
        reference = torch.softmax((q.float()/2) @ k.float().transpose(2,3),dim=-1).to(dtype)
        _, unmasked = attention(q,k,v)
        torch.testing.assert_close(unmasked,reference)
        output.float().sum().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))
        assert q.grad[:,:,0].count_nonzero() == 0

    # 6. 大点积分数会超出 float16 范围，但在 autocast 下也应稳定。
    with torch.autocast('cpu',dtype=torch.float16):
        q = torch.full((1,1,1,4),300,dtype=torch.float16)
        output, weights = attention(q,q,torch.ones_like(q),torch.ones(1,1,1,1,dtype=torch.bool))
        assert torch.isfinite(output).all() and torch.isfinite(weights).all()
        torch.testing.assert_close(output,torch.ones_like(output))
    print('通过：Pre-LN、全部参数梯度、词表共享、mask、FP16/BF16 和 autocast')


if __name__ == '__main__':
    main()
