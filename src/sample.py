#!/usr/bin/env python3
"""用 results/models/gpt 做一次生成：python src/sample.py "ROMEO:" """

import sys
from pathlib import Path

import torch
import torch.nn as nn

from config import GPTConfig
from dataset import CharDataset
from gpt import GPTLanguageModel
from lm import LM

CKPT = Path("results/models/gpt")


def load_checkpoint(path: Path, cfg: GPTConfig) -> nn.Module:
    """加载 RunnerV3 存下的 state_dict，并按键判断它属于哪个模型。

    Runner 保存的是 LMAdapter 的 state_dict，每个键多一层 "model." 前缀；
    如果训练时还开了 torch.compile，还会多一层 "_orig_mod."，一并剥掉。
    """
    state = torch.load(path, map_location=cfg.device)
    state = {
        k.removeprefix("model.").replace("_orig_mod.", ""): v for k, v in state.items()
    }

    # lm.py 的 LM 包着 my_transformer.Encoder，键里带 backbone.
    model = LM(cfg) if any(k.startswith("backbone.") for k in state) else GPTLanguageModel(cfg)
    model.load_state_dict(state)
    return model.to(cfg.device).eval()


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    if not prompt:
        raise SystemExit("输入不能为空")

    cfg = GPTConfig()
    dataset = CharDataset(cfg)  # 必须先建，它会回填 cfg.vocab_size
    try:
        ids = dataset.encode(prompt)
    except KeyError as e:
        raise SystemExit(f"输入里有词表外的字符 {e.args[0]!r}")

    model = load_checkpoint(CKPT, cfg)
    idx = torch.tensor(ids, dtype=torch.long, device=cfg.device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=1000)
    print(dataset.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
