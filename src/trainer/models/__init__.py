"""模型实现：import 本包即完成注册。加新模型 = 这里加一行 import。"""

import importlib

from . import gpt, toy_rnn  # noqa: F401

# 仓库根 src/ 下的自定义模型（gpt.py、gptv1.py …）：import 即注册；作为独立包装走时跳过
for _name in ("gpt", "gptv1"):
    try:
        importlib.import_module(_name)
    except ImportError:
        pass

__all__ = ["gpt", "toy_rnn"]
