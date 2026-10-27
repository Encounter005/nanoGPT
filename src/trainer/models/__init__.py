"""模型实现：import 本包即完成注册。加新模型 = 这里加一行 import。"""

from . import gpt, toy_rnn  # noqa: F401

try:  # 仓库里自己的模型（src/gpt.py）：存在就注册，作为独立包装走时跳过
    import gpt  # noqa: F401
except ImportError:
    pass

__all__ = ["gpt", "toy_rnn"]
