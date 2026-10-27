"""注册表：名字 → 构建函数。models/ 与 data/ 在 import 时自我注册。"""

from __future__ import annotations

from typing import Callable

from .configs import DataConfig, ModelConfig
from .protocols import BatchSource, TrainableModel
from .utils import ConfigError, suggest

MODELS: dict[str, Callable[[ModelConfig, int], TrainableModel]] = {}
DATASETS: dict[str, Callable[[DataConfig], BatchSource]] = {}


def register_model(name: str) -> Callable:
    """@register_model("gpt") 标在构建函数上，import 该模块即生效。"""

    def deco(fn: Callable[[ModelConfig, int], TrainableModel]) -> Callable:
        MODELS[name] = fn
        return fn

    return deco


def register_dataset(name: str) -> Callable:
    def deco(fn: Callable[[DataConfig], BatchSource]) -> Callable:
        DATASETS[name] = fn
        return fn

    return deco


def build_model(cfg: ModelConfig, vocab_size: int) -> TrainableModel:
    from . import models  # noqa: F401  import 即注册

    if cfg.name not in MODELS:
        raise ConfigError(f"未注册的模型 {cfg.name!r}{suggest(cfg.name, MODELS)}")
    return MODELS[cfg.name](cfg, vocab_size)


def build_dataset(cfg: DataConfig) -> BatchSource:
    from . import data  # noqa: F401  import 即注册

    if cfg.reader not in DATASETS:
        raise ConfigError(f"未注册的数据集实现 {cfg.reader!r}{suggest(cfg.reader, DATASETS)}")
    return DATASETS[cfg.reader](cfg)
