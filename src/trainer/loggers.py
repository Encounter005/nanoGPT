"""日志 sink：控制台 / metrics.jsonl / TensorBoard / wandb。训练器只认 log(step, metrics, split)。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .configs import LogConfig


class ConsoleLogger:
    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None:
        body = " | ".join(f"{k} {v:.4g}" for k, v in metrics.items())
        print(f"step {step:5d} | {split:<5} | {body}", flush=True)

    def close(self) -> None:
        pass


class FileLogger:
    """每行一条 JSON，sweep 汇总就靠读这个文件。"""

    def __init__(self, path: Path):
        self.path = path

    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "split": split, **metrics}) + "\n")

    def close(self) -> None:
        pass


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{split}/{key}", value, step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    def __init__(self, cfg: LogConfig, run_name: str, snapshot: dict[str, Any]):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=cfg.wandb_project, name=run_name, config=snapshot)

    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None:
        self.wandb.log({f"{split}/{key}": value for key, value in metrics.items()}, step=step)

    def close(self) -> None:
        self.wandb.finish()


def build_loggers(cfg: LogConfig, out_dir: Path, run_name: str, snapshot: dict[str, Any]) -> list:
    loggers: list = []
    if cfg.console:
        loggers.append(ConsoleLogger())
    if cfg.file:
        loggers.append(FileLogger(out_dir / "metrics.jsonl"))
    if cfg.tensorboard:
        try:
            loggers.append(TensorBoardLogger(out_dir / "tb"))
        except ImportError:
            print("跳过 TensorBoard：未安装 tensorboard（uv add tensorboard）")
    if cfg.wandb:
        try:
            loggers.append(WandbLogger(cfg, run_name, snapshot))
        except ImportError:
            print("跳过 wandb：未安装 wandb（uv add wandb）")
    return loggers
