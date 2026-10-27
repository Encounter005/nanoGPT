#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

MODEL_DIR = Path.cwd() / "results" / "models"


class RunnerV3:
    """小批量训练 + 任意 metric + best-checkpoint（state_dict）。

    - model：nn.Module 子类
    - optimizer：torch.optim.Optimizer
    - loss_fn：例如 nn.CrossEntropyLoss()
    - metric_fn(out, y) -> float：标量评价指标；缺省时以 dev_loss 作 metric
    - higher_is_better：metric 越大越好（accuracy）还是越小越好（loss/MAE）
    - device：若给定（'cuda' / 'cpu' / torch.device），构造时把 model 搬到该 device，
      fit/_eval/predict 时自动把每个 batch 的张量也搬过去。默认 None 表示不接管 device。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Adam,
        loss_fn,
        metric_fn=None,
        higher_is_better: bool = True,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.higher_is_better = higher_is_better
        self.device = torch.device(device) if device is not None else None
        if self.device is not None:
            self.model = model.to(self.device)

        # log
        self.history = {
            "train_loss": [],
            "dev_loss": [],
            "dev_metric": [],
            "lr": [],
            "train_step_loss": [],
        }

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_metric, n = 0.0, 0.0, 0
        for batch in loader:
            *data, target = self._to_device(batch)
            output = self.model(*data)
            batch_size = data[0].size(0)
            total_loss += self.loss_fn(output, target).item() * batch_size
            if self.metric_fn is not None:
                total_metric += self.metric_fn(output, target) * batch_size
            n += batch_size

        dev_loss = total_loss / n
        dev_metric = (total_metric / n) if self.metric_fn is not None else dev_loss
        return dev_loss, dev_metric

    @torch.no_grad()
    def predict(self, *inputs):
        self.model.eval()
        if self.device is not None:
            inputs = tuple(self._to_device(t) for t in inputs)
        return self.model(*inputs)

    def evaluate(self, inputs: DataLoader) -> tuple[float, float]:
        return self._eval(inputs)

    def _to_device(self, batch):
        """
        moving tensor to the device
        """
        if self.device is None:
            return batch

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch}
        else:
            return batch

    def _do_save(self, best_path: str = ""):
        if best_path == "":
            return

        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / best_path)

    def load(self, best_path: str = ""):
        if best_path == "":
            return

        full_path = MODEL_DIR / best_path
        if not full_path.exists():
            raise FileNotFoundError(f"model weights file is not existed: {full_path}")

        map_location = self.device if self.device is not None else "cpu"

        state_dict = torch.load(full_path, weights_only=True, map_location=map_location)

        self.model.load_state_dict(state_dict)

    def summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print("-" * 10, "MODEL SUMMARY", "-" * 10)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device or 'cpu'}")
        print("-" * 35)

    def fit(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs=100,
        log_every=10,
        best_path: str = "",
        grad_clip_norm=None,
        lr_scheduler=None,
        lr_scheduler_step_per_batch: bool = False,
        patience=None,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        best = -float("inf") if self.higher_is_better else float("inf")
        no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss, n = 0.0, 0
            for batch in train_loader:
                *inputs, target = self._to_device(batch)

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(*inputs), target)
                loss.backward()

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip_norm
                    )  # 防止梯度爆炸
                self.optimizer.step()

                if lr_scheduler is not None and lr_scheduler_step_per_batch:
                    lr_scheduler.step()

                batch_size = inputs[0].size(0)
                running_loss += loss.item() * batch_size
                n += batch_size
                self.history["train_step_loss"].append(loss.item())

            train_loss = running_loss / n
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            should_log = (
                True
                if log_every is not None and (epoch + 1) % log_every == 0
                else False
            )
            if dev_loader is not None:
                dev_loss, dev_metric = self._eval(dev_loader)
                self.history["dev_loss"].append(dev_loss)
                self.history["dev_metric"].append(dev_metric)
                current_metric = dev_metric if self.metric_fn is not None else dev_loss
                isImproved = (
                    (current_metric > best)
                    if self.higher_is_better
                    else (dev_metric < best)
                )

                if isImproved:
                    best = dev_metric
                    no_improve = 0
                    if best_path is not None:
                        self._do_save(best_path)
                else:
                    no_improve += 1

                if should_log:
                    tag = " *" if isImproved else ""
                    print(
                        f"epoch {epoch + 1:4d} train_loss={train_loss:.4f} "
                        f"dev_loss={dev_loss:.4f} dev_metric={dev_metric:.4f}{tag}"
                    )

                if patience is not None and no_improve >= patience:
                    if log_every is not None:
                        print(
                            f"early stop at epoch:{epoch + 1} (no improvement for {patience} epochs)"
                        )
                    break

            elif should_log:
                print(f"epoch {epoch + 1:4d} train_loss={train_loss:.4f}")

            if lr_scheduler is not None and not lr_scheduler_step_per_batch:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = dev_loss if dev_loader is not None else train_loss
                    lr_scheduler.step(metric)
                else:
                    lr_scheduler.step()

        if best_path:
            self.load(best_path)

        self.summary()
        self.plot(num_epochs)

    def plot(self, num_epochs: int):
        """
        绘制训练历史曲线
        """
        # 检查是否有数据
        if len(self.history["train_loss"]) == 0:
            print("No training history to plot.")
            return

        # 计算实际训练的 epoch 数（可能因为 early stopping 少于 num_epochs）
        actual_epochs = len(self.history["train_loss"])
        epochs = range(1, actual_epochs + 1)

        # 判断需要绘制几个子图
        has_dev = len(self.history["dev_loss"]) > 0
        has_metric = len(self.history["dev_metric"]) > 0

        if has_dev and has_metric:
            # 有验证集：三个子图
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            # Loss 曲线
            ax1.plot(
                epochs,
                self.history["train_loss"],
                label="train_loss",
                marker="o",
                markersize=3,
            )
            ax1.plot(
                epochs,
                self.history["dev_loss"],
                label="dev_loss",
                marker="o",
                markersize=3,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Metric 曲线
            metric_name = "Metric" if self.metric_fn is not None else "Dev Loss"
            ax2.plot(
                epochs,
                self.history["dev_metric"],
                label=metric_name,
                marker="o",
                markersize=3,
                color="green",
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel(metric_name)
            ax2.set_title(f"Validation {metric_name}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Learning Rate 曲线
            ax3.plot(
                epochs,
                self.history["lr"],
                label="learning rate",
                marker="o",
                markersize=3,
                color="red",
            )
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        elif has_dev:
            # 只有验证损失，没有 metric：两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Loss 曲线
            ax1.plot(
                epochs,
                self.history["train_loss"],
                label="train_loss",
                marker="o",
                markersize=3,
            )
            ax1.plot(
                epochs,
                self.history["dev_loss"],
                label="dev_loss",
                marker="o",
                markersize=3,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Learning Rate 曲线
            ax2.plot(
                epochs,
                self.history["lr"],
                label="learning rate",
                marker="o",
                markersize=3,
                color="red",
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        else:
            # 只有训练损失：两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Loss 曲线
            ax1.plot(
                epochs,
                self.history["train_loss"],
                label="train_loss",
                marker="o",
                markersize=3,
                color="blue",
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Learning Rate 曲线
            ax2.plot(
                epochs,
                self.history["lr"],
                label="learning rate",
                marker="o",
                markersize=3,
                color="red",
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
