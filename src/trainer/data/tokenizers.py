"""分词器适配：把不同来源的 encode/decode 统一成 BatchSource 需要的样子。

[data] tokenizer 的写法：
    ""                        自动：有 meta.pkl 用字符级字表，否则 GPT-2 BPE（保持旧行为）
    "tiktoken:p50k_base"      tiktoken 内置编码（gpt2 / r50k_base / p50k_base；词表须 ≤ 65535）
    "hf:bert-base-uncased"    HuggingFace 上的 tokenizer 仓库（首次需要网络下载）

注意：分词器必须和数据准备时用的一致，否则 id 能对上、文本解出来是乱码。
"""

from __future__ import annotations

from typing import Protocol

from ..utils import DataError


class Tokenizer(Protocol):
    """数据集只需要这三样：词表大小 + 编解码。"""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class _TiktokenTokenizer:
    def __init__(self, name: str):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - tiktoken 是直接依赖
            raise DataError("使用 tiktoken 分词器需要 tiktoken（uv add tiktoken）") from exc
        try:
            self._enc = tiktoken.get_encoding(name)
        except ValueError as exc:
            available = ", ".join(tiktoken.list_encoding_names())
            raise DataError(f"未知的 tiktoken 编码 {name!r}；可用：{available}") from exc
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class _HFTokenizer:
    def __init__(self, repo_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - transformers 是直接依赖
            raise DataError("使用 HF 分词器需要 transformers（uv add transformers）") from exc
        try:
            self._tok = AutoTokenizer.from_pretrained(repo_id)
        except Exception as exc:  # noqa: BLE001 - 网络/仓库名/缓存问题，统一转中文提示
            raise DataError(f"加载 HF 分词器 {repo_id!r} 失败（需要网络或本地缓存）：{exc}") from exc
        self.vocab_size = len(self._tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


def build_tokenizer(spec: str) -> Tokenizer:
    """把 "tiktoken:名字" / "hf:仓库id" 变成分词器对象；空串由调用方决定。"""
    kind, _, name = spec.partition(":")
    kind, name = kind.strip(), name.strip()
    if not name:
        raise DataError(f"[data] tokenizer 写法应为 'tiktoken:gpt2' 或 'hf:repo/id'，收到 {spec!r}")
    if kind == "tiktoken":
        return _TiktokenTokenizer(name)
    if kind == "hf":
        return _HFTokenizer(name)
    raise DataError(f"未知的 tokenizer 类型 {kind!r}：只支持 tiktoken / hf")
