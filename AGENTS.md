# Repository guidance

## Scope and entry points

Two independent stacks share this repository; know which one you are changing.

- **Root nanoGPT (upstream tutorial code)**: `model.py` (GPT), `train.py`, `sample.py`, `bench.py`, `configurator.py`, `config/*.py`. These run as scripts from the repository root and should keep upstream behavior.
- **`src/my_trainer` (packaged, config-driven template)**: entry points are `python -m my_trainer.train|sample|bench|sweep [config.toml]`. Import name is `my_trainer`, distribution name is `nanogpt`; `pyproject.toml` maps the two with `[tool.uv.build-backend] module-name = "my_trainer"` (removing it breaks `uv run`/`uv build` with `Expected a Python module at: src/nanogpt/__init__.py`). Behavior is driven entirely by one TOML file — there are **no command-line overrides**.
- `docs/training-template/` documents the template: `使用手册.md` (usage, config reference, troubleshooting) and `config.template.toml` (every field, commented; must stay equal to the dataclass defaults).
- `configs/*.toml` are runnable template experiments (`debug_cpu.toml`, `shakespeare_char.toml`, `sweep_lr.toml`, `gpt_v0.toml`, `gpt_v1.toml`). `config/*.py` (singular) are executable overrides for the **root** scripts only — do not mix the two directories.
- Other modules under `src/` (`gpt.py`, `gptv1.py`, `lm.py`, `config.py`, `runner.py`, `my_transformer/`, …) are the author's own experiments. `src/gpt.py` registers `gpt_v0` and `src/gptv1.py` registers `gpt-v1`; both are pulled in only by `[model] location` in `configs/gpt_v0.toml` / `configs/gpt_v1.toml` (`src/my_trainer/models/__init__.py` imports just the built-in `gpt` / `toy_rnn`). A model file that nothing imports never runs its `@register_model` and fails at config time with 未注册的模型 — point `[model] location` at it.
- `test_lm.py` targets `src/lm.py`, but it calls `LM(16, block_size=5)` while `src/lm.py` defines `LM(cfg)` — it is stale. Do not treat it as a test suite.

## Environment and configuration

- Python 3.13 (`.python-version`), `uv` + `uv.lock`. The `.venv` has the project installed editable through a `.pth` that puts `src/` on `sys.path`, so `python -m my_trainer.*` works from the repository root without reinstalling — and top-level names `utils`, `config`, `gpt`, `lm`, `test` resolve into `src/`.
- Template entry points must run **from the repository root**: `src/my_trainer/models/gpt.py` lazily `import model`, and dataset paths are relative (`data/...`). Anywhere else fails with `找不到根目录 model.py`.
- With `uv` the invocation is `uv run python -m my_trainer.train configs/debug_cpu.toml`. `uv run my_trainer.train ...` fails (`Failed to spawn`); `uv` rebuilds the package whenever `pyproject.toml` changes.
- Template config rules: exactly one optional positional argument (TOML path, default `configs/debug_cpu.toml`); unknown sections/keys or wrong value types raise Chinese `ConfigError`s with spelling suggestions; exit code 1 = config/data problem, 2 = unexpected exception.
- Root scripts use `configurator.py`, which `exec`s the positional config file and then applies `--key=value` in argument order. Override names must already exist and value types must match exactly (`--compile=False`, not a bare flag). `train.py` defaults to CUDA, `torch.compile`, and a large OpenWebText run; importing `train.py`, `sample.py`, or `bench.py` executes their workloads.

## Data and output handling

- `.gitignore` covers `input.txt`, `*.bin`, `*.pkl`, `*.pt`, `__pycache__/`, `runs/`. These are generated artifacts: never hand-edit or commit them.
- `data/<name>/train.bin` and `val.bin` are `uint16` token ids, so any vocabulary must stay ≤ 65535 ids (`cl100k_base`/`o200k_base` are rejected at load with a Chinese `DataError`). The tokenizer comes from `[data] tokenizer` (adapter: `src/my_trainer/data/tokenizers.py`): empty → `meta.pkl` `stoi`/`itos` if present, else GPT-2 BPE; `tiktoken:<name>` or `hf:<repo id>` selects another tokenizer and must match how the bins were encoded — a mismatch trains normally but samples decode as garbage. Prepared sets: `data/shakespeare_char` (char-level, vocab 65), `data/chinese` (char-level, vocab 4244), `data/shakespeare` (GPT-2 BPE, no `meta.pkl`, vocab 50257). `data/chinese/prepare.py` reads the local `input.txt`; `data/shakespeare_char/prepare.py` downloads when the text is missing (needs network + `requests`). `[data] reader = "hf"` downloads from HuggingFace and is unverified in this repository.
- Template runs write `runs/<run.name>/`: `ckpt.pt`, `metrics.jsonl`, `config.toml` (snapshot = reproduction entry), `samples.txt`, `tb/`. To verify without disturbing existing experiments, copy a config and change `[run] name`.
- Template checkpoints store the **uncompiled** model state dict (no `_orig_mod.` prefix), so `[train] compile` can be toggled between runs; `[run] resume` is `auto` / `none` / an explicit path.
- Root scripts write `ckpt.pt` under their `out_dir`; `config/train_shakespeare_char.py` uses `out-shakespeare-char`.

## Focused verification

No formatter, linter, type-checker, test runner, or CI workflow is configured. Do not claim a configured suite exists.

- Template syntax only: `python -m compileall -q src/my_trainer`.
- Config validity only: `python -c "from my_trainer.configs import Config; Config.load('configs/debug_cpu.toml')"`.
- Vocabulary/tokenizer check (no training): `python -c "from my_trainer.configs import DataConfig; from my_trainer.registry import build_dataset; print(build_dataset(DataConfig()).vocab_size)"` → 65 for the default char-level set; pass `DataConfig(name='shakespeare')` (50257) or a `tokenizer=` spec to test other paths.
- Template smoke (needs prepared `data/shakespeare_char`, CPU, ~40 steps): `python -m my_trainer.train configs/debug_cpu.toml`, then check `runs/debug-cpu/{ckpt.pt,metrics.jsonl,config.toml}` exist. `python -m my_trainer.bench configs/debug_cpu.toml` is the fastest timing probe and writes no checkpoint.
- Root `model.py` probe (PyTorch only, no dataset or checkpoint):

  ```sh
  python -c 'import torch; from model import GPT, GPTConfig; m = GPT(GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=1, n_embd=8)); x = torch.randint(16, (2, 8)); logits, loss = m(x, x); assert logits.shape == (2, 8, 16); assert torch.isfinite(loss); loss.backward(); logits, loss = m(x); assert logits.shape == (2, 1, 16) and loss is None'
  ```

  `GPT.forward` returns full-sequence logits only when targets are supplied; inference returns only the last position. Preserve this distinction.
- Root training: `python train.py config/train_shakespeare_char.py` (prepared char data, CUDA by default). Root `bench.py` defaults to CUDA + OpenWebText and is not a substitute for the template smoke.
- Root sampling: `python sample.py --out_dir=out-shakespeare-char --device=cpu` needs a trusted `ckpt.pt` plus tiktoken. Template sampling: `python -m my_trainer.sample configs/debug_cpu.toml` (reads `runs/debug-cpu/ckpt.pt`).

## Conventions that tooling does not enforce

- `src/my_trainer`, `configs/*.toml`, and `docs/training-template/` are written in Chinese (docstrings, comments, error messages); match that when editing them.
- One config source of truth: a new knob goes into the dataclass in `src/my_trainer/configs.py`, into `validate()`, into `docs/training-template/config.template.toml` (kept equal to the defaults), and into the manual. Do not add CLI flags that override TOML.
- New model: satisfy `TrainableModel` in `src/my_trainer/protocols.py` (`__init__(cfg, vocab_size)`, `(B,T,V)` logits plus loss when targets are given, `generate` accepting `temperature`/`top_k`, `num_params`) and register it with `@register_model`; then either import it from `src/my_trainer/models/__init__.py` (in-package model) or keep it anywhere as a `.py` file and point `[model] location` at it (`src/my_trainer/registry.py:load_source`, imported before the model is built). New data source: `@register_dataset` plus an import in `src/my_trainer/data/__init__.py`. New tokenizer kind: satisfy `Tokenizer` in `src/my_trainer/protocols.py` and register it with `@register_tokenizer`; `[data] tokenizer` is `"<kind>:<param>"` and the param is passed to the builder verbatim. User extension files are declared per section — `[model] location` and `[data] tokenizer_location` (both `.py` paths relative to the repo root, imported by `load_source`, validated to exist in `Config.validate`) — so adding one never edits `src/my_trainer`. A file that registers both a model and a tokenizer must be named in both fields; it is imported once (dedupe on module name).
- All metrics go through `Logger.log(step, metrics, split)` so `metrics.jsonl`, TensorBoard, and `sweep` stay consistent; `[log] progress = true` deliberately suppresses per-step console lines in favour of a tqdm bar.
- Two config classes coexist: `src/config.py:GPTConfig` (`num_embed`/`num_heads`/`num_layers`, used by `src/lm.py` + `my_transformer/`) and `my_trainer.configs.ModelConfig` (`n_embd`/`n_head`/`n_layer`, used by the template). Do not swap field names.

## Trust and credentials

- `configurator.py` `exec`s config files, `meta.pkl` is loaded with `pickle.load`, and template checkpoints use `torch.load(..., weights_only=False)`. Only load configs and artifacts you trust from this repository.
- `wandb` is disabled by default; do not enable it or hit the network merely to verify a change. Dependency changes go through `uv` (index is a mirror configured in `pyproject.toml`) and must update `uv.lock`.
- Do not read `.pi/`, `.venv/`, or credential/secret files.

## Completion

Report the exact command and its outcome, or state the missing prerequisite (dataset, GPU, checkpoint) instead of skipping silently. Keep `runs/`, `dist/`, `data` artifacts, and unrelated scratch scripts under `src/` out of implementation diffs. A template smoke run says nothing about the root `train.py` path, and vice versa.
