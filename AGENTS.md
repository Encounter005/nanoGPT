# Repository guidance

## Scope and entry points

- The training implementation is in root-level `model.py`, `train.py`, and `sample.py`; `bench.py` benchmarks the training workload. Run these scripts from the repository root: configuration and dataset paths are relative to the working directory.
- `src/nanogpt/__init__.py` is a separate, minimal packaged entry point. The `nanogpt` console command only prints a greeting; it does not launch training.
- `test.py` is a character-level Bigram learning/demo script, not a test suite. It reads `data/shakespeare_char/input.txt` at import time and prints batches and output shapes when run.
- `config/*.py` supplies executable overrides for the root scripts. See [README.md](README.md) for training, evaluation, sampling, and distributed examples.

## Environment and configuration

- `.python-version` pins Python 3.13; `pyproject.toml` requires Python >=3.13 and declares dependencies, the `uv_build` backend, and the console entry point. `uv.lock` is the dependency lockfile. Consult the manifest for dependency requirements rather than treating the README's unversioned install command as the full environment specification.
- Shakespeare preparation scripts also import `requests`, which is not a direct dependency in `pyproject.toml`; verify it is available before preparing data.
- `configurator.py` executes positional Python configuration files in script globals, then applies `--key=value` overrides in argument order. Override names must already exist and value types must match exactly; use `--compile=False`, not a bare flag.
- `train.py` defaults to CUDA, compilation, and a large OpenWebText run. Do not use bare `python train.py` as a quick check. Importing `train.py`, `sample.py`, or `bench.py` also executes their workloads.

## Data and output handling

- `python data/shakespeare_char/prepare.py` downloads input text if absent and writes `train.bin`, `val.bin`, and `meta.pkl` beside the script. It requires NumPy and requests, and network access when the text is missing.
- Training reads `data/<dataset>/train.bin` and `val.bin` as `uint16` token IDs, converting batches to integer tensors. Targets are shifted one token ahead of inputs. Keep preparation and loading formats consistent.
- Character-level vocabulary mappings live in `meta.pkl`; sampling uses them when available, otherwise it assumes GPT-2 tokenization. Do not substitute BPE-prepared Shakespeare data for character-level data without updating the configuration and encoding path.
- `.gitignore` excludes `input.txt`, `*.bin`, `*.pkl`, and `*.pt`. These are dataset/checkpoint artifacts, not source files; do not hand-edit or include them in code changes.
- Training writes `ckpt.pt` under `out_dir`; the Shakespeare config points to `out-shakespeare-char`. Use a separate output directory for verification so existing training results are not overwritten.
- Configuration files are executed with `exec`, metadata is loaded with `pickle.load`, and checkpoints use `torch.load`. Only run reviewed configuration and load trusted local artifacts; do not use these paths to inspect untrusted downloads. Do not inspect credential files or enable `wandb_log` merely to verify a change.

## Focused verification

No formatter, linter, type-checker, test runner, or CI workflow is configured in the inspected repository. Do not claim a configured test suite exists.

- For `test.py` changes: `python test.py`, with PyTorch installed and `data/shakespeare_char/input.txt` present. The current demo prints a final shape of `torch.Size([4, 8, vocab_size])`; it has no pass/fail assertions.
- For `model.py` changes, this small CPU forward/backward probe needs only PyTorch and no dataset or checkpoint:

  ```sh
  python -c 'import torch; from model import GPT, GPTConfig; m = GPT(GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=1, n_embd=8)); x = torch.randint(16, (2, 8)); logits, loss = m(x, x); assert logits.shape == (2, 8, 16); assert torch.isfinite(loss); loss.backward(); logits, loss = m(x); assert logits.shape == (2, 1, 16) and loss is None'
  ```

  `GPT.forward` returns full sequence logits only when targets are supplied; inference returns only the last position. Preserve this distinction.

- Common training command: `python train.py config/train_shakespeare_char.py`. Requires prepared character-level data and, with its defaults, CUDA. The README's **quick start** section gives the CPU command with `--device=cpu --compile=False` and a smaller model; this is training, not a fast unit test.
- Common sampling command: `python sample.py --out_dir=out-shakespeare-char --device=cpu`. Requires a trusted `ckpt.pt` in that directory plus PyTorch and tiktoken. Character-level sampling also needs matching metadata.
- `bench.py` defaults to CUDA, compilation, and prepared OpenWebText data; it is not a substitute for a focused correctness check.

## Completion

Use the smallest applicable behavioral check above or a change-specific probe. Report the exact command and outcome, or the missing prerequisite if verification cannot run. Do not count the greeting console entry point or the Bigram demo as verification of Transformer training. Keep generated data, checkpoints, and unrelated learning-script changes out of the implementation diff.
