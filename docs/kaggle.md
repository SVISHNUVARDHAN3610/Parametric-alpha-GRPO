# Kaggle runbook

Use a Linux Kaggle session with GPU T4 ×2 and Internet access. Place a checkout of this repository in a writable location and run the commands below from its root in that remote session. Install `uv` into the session first if Kaggle does not provide it. The setup selects the CUDA 12.9 PyTorch wheel family for the vLLM 0.21.0 build. A driver, wheel, Torch ABI, or SM75 kernel incompatibility blocks the feasibility run. Capture the installer and worker logs before changing dependencies. [vLLM 0.21.0 GPU requirements](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/)

## Prepare and validate

### Run components from notebook cells

Select **GPU T4 ×2**, enable Internet, and change into your repository checkout with `%cd`. Run the following cells in order. The `PY` variable points at the separate Python 3.11 environment used by the project.

```python
!nvidia-smi -L
!uv venv --python 3.11 /kaggle/temp/alpha-grpo-venv
PY = "/kaggle/temp/alpha-grpo-venv/bin/python"
A = "/kaggle/working/alpha-grpo"
%env HF_HOME=/kaggle/temp/alpha-grpo-hf
!uv pip install --python {PY} --torch-backend=cu129 -e '.[kaggle,evaluation,test]'
!{PY} -m nltk.downloader punkt punkt_tab
```

Install `uv` in the Kaggle session with `!python -m pip install uv` if the command is unavailable. Run each of these checks in its own cell so you can see which component fails.

```python
!{PY} -m pytest -q tests/test_objective.py
!{PY} -m pytest -q tests/test_data_and_verifiers.py
!{PY} -m pytest -q tests/test_sync_and_recovery.py
```

The unit tests run without a model download. The following GPU probes download the configured starting model and use a fixed prompt. `trainer` scores two synthetic completions and performs one optimizer update on GPU 0. `policy` generates and scores a short response on GPU 1. `reference` scores a fixed response on GPU 1. `both` keeps both workers alive together and scores the same generated tokens through both engines. Each command prints a JSON report. Worker logs default to `$A/component-checks/<component>`.

```python
!{PY} scripts/check_components.py trainer
!{PY} scripts/check_components.py policy
!{PY} scripts/check_components.py reference
!{PY} scripts/check_components.py both
```

After the component checks, use the Python launcher with the same stage names and flags as the commands below. The one-update preflight exercises the full training and weight transfer path with the curated dataset.

```python
!{PY} scripts/kaggle.py prepare-data --out {A}/prepared
!{PY} scripts/kaggle.py prescreen --data {A}/prepared --out {A}/curated
!{PY} scripts/kaggle.py preflight --data {A}/curated --out {A}/checkpoint-gate --sync checkpoint
!{PY} scripts/kaggle.py preflight --data {A}/curated --out {A}/nccl-gate --sync nccl
```

### Run stages from Bash

```bash
bash scripts/kaggle.sh setup
export A=/kaggle/working/alpha-grpo
bash scripts/kaggle.sh prepare-data --out "$A/prepared"
bash scripts/kaggle.sh prescreen --data "$A/prepared" --out "$A/curated"
bash scripts/kaggle.sh preflight --data "$A/curated" --out "$A/checkpoint-gate" --sync checkpoint
bash scripts/kaggle.sh preflight --data "$A/curated" --out "$A/nccl-gate" --sync nccl
```

Run `pytest tests` in the remote virtual environment at `/kaggle/temp/alpha-grpo-venv`. The preflight runs one optimizer update, checks policy scoring against Transformers and teacher-forced scoring, requests an invalid policy version, and checks that reference scores stay fixed. Inspect `gate.json`, `steps.jsonl`, `resources.jsonl`, and the worker logs. GPU memory is sampled at 250 ms intervals, so use additional profiling for short-lived allocations if the observed margin is close to 1 GiB.

Use fresh directories for each gate. Budget storage for the Hub cache, a temporary 2.8 GiB inference export, full optimizer checkpoints, and persistent evaluation snapshots. Choose `--checkpoint-root` on a suitable writable volume before starting. CPU AdamW needs about 16.5 GiB for model and optimizer tensors after the first update. Two full CPU checkpoints plus the inference snapshot can exceed the Kaggle working volume. A model export alone cannot resume training.

If GPU 0 fails at microbatch 1, repeat both gates with `--optimizer cpu` and fresh output and checkpoint directories. If the checkpoint gate works and NCCL fails, use `--sync checkpoint` for the study. Checkpoint synchronization restarts the policy after every optimizer step, so include its time in the feasibility estimate. If NCCL hangs inside CUDA, the watchdog records `nccl-timeout` and exits with status 75. Resume from the latest complete checkpoint with `--resume --sync checkpoint`. A failure between durable checkpoints can replay completed updates.

## Smoke and resume

```bash
bash scripts/kaggle.sh smoke --data "$A/curated" --out "$A/smoke" --max-new-steps 15
bash scripts/kaggle.sh smoke --data "$A/curated" --out "$A/smoke" --resume
```

Pass your selected `--optimizer`, `--sync`, and storage options on both commands. Confirm the second segment starts at version 15 and finishes at version 30. Review free GPU memory on both devices, host memory, disk, step time, and token throughput before committing a session to a full run. A 1 GiB minimum sampled margin is the configured gate. The original proposal's step-time target is informational.

## Run the study in stages

```bash
bash scripts/kaggle.sh train --data "$A/curated" --out "$A/control-a05-42" \
  --experiment control-a05 --seed 42
```

The default target is 375 successful updates. Use `--max-new-steps` to stop at a checkpoint boundary, and `--resume` with the original arguments and paths for the next session. The runtime saves recovery state at step 0, every 75 successful updates, and on graceful stop. It exports FP16 evaluation snapshots at steps 125, 250, and 375. Copy complete checkpoints and manifests to durable storage before the session ends. Use a separate run directory and checkpoint directory for each experiment and seed. Use the same curated manifest, selected optimizer, and microbatch for all comparisons.

## Evaluate and analyze

Stop training before evaluation. Use the same benchmark revisions and settings for published baselines and trained snapshots. A first evaluation can use `--limit 5 --greedy-only` in a distinct output directory. Full generative evaluation includes greedy and three sampled runs by default.

```bash
bash scripts/reproduce_eval.sh --model allenai/OLMo-2-0425-1B-DPO \
  --data "$A/curated" --out "$A/eval-dpo" --label dpo
bash scripts/reproduce_eval.sh --model "$A/control-a05-42/snapshots/step-000375" \
  --data "$A/curated" --out "$A/eval-control-a05-42" --label control-a05 --training-seed 42
bash scripts/kaggle.sh analyze --runs "$A/control-a05-42" \
  --evaluations "$A/eval-dpo" "$A/eval-control-a05-42" \
  --baseline "$A/eval-dpo" --out "$A/report"
```

Use `--device cpu` for evaluation when necessary. Evaluate `allenai/OLMo-2-0425-1B-SFT` and `allenai/OLMo-2-0425-1B-RLVR1` with the same command. Use `--revision` to select an explicit Hub model revision. The `export` command copies a complete model snapshot to a separate directory and optionally uploads it when you explicitly supply `--hub-repo` and configured credentials.
