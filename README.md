# Parametric α-GRPO on OLMo 2

Research code for a full-parameter study of how completion length weighting affects verifiable RL, response length, and group filtering. The registered comparisons use OLMo-2-0425-1B-DPO and a shared 1,500-prompt mixture of GSM8K, MATH, and instruction constraints.

| Compute | Role |
| --- | --- |
| T4 GPU 0 | Transformers policy, custom PyTorch α-GRPO, gradients, and optimizer |
| T4 GPU 1 | Two independent FP16 vLLM processes, each configured for 45% of memory |
| CPU | Reward verification, orchestration, and optional AdamW optimizer offload |

The policy vLLM process generates and supplies old-policy scores. The reference vLLM process scores responses with the original DPO weights for the full run. The trainer synchronizes policy weights before the next rollout, using NCCL when available and a versioned FP16 checkpoint restart otherwise.

This repository was authored on a development-only laptop. Dependency installation, tests, model loading, data preparation, and experiments are for the separate Linux environment. Kaggle compatibility and results are pending remote validation.

- [Kaggle runbook](docs/kaggle.md) covers setup, feasibility, resume, evaluation, and export.
- [Research protocol](docs/methods.md) records the objective, experiment matrix, provenance, and differences from the original proposal.
- [Kaggle notebook](notebooks/kaggle.ipynb) provides a thin launcher for one stage at a time.
- [Study config](configs/study.yaml) defines the eleven named configurations and sixteen registered runs.

Source code uses the [MIT license](LICENSE). Upstream models, data, and verifier code retain their own licenses.
