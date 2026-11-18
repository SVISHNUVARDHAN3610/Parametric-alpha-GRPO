# Parametric α-GRPO on OLMo 2

**How much weight should a 200-token wrong answer receive alongside a 20-token correct one?** This study varies response-length weighting in Group Relative Policy Optimization (GRPO) and measures how it changes mathematical accuracy, instruction following, and the length of correct and incorrect responses.

**Study status** · Implementation available. Kaggle dual-T4 validation, training runs, and benchmark results are pending.

[Research protocol](docs/methods.md) · [Experiment configurations](configs/study.yaml) · [Kaggle notebook](notebooks/kaggle.ipynb) · [Reproduction guide](docs/kaggle.md)

## Problem and hypothesis

In reinforcement learning with verifiable rewards, a grader assigns one outcome to an entire response. Every generated token then shares the response’s advantage. Averaging the token losses gives each response the same total length weight. Summing them lets longer responses contribute more. That choice matters when an incorrect derivation runs for hundreds of tokens or an instruction explicitly asks for a short answer.

Our hypothesis is that intermediate weighting can reduce the growth of incorrect responses while preserving accuracy. We test it separately on math and instruction following because the same response length can serve different purposes in the two tasks. The endpoint comparisons build on the analysis of response-length normalization in [Understanding R1-Zero-Like Training](https://github.com/sail-sg/understand-r1-zero).

## Method

For a response of length $L$, multiply its **summed** clipped token surrogate by $L^{\alpha-1}$. Given equal absolute advantages, the resulting relative weight is proportional to $L^\alpha$. For the 200-token and 20-token responses above, this gives a concrete prediction.

| Exponent | Token reduction | Longer response’s share of the pair’s length weight |
| --- | --- | ---: |
| $\alpha=0$ | Average tokens within each response | 50.0% |
| $\alpha=0.5$ | Divide the token sum by $\sqrt{L}$ | 76.0% |
| $\alpha=1$ | Sum tokens within each response | 90.9% |

These percentages follow from the weighting rule. Actual parameter updates also depend on token gradients, clipping, and regularization. The controlled sweep uses an additional fixed factor of $512^{-\alpha}$ to anchor scale at the completion cap. Separate runs examine the effect of removing this factor. [Objective implementation](src/alpha_grpo/objective.py)

## Experimental design

1. **Curate the prompts.** Target 500 GSM8K, 500 MATH level 1–3, and 500 instruction-constraint prompts. Cap formatted prompts at 256 tokens. Filter benchmark overlap using token-level longest common subsequence, then remove candidates the initial policy solves on all eight prescreen samples. [Training mixture](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)
2. **Train the policy.** Start from the same DPO checkpoint for every run. Sample six completions per prompt, cap each at 512 tokens, grade math symbolically and instructions with constraint functions, and center rewards within each group. A full update uses four groups. The target is 375 updates with asymmetric ratio clipping and a sampled KL penalty against the frozen starting policy. [Starting checkpoint](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO)
3. **Isolate the experimental choices.** Compare the three exponents with unrestricted sampling and training seeds 42 and 1337. At matched α, test reward standardization and filtering of groups whose rewards are all identical. Four additional proposal recipes use temperature 0.9 and top-p 0.95 with raw model log probabilities in the surrogate. [Full ablation matrix](configs/study.yaml)
4. **Evaluate across training.** Compare snapshots at updates 125, 250, and 375 with the SFT, starting DPO, and official RLVR1 checkpoints. Plan greedy decoding plus three sampled evaluation seeds for generative tasks. Report item-level uncertainty, paired differences against the starting policy, and variation across training seeds. [Evaluation protocol](docs/methods.md)

## Results

| Research question | Primary readout | Status |
| --- | --- | --- |
| Does α change mathematical accuracy? | GSM8K 8-shot chain-of-thought accuracy and MATH-500 pass@1 | Pending |
| Does the policy still satisfy explicit constraints? | IFEval prompt-level strict accuracy | Pending |
| Do incorrect responses grow longer? | Completion length by task, reward, and training step | Pending |
| Does filtering pay for discarded rollouts? | Mixed-reward groups per hour and total generated tokens | Pending |
| Do gains transfer or coincide with regressions? | MMLU 5-shot accuracy and TruthfulQA MC2 | Pending |

The planned accuracy-versus-length plot compares policies on both axes. Length-drift and within-group weight plots help explain cases where two policies reach similar accuracy with different generation costs. [Analysis code](src/alpha_grpo/analysis.py)

## Reproduce on Kaggle

The target environment is Linux with two NVIDIA T4 GPUs. GPU 0 holds the full-parameter PyTorch trainer. GPU 1 hosts two independent FP16 vLLM processes, each configured for 45% of its memory. The rollout process receives current policy weights before the next batch. The reference process scores the same completion tokens using the original DPO checkpoint. The trainer supports NCCL synchronization and a checkpoint-reload fallback. CPU handles grading and optional optimizer offload.

From a writable checkout in the Kaggle session with internet access and `uv` installed, begin with data preparation and a one-update preflight:

```bash
export A=/kaggle/working/alpha-grpo
bash scripts/kaggle.sh setup
bash scripts/kaggle.sh prepare-data --out "$A/prepared"
bash scripts/kaggle.sh prescreen --data "$A/prepared" --out "$A/curated"
bash scripts/kaggle.sh preflight --data "$A/curated" --out "$A/checkpoint-gate" --sync checkpoint
```

The runbook continues with the NCCL check, a 30-step smoke run split across a restart, training, evaluation, and figure generation. GPU memory fit, synchronization parity, recovery, and throughput still require measurement on the target hardware. [Kaggle runbook](docs/kaggle.md) · [Notebook launcher](notebooks/kaggle.ipynb)

## Attribution and license

The study uses Ai2’s OLMo 2 checkpoints, RLVR mixture, and instruction verifiers. Benchmark evaluation uses the EleutherAI harness. Repository code is MIT licensed. Models, datasets, and upstream verifier code retain their respective licenses. [Open Instruct](https://github.com/allenai/open-instruct) · [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) · [License](LICENSE)
