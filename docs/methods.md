# Research protocol

For each prompt, sample six outcomes. Center each binary reward by its group mean. The standard GRPO comparison also divides by the population standard deviation plus 0.0001. Weight the summed PPO surrogate over each response by `c_alpha * L ** (alpha - 1)` and average over trajectories. Here `L` includes sampled EOS and excludes prompt and padding tokens. Clip the ratio to `[0.80, 1.28]` inside the PPO minimum. Use `c_alpha = 512 ** -alpha` in the controlled study. Proposal and scale runs use `c_alpha = 1`.

Compute sampled KL separately, as the mean of response-level means of `expm1(d) - d`, with `d = log p_ref - log p_current`. Multiply this term by 0.01. One fresh rollout batch supplies each optimizer update. As a result, clipping should usually be inactive apart from discrepancies between execution backends. The weighted objective and the sampled KL estimator are evaluated under the actual recorded sampling policy.

| Configurations | Comparison | Training seeds |
| --- | --- | --- |
| `control-a0`, `control-a05`, `control-a1` | α spectrum with mean-centered rewards | 42, 1337 |
| `control-std` | Add standard-deviation normalization at α=0 | 42, 1337 |
| `control-filter` | Drop constant-reward groups at α=0.5 | 42, 1337 |
| `proposal-a` through `proposal-d` | Proposal weighting and sampling recipes | 42 |
| `scale-a05`, `scale-a1` | Remove the fixed normalization at matched sampling | 42 |

Controlled runs use temperature 1 and top-p 1. The proposal runs use temperature 0.9 and top-p 0.95 while retaining raw log probabilities in the objective. Those runs study the practical surrogate under filtered sampling. The length/advantage share `L ** alpha * abs(A)` describes relative weighting within a group. It does not measure parameter-gradient magnitude. The training results cannot establish the unbiasedness of the clipped, regularized estimator. [Length weighting and the theoretical tradeoff](https://arxiv.org/abs/2607.23364)

## Data and outputs

Resolve upstream revisions before curation and record them alongside content hashes. Strip Ai2's preformatted math demonstrations to recover the last question, join MATH levels exactly after Unicode and whitespace normalization, and apply the published OLMo chat template. Audit benchmark overlap with exact token LCS, using 13 tokens and 60% of the shorter input. Prescreen candidates using eight unrestricted samples from the frozen initial policy. Keep the first 500 eligible rows per task in seeded candidate order and retain the prescreen audit. For dynamic filtering, retain full groups with mixed rewards and count all candidates, including rejected groups and empty collection attempts.

Benchmark metrics are GSM8K 8-shot CoT strict extraction, MATH-500 pass@1 with symbolic grading, IFEval prompt-level strict accuracy, MMLU 5-shot accuracy, and TruthfulQA MC2. MATH-500 uses exact normalized text when a gold answer cannot be parsed symbolically. The harness uses a 512-token generation cap for consistency. Report evaluation response lengths from retokenized decoded outputs. The analysis exports paired item bootstrap intervals against a designated baseline, and raw training-seed scores and sample standard deviations separately. Runtime efficiency in the report includes rollouts, verification, optimization, and weight transfer. Session startup and durable artifact copying are outside that denominator.

## Changes from the proposal

| Proposal | Repository protocol |
| --- | --- |
| ZeRO-2 and Liger split training across two GPUs | Custom PyTorch training uses GPU 0. GPU 1 hosts both vLLM workers. Completion-token LM-head chunks limit logit memory. |
| Microbatch 2 and six accumulation passes per training GPU | A single training GPU uses microbatch 1 × 24 or microbatch 2 × 12 for a full batch. |
| FP16 trainable weights and an int8 reference | Stable FP32 trainable weights or CPU master weights, FP16 compute, and a frozen FP16 reference. |
| 8.73 GiB per GPU and fixed runtime estimates | Measured resource and throughput gates for the new device layout. |
| Vocabulary size 100,278 | Model configuration supplies the vocabulary size, currently 100,352 for the source checkpoint. |
| Proposal-endpoint comparisons alone | Controlled sampling, advantage, scale, and filtering comparisons supplement the four proposal recipes. |

The original proposal remains a separate source document. Its projected benchmark improvements remain hypotheses.

Model and dataset attribution goes to [Ai2 OLMo 2](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO), the [Ai2 mixture and its constituent datasets](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints), and the [MATH train-set mirror](https://huggingface.co/datasets/SevenAIGmbH/MATH-lighteval). Instruction grading uses a commit-pinned [Ai2 open-instruct verifier registry](https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py), retrieved during remote preparation. Benchmarks use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Math-Verify](https://github.com/huggingface/Math-Verify).
