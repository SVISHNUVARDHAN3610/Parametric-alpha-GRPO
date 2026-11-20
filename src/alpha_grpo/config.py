from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EngineConfig:
    policy_budget: float = 0.45
    reference_budget: float = 0.45
    max_sequences: int = 6
    sync: str = "auto"
    timeout_seconds: int = 180
    startup_timeout_seconds: int = 600
    nccl_policy_budget: float = 0.33

    def __post_init__(self):
        if min(self.policy_budget, self.reference_budget) <= 0:
            raise ValueError("Both inference budgets must be positive")
        if self.policy_budget + self.reference_budget > 0.90 + 1e-9:
            raise ValueError("Leave at least 10% of GPU 1 unallocated")
        if not 0 < self.nccl_policy_budget <= 1:
            raise ValueError("The NCCL policy budget must be in (0, 1]")
        if self.sync not in {"auto", "nccl", "checkpoint"}:
            raise ValueError("Unknown synchronization backend")
        if min(self.max_sequences, self.timeout_seconds, self.startup_timeout_seconds) < 1:
            raise ValueError("Inference sequence and timeout limits must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    max_prompt: int = 256
    max_completion: int = 512
    prompts_per_step: int = 4
    group_size: int = 6
    microbatch: int = 1
    logprob_chunk: int = 128
    steps: int = 375
    lr: float = 1.5e-6
    beta: float = 0.01
    clip_low: float = 0.20
    clip_high: float = 0.28
    filter_cap: int = 40
    optimizer: str = "gpu"

    def __post_init__(self):
        if min(self.max_prompt, self.max_completion, self.prompts_per_step,
               self.microbatch, self.logprob_chunk, self.steps) < 1:
            raise ValueError("Training sizes must be positive")
        if self.group_size < 2 or self.filter_cap < self.prompts_per_step:
            raise ValueError("Invalid collection limits")
        if self.max_prompt + self.max_completion + 1 > 4096:
            raise ValueError("Training context exceeds OLMo 2 context")
        if self.optimizer not in {"gpu", "cpu"}:
            raise ValueError("Invalid optimizer backend")
        if self.lr <= 0 or self.beta < 0 or not 0 <= self.clip_low < 1 or self.clip_high < 0:
            raise ValueError("Invalid loss or optimizer setting")


@dataclass(frozen=True)
class Experiment:
    alpha: float
    normalize_std: bool = False
    filter_constant: bool = False
    anchored: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    seeds: tuple[int, ...] = (42,)

    def __post_init__(self):
        if not 0 <= self.alpha <= 1 or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid alpha or sampling configuration")


@dataclass
class Study:
    model: str
    model_revision: str
    dataset: str
    dataset_revision: str
    engine: EngineConfig = field(default_factory=EngineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiments: dict[str, Experiment] = field(default_factory=dict)
    chat_template: str | None = None


def load_study(path: Path) -> Study:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["engine"] = EngineConfig(**raw.get("engine", {}))
    raw["training"] = TrainingConfig(**raw.get("training", {}))
    raw["experiments"] = {
        name: Experiment(**settings) for name, settings in raw["experiments"].items()
    }
    return Study(**raw)
