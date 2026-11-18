from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRecord:
    source_id: str
    task: str
    messages: list[dict[str, str]]
    token_ids: list[int]
    verifier: dict


@dataclass(frozen=True)
class Completion:
    token_ids: list[int]
    logprobs: list[float] | None
    finish_reason: str


@dataclass(frozen=True)
class Trajectory:
    completion: Completion
    reward: float
    advantage: float
    reference_logprobs: list[float]


@dataclass(frozen=True)
class Group:
    prompt: PromptRecord
    trajectories: list[Trajectory]


@dataclass(frozen=True)
class RolloutBatch:
    policy_version: int
    groups: list[Group]

    @property
    def size(self) -> int:
        return sum(len(group.trajectories) for group in self.groups)


@dataclass
class RunState:
    version: int = 0
    cursor: int = 0
    attempted_groups: int = 0
    generated_tokens: int = 0
    empty_attempts: int = 0
    collection_id: int = 0
    pending_collection_seconds: float = 0.0
