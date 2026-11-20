import os
import sys
from types import SimpleNamespace
from uuid import UUID

import pytest

from alpha_grpo.campaign import _truncate_uncommitted, inference_launcher, latest_checkpoint
from alpha_grpo.checkpoint import atomic_directory, export_model
from alpha_grpo.config import EngineConfig, TrainingConfig
from alpha_grpo.inference import InferenceError
from alpha_grpo.inference_worker import Engine, checked_device_uuid, configure_cuda_linker
from alpha_grpo.sync import Policy
from alpha_grpo.weights import model_fingerprint


def test_device_identity_accepts_nvml_prefix_and_rejects_wrong_gpu():
    actual = UUID("9f53b41a-67cc-4a77-9985-cc75b129ce48")
    requested = f"GPU-{str(actual).upper()}"
    assert checked_device_uuid(list(actual.bytes), requested) == requested
    with pytest.raises(RuntimeError, match="CUDA device UUID mismatch"):
        checked_device_uuid(UUID(int=1).bytes, requested)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux inference worker")
def test_cuda_linker_uses_loaded_driver_and_preserves_search_path(tmp_path, monkeypatch):
    driver = tmp_path / "driver libraries" / "libcuda.so.580.95.05"
    driver.parent.mkdir()
    driver.touch()
    runtime = driver.with_name("libcudart.so.12")
    runtime.touch()
    link_directory = tmp_path / "worker-links"
    link_directory.mkdir()
    maps = tmp_path / "maps"
    maps.write_text(
        f"1000-2000 r-xp 00000000 00:01 1 {runtime}\n"
        f"2000-3000 r-xp 00000000 00:01 2 {driver}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBRARY_PATH", "/opt/compiler/lib")

    configure_cuda_linker(link_directory, maps)

    assert (link_directory / "libcuda.so").readlink() == driver
    assert os.environ["LIBRARY_PATH"].split(os.pathsep) == ["/opt/compiler/lib", str(link_directory)]


def test_rollout_preserves_sampled_stop_token_and_its_logprob(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    item = SimpleNamespace(token_ids=[23], logprobs=[{23: SimpleNamespace(logprob=-0.5)}],
                           finish_reason="stop")
    engine = Engine.__new__(Engine)
    engine.concurrency = 1
    engine.llm = SimpleNamespace(generate=lambda *args, **kwargs: [SimpleNamespace(outputs=[item])])

    assert engine.generate([11], count=1, max_tokens=1, temperature=1, top_p=1, seed=42) == [
        dict(token_ids=[23], logprobs=[-0.5], finish_reason="stop"),
    ]


def test_teacher_forced_scoring_reports_missing_logprobs(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    output = SimpleNamespace(prompt_token_ids=[11, 23], prompt_logprobs=None)
    engine = Engine.__new__(Engine)
    engine.concurrency = 1
    engine.llm = SimpleNamespace(generate=lambda *args, **kwargs: [output])

    with pytest.raises(ValueError, match="Missing teacher-forced log probabilities"):
        engine.score([([11], [23])])


def test_failed_atomic_checkpoint_leaves_no_committed_directory(tmp_path):
    destination = tmp_path / "step-000001"
    with pytest.raises(RuntimeError):
        with atomic_directory(destination) as partial:
            (partial / "training.pt").write_text("partial")
            raise RuntimeError("interrupted")
    assert latest_checkpoint(tmp_path) is None
    with atomic_directory(destination) as partial:
        (partial / "complete").touch()
    incomplete = tmp_path / "step-000002"
    incomplete.mkdir()
    assert latest_checkpoint(tmp_path) == destination


def test_resume_trims_steps_after_committed_checkpoint(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"version": 1}\n{"version": 2}\n{"vers')
    _truncate_uncommitted(path, "version", 1)
    assert path.read_text() == '{"version": 1}\n'


def test_reference_cannot_receive_policy_weights():
    engine = Engine.__new__(Engine)
    engine.version = 0
    engine.role = "reference"
    engine.transfer = False
    engine.updating = False
    with pytest.raises(RuntimeError, match="Weight transfer is unavailable"):
        engine.execute("start", version=1)
    with pytest.raises(RuntimeError, match="stale"):
        engine.execute("score", version=1, sequences=[])


def test_tied_embeddings_are_fingerprinted_and_exported(tmp_path):
    import torch
    from torch import nn

    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    embedding = model.model.embed_tokens.weight
    model.lm_head.weight = nn.Parameter(embedding.detach().clone())
    untied_fingerprint = model_fingerprint(model)

    model.lm_head.weight = embedding
    assert "lm_head.weight" not in dict(model.named_parameters())
    assert model_fingerprint(model) == untied_fingerprint
    with torch.no_grad():
        embedding.fill_(32.0)
    assert model_fingerprint(model) != untied_fingerprint

    def save_pretrained(_path, *, state_dict, **_options):
        exported = state_dict["model.embed_tokens.weight"]
        assert state_dict["lm_head.weight"] is exported
        assert exported.dtype == torch.float16

    model.save_pretrained = save_pretrained
    tokenizer = SimpleNamespace(save_pretrained=lambda _path: None)
    destination = tmp_path / "weights"
    export_model(model, tokenizer, destination)
    assert (destination / "complete").is_file()


@pytest.mark.parametrize("policy_budget, expected", [(0.45, 0.33), (0.30, 0.30)])
def test_nccl_policy_budget_reserves_transfer_headroom(tmp_path, monkeypatch, policy_budget, expected):
    monkeypatch.setattr("alpha_grpo.campaign.InferenceClient", lambda **options: options)
    study = SimpleNamespace(engine=EngineConfig(policy_budget=policy_budget), training=TrainingConfig())
    start = inference_launcher(study, "GPU-test", "source", tmp_path)

    assert start("policy", "source", 0, True)["budget"] == expected
    assert start("policy", "source", 0, False)["budget"] == policy_budget
    assert start("reference", "source", 0, False)["budget"] == study.engine.reference_budget


def test_failed_native_transfer_uses_checkpoint_before_acknowledging(tmp_path, monkeypatch):
    policy = Policy(start=None, source=None, tokenizer=None, trainer=None, mode="auto",
                    timeout=10, root=tmp_path)
    policy.client = SimpleNamespace()
    policy.acknowledged_version = 1
    calls = []

    def fail(version):
        assert policy.acknowledged_version is None
        calls.append(("nccl", version))
        raise InferenceError("partial transfer")

    def checkpoint(version):
        assert policy.acknowledged_version is None
        calls.append(("checkpoint", version))

    def verify(version):
        assert policy.acknowledged_version is None
        calls.append(("verify", version))

    monkeypatch.setattr(policy, "_nccl", fail)
    monkeypatch.setattr(policy, "_checkpoint", checkpoint)
    monkeypatch.setattr(policy, "_verify", verify)
    assert policy.synchronize(2) == "checkpoint"
    assert policy.acknowledged_version == 2
    assert calls == [("nccl", 2), ("checkpoint", 2), ("verify", 2)]
