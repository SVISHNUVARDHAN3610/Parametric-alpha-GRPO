from types import SimpleNamespace

import pytest

from alpha_grpo.analysis import paired_interval
from alpha_grpo.data import bit_profile, final_question, is_contaminated, lcs_length, model_snapshot, shuffled
from alpha_grpo.inference_worker import selected_scores
from alpha_grpo.records import PromptRecord
from alpha_grpo.verifiers import constraint_specs, math_reward, reward


def test_model_snapshot_includes_bin_only_weights(monkeypatch, tmp_path):
    from fnmatch import fnmatchcase

    import huggingface_hub

    def download(model, *, revision, allow_patterns):
        assert revision == "pinned"
        for name in ("config.json", "pytorch_model.bin"):
            if any(fnmatchcase(name, pattern) for pattern in allow_patterns):
                (tmp_path / name).touch()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda *_args, **_kwargs: SimpleNamespace(sha="pinned"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert model_snapshot("allenai/OLMo-2-0425-1B-DPO", "main") == ("pinned", str(tmp_path))
    assert (tmp_path / "pytorch_model.bin").is_file()


def test_lcs_can_match_noncontiguous_tokens():
    common = list(range(13))
    interleaved = [token for item in common for token in (item, -1)]
    assert lcs_length(interleaved, bit_profile(common)) == 13
    assert is_contaminated(interleaved, [(len(common), bit_profile(common))])
    assert not is_contaminated(interleaved[:24], [(len(common), bit_profile(common))])


def test_lcs_requires_both_thresholds():
    left = list(range(25))
    right = list(range(14)) + list(range(25, 36))
    assert lcs_length(left, bit_profile(right)) == 14
    assert not is_contaminated(left, [(len(right), bit_profile(right))])
    right[14] = 14
    assert is_contaminated(left, [(len(right), bit_profile(right))])


def test_last_question_and_stable_candidate_pool():
    assert final_question("Question: example\nAnswer: solution\n\nQuestion: target\nAnswer:") == "target"
    assert shuffled(range(10), 42) == shuffled(range(10), 42)
    assert shuffled(range(10), 42) != shuffled(range(10), 1337)


def test_instruction_requires_every_constraint():
    specs = constraint_specs(dict(
        ground_truth='[{"func_name":"starts","prefix":"A"},{"func_name":"ends","suffix":"Z"}]'
    ))
    prompt = PromptRecord("fixture", "ifeval", [], [1], {"constraints": specs})
    registry = dict(starts=lambda text, prefix: text.startswith(prefix),
                    ends=lambda text, suffix: text.endswith(suffix))
    assert reward(prompt, "AZ", registry) == 1
    assert reward(prompt, "AB", registry) == 0
    with pytest.raises(ValueError):
        constraint_specs(dict(ground_truth="invalid"))


def test_boxed_and_numeric_math_answers():
    assert math_reward(r"\frac{1}{2}", r"The result is \boxed{0.5}.") == 1
    assert math_reward("12", "The answer is 12.") == 1
    assert math_reward("12", r"12 was an intermediate result. \boxed{17}") == 0


def test_selected_logprobs_keep_sampled_id_and_eos():
    assert selected_scores([7, 99], [{7: SimpleNamespace(logprob=-2.0)},
                                      {99: SimpleNamespace(logprob=-3.0)}]) == [-2.0, -3.0]
    assert selected_scores([7, 99], [{7: SimpleNamespace(logprob=-2.0)}]) is None
    assert selected_scores([7], [{8: SimpleNamespace(logprob=-2.0)}]) is None


def test_bootstrap_pairs_the_same_items():
    assert paired_interval({1: 1, 2: 0}, {1: 0, 2: 0}) == [0, 1]
    with pytest.raises(ValueError):
        paired_interval({1: 1}, {2: 1})
