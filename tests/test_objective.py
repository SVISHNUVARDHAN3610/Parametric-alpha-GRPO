import pytest
import torch

from alpha_grpo.config import Experiment, TrainingConfig
from alpha_grpo.objective import gradient_weight_shares, group_advantages, trajectory_terms


@pytest.mark.parametrize(("alpha", "expected"), [(0.0, -1.0), (0.5, -2.0), (1.0, -4.0)])
def test_alpha_scales_the_token_sum(alpha, expected):
    logprobs = torch.full((4,), -1.0, requires_grad=True)
    policy, kl = trajectory_terms(logprobs, logprobs.detach(), logprobs.detach(), 1.0,
                                  Experiment(alpha, anchored=False), TrainingConfig())
    assert policy.item() == pytest.approx(expected)
    assert kl.item() == pytest.approx(0)


@pytest.mark.parametrize(("advantage", "ratio", "expected", "gradient"), [
    (1, 1.5, -1.28, 0),
    (1, 0.7, -0.7, -0.7),
    (-1, 0.7, 0.80, 0),
    (-1, 1.5, 1.5, 1.5),
])
def test_asymmetric_clipping(advantage, ratio, expected, gradient):
    old = torch.tensor([-3.0])
    current = (old + torch.tensor(ratio).log()).requires_grad_()
    loss, _ = trajectory_terms(current, old, old, advantage, Experiment(0), TrainingConfig())
    loss.backward()
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert current.grad.item() == pytest.approx(gradient, abs=1e-6)


def test_group_normalization_and_length_proxy():
    assert group_advantages([0, 1], False) == [-0.5, 0.5]
    assert group_advantages([1, 1], True) == [0, 0]
    assert group_advantages([0, 1], True) == pytest.approx([-0.9998, 0.9998])
    assert gradient_weight_shares([10, 100], [0.5, -0.5], 1) == pytest.approx([1 / 11, 10 / 11])
    assert gradient_weight_shares([10, 100], [0, 0], 1) == [None, None]


def test_sampled_kl_uses_reference_over_current():
    old = current = torch.tensor([-3.0])
    reference = current + 1.0
    _, kl = trajectory_terms(current, old, reference, 0.0, Experiment(0), TrainingConfig())
    assert kl.item() == pytest.approx(0.7182818)


def test_trajectory_accumulation_keeps_a_global_denominator():
    def grad(split):
        parameter = torch.tensor(0.0, requires_grad=True)
        for chunk in split:
            terms = []
            for length, advantage in chunk:
                old = torch.full((length,), -3.0)
                loss, _ = trajectory_terms(
                    old + parameter, old, old, advantage, Experiment(0.5), TrainingConfig(),
                )
                terms.append(loss)
            (sum(terms) / 3).backward()
        return parameter.grad

    batch = [(1, 1), (3, -1), (5, 1)]
    torch.testing.assert_close(grad([batch]), grad([batch[:2], batch[2:]]))
