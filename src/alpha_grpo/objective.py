import torch


def group_advantages(rewards: list[float], normalize_std: bool) -> list[float]:
    mean = sum(rewards) / len(rewards)
    centered = [reward - mean for reward in rewards]
    if not normalize_std:
        return centered
    std = (sum(value ** 2 for value in centered) / len(centered)) ** 0.5
    return [value / (std + 1e-4) for value in centered]


def gradient_weight_shares(lengths: list[int], advantages: list[float], alpha: float):
    weights = [length ** alpha * abs(adv) for length, adv in zip(lengths, advantages, strict=True)]
    total = sum(weights)
    return [weight / total for weight in weights] if total else [None] * len(weights)


def trajectory_terms(current, old, reference, advantage, experiment, training):
    """Return one trajectory's policy term and separately normalized sampled KL."""
    if current.ndim != 1 or current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("Expected aligned vectors of completion-token log probabilities")
    length = current.numel()
    if not length:
        raise ValueError("Empty completion in objective")
    current = current.float()
    ratio = (current - old.float()).exp()
    clamped = ratio.clamp(1 - training.clip_low, 1 + training.clip_high)
    surrogate = torch.minimum(ratio * advantage, clamped * advantage)
    scale = training.max_completion ** -experiment.alpha if experiment.anchored else 1.0
    policy = -scale * length ** (experiment.alpha - 1) * surrogate.sum()
    delta = reference.float() - current
    kl = (torch.expm1(delta) - delta).mean()
    return policy, kl
