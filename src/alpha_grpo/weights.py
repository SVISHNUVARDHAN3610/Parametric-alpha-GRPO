import hashlib


def model_fingerprint(model):
    import torch

    names = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    digest = hashlib.sha256()
    for name in names:
        # A tied output embedding may be omitted by named_parameters().
        parameter = model.get_parameter(name).detach()
        # Sample across each tensor without allocating a whole FP16 copy.
        stride = max(1, parameter.numel() // 4096)
        sample = parameter.flatten()[::stride].to(device="cpu", dtype=torch.float16)
        digest.update(name.encode())
        digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


class FingerprintWorker:
    def alpha_grpo_fingerprint(self):
        return model_fingerprint(self.model_runner.model)
