import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from .objective import trajectory_terms


class Trainer:
    def __init__(self, source, tokenizer, training, experiment):
        self.training = training
        self.experiment = experiment
        self.offload = training.optimizer == "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.float32, attn_implementation="sdpa",
        )
        self.parameters = list(self.model.parameters())
        self.master_parameters = None
        if self.offload:
            self.master_parameters = [
                torch.nn.Parameter(parameter.detach()) for parameter in self.parameters
            ]
            self.model.to(device="cuda", dtype=torch.float16)
            self.optimizer = torch.optim.AdamW(
                self.master_parameters, lr=training.lr, betas=(0.9, 0.999),
                weight_decay=0.01, foreach=False,
            )
        else:
            from bitsandbytes.optim import PagedAdamW8bit

            self.model.to(device="cuda")
            self.optimizer = PagedAdamW8bit(
                self.model.parameters(), lr=training.lr, betas=(0.9, 0.999), weight_decay=0.01,
            )
        self.parameters = list(self.model.parameters())
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.pad_id = tokenizer.pad_token_id
        self.scale = 128.0
        self.scale_growth = 0
        warmup = max(1, round(training.steps * 0.10))

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, training.steps - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)

    def logprobs(self, sequences):
        lengths = [len(prompt) + len(completion) for prompt, completion in sequences]
        maximum = max(lengths)
        ids = torch.tensor([
            prompt + completion + [self.pad_id] * (maximum - length)
            for (prompt, completion), length in zip(sequences, lengths, strict=True)
        ], device="cuda")
        mask = torch.arange(maximum, device="cuda")[None, :] < torch.tensor(
            lengths, device="cuda"
        )[:, None]
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            hidden = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

        def selected(features, targets):
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                logits = self.model.lm_head(features)
            return -F.cross_entropy(logits.float(), targets, reduction="none")

        result = []
        chunk_size = self.training.logprob_chunk
        for row, (prompt, completion) in enumerate(sequences):
            start = len(prompt)
            end = start + len(completion)
            chunks = []
            for position in range(start, end, chunk_size):
                stop = min(end, position + chunk_size)
                features = hidden[row, position - 1:stop - 1]
                targets = ids[row, position:stop]
                if torch.is_grad_enabled():
                    chunks.append(checkpoint(selected, features, targets, use_reentrant=False))
                else:
                    chunks.append(selected(features, targets))
            result.append(torch.cat(chunks))
        return result

    @torch.no_grad()
    def score(self, sequences):
        was_training = self.model.training
        self.model.eval()
        try:
            values = self.logprobs(sequences)
            if any(not torch.isfinite(item).all().item() for item in values):
                raise FloatingPointError("Nonfinite trainer scoring result")
            return [item.cpu().tolist() for item in values]
        finally:
            self.model.train(was_training)

    def _gradient_norm(self):
        norms = [
            torch.linalg.vector_norm(parameter.grad, dtype=torch.float32)
            for parameter in self.parameters if parameter.grad is not None
        ]
        return torch.linalg.vector_norm(torch.stack(norms)).item() / self.scale

    @torch.no_grad()
    def _optimizer_step(self, factor):
        if not self.offload:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            self.optimizer.step()
            return
        groups = self.optimizer.param_groups
        # Preserve AdamW state and scheduler groups while staging one tensor at a time.
        try:
            for gpu, master in zip(self.parameters, self.master_parameters, strict=True):
                if gpu.grad is None:
                    continue
                master.grad = gpu.grad.to(device="cpu", dtype=torch.float32).mul_(factor)
                self.optimizer.param_groups = [dict(groups[0], params=[master])]
                self.optimizer.step()
                master.grad = None
                gpu.copy_(master)
        finally:
            self.optimizer.param_groups = groups

    def update(self, batch):
        self.model.train()
        rows = [(group.prompt.token_ids, item) for group in batch.groups
                for item in group.trajectories]
        microbatch = self.training.microbatch
        for attempt in range(3):
            self.model.zero_grad(set_to_none=True)
            metric_policy, metric_kl, metric_drift = [], [], []
            for offset in range(0, len(rows), microbatch):
                chunk = rows[offset:offset + microbatch]
                values = self.logprobs([(prompt, item.completion.token_ids) for prompt, item in chunk])
                loss = 0.0
                for current, (_, item) in zip(values, chunk, strict=True):
                    old = torch.as_tensor(item.completion.logprobs, device=current.device)
                    reference = torch.as_tensor(item.reference_logprobs, device=current.device)
                    policy, kl = trajectory_terms(
                        current, old, reference, item.advantage, self.experiment, self.training,
                    )
                    loss = loss + policy + self.training.beta * kl
                    metric_policy.append(policy.detach())
                    metric_kl.append(kl.detach())
                    metric_drift.append((current.detach() - old).abs().mean())
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite loss before gradient scaling")
                (loss * (self.scale / len(rows))).backward()
            norm = self._gradient_norm()
            if math.isfinite(norm):
                self._optimizer_step(min(1.0, 1.0 / max(norm, 1e-12)) / self.scale)
                self.scheduler.step()
                self.model.zero_grad(set_to_none=True)
                self.scale_growth += 1
                if self.scale_growth >= 2000:
                    self.scale = min(self.scale * 2, 65536)
                    self.scale_growth = 0
                return dict(policy_loss=torch.stack(metric_policy).mean().item(),
                            kl=torch.stack(metric_kl).mean().item(), gradient_norm=norm,
                            old_policy_logprob_drift=torch.stack(metric_drift).mean().item(),
                            gradient_scale=self.scale, overflow_retries=attempt)
            self.scale = max(1.0, self.scale / 2)
            self.scale_growth = 0
        self.model.zero_grad(set_to_none=True)
        raise FloatingPointError("Three consecutive gradient overflows")
