"""Spiking neurons used by SpikeViMFormer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _RoundWithStraightThroughGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lower: float, upper: float) -> Tensor:
        ctx.lower = lower
        ctx.upper = upper
        ctx.save_for_backward(x)
        return torch.round(torch.clamp(x, min=lower, max=upper))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[(x < ctx.lower) | (x > ctx.upper)] = 0
        return grad, None, None


class NormalizedIntegerLIF(nn.Module):
    """Normalized integer LIF neuron (NI-LIF).

    The neuron integrates the same static activation for ``time_steps`` and
    emits normalized integer spikes in ``[0, 1]``. A straight-through
    estimator is used for the rounding operation during backpropagation.
    """

    def __init__(
        self,
        max_spikes: int = 4,
        beta: float = 0.5,
        time_steps: int = 4,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        if max_spikes < 1:
            raise ValueError("max_spikes must be positive")
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        self.max_spikes = int(max_spikes)
        self.beta = float(beta)
        self.time_steps = int(time_steps)
        self.detach_reset = detach_reset

        # Updated on each forward pass; useful for energy analysis.
        self.last_total_events = 0
        self.last_fired_events = 0
        self.last_spike_count = 0.0

    def forward(self, x: Tensor) -> Tensor:
        membrane = torch.zeros_like(x)
        output = torch.zeros_like(x)
        fired = torch.zeros((), device=x.device, dtype=torch.float32)
        spike_sum = torch.zeros((), device=x.device, dtype=torch.float32)

        for _ in range(self.time_steps):
            membrane = self.beta * membrane + x
            spikes = _RoundWithStraightThroughGradient.apply(
                membrane, 0.0, float(self.max_spikes)
            )
            membrane = membrane - (spikes.detach() if self.detach_reset else spikes)
            output = output + spikes / self.max_spikes
            with torch.no_grad():
                fired += (spikes > 0).sum(dtype=torch.float32)
                spike_sum += spikes.float().sum()

        with torch.no_grad():
            self.last_total_events = x.numel() * self.time_steps
            self.last_fired_events = int(fired.item())
            self.last_spike_count = float(spike_sum.item())
        return output / self.time_steps


def reset_spike_statistics(module: nn.Module) -> None:
    """Reset counters on every NI-LIF module below ``module``."""

    for child in module.modules():
        if isinstance(child, NormalizedIntegerLIF):
            child.last_total_events = 0
            child.last_fired_events = 0
            child.last_spike_count = 0.0
