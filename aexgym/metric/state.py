from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianMetricState:
    """Posterior state for independent arms with vector-valued metrics."""

    mean: Tensor
    cov: Tensor
    active: Tensor
    t: int
    stopped: bool = False
    stop_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mean.ndim != 2:
            raise ValueError("mean must have shape (n_arms, n_metrics)")
        if self.cov.ndim != 3:
            raise ValueError("cov must have shape (n_arms, n_metrics, n_metrics)")
        if self.cov.shape[0] != self.mean.shape[0]:
            raise ValueError("cov and mean must agree on n_arms")
        if self.cov.shape[1:] != (self.mean.shape[1], self.mean.shape[1]):
            raise ValueError("cov metric dimensions must match mean")
        if self.active.shape != (self.mean.shape[0],):
            raise ValueError("active must have shape (n_arms,)")
        object.__setattr__(self, "active", self.active.to(dtype=torch.bool, device=self.mean.device))

    @property
    def n_arms(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_metrics(self) -> int:
        return int(self.mean.shape[1])

    @property
    def active_count(self) -> int:
        return int(self.active.sum().item())

    def active_indices(self) -> Tensor:
        return torch.nonzero(self.active, as_tuple=False).flatten()

    def clone(self) -> "GaussianMetricState":
        return GaussianMetricState(
            mean=self.mean.clone(),
            cov=self.cov.clone(),
            active=self.active.clone(),
            t=self.t,
            stopped=self.stopped,
            stop_reason=self.stop_reason,
        )

    def replace(self, **kwargs) -> "GaussianMetricState":
        return replace(self, **kwargs)

    def selected_arm(self, target_idx: int) -> int:
        active_idx = self.active_indices()
        if active_idx.numel() == 0:
            raise ValueError("cannot select an arm from an empty active set")
        target_values = self.mean[active_idx, target_idx]
        return int(active_idx[torch.argmax(target_values)].item())

    def terminal_value(self, target_idx: int) -> Tensor:
        active_idx = self.active_indices()
        if active_idx.numel() == 0:
            raise ValueError("cannot evaluate terminal value for an empty active set")
        return torch.max(self.mean[active_idx, target_idx])


def active_indices_to_list(active: Tensor) -> list[int]:
    return [int(i) for i in torch.nonzero(active, as_tuple=False).flatten().tolist()]
