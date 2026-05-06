from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianMetricState:
    """Information-form state for independent arms with vector-valued metrics."""

    n_eff: Tensor
    sum_g: Tensor
    active: Tensor
    t: int

    def __post_init__(self) -> None:
        if self.n_eff.ndim != 1:
            raise ValueError("n_eff must have shape (n_arms,)")
        if self.sum_g.ndim != 2:
            raise ValueError("sum_g must have shape (n_arms, n_metrics)")
        if self.sum_g.shape[0] != self.n_eff.shape[0]:
            raise ValueError("n_eff and sum_g must agree on n_arms")
        if self.active.shape != self.n_eff.shape:
            raise ValueError("active must have shape (n_arms,)")
        if not self.n_eff.dtype.is_floating_point:
            object.__setattr__(self, "n_eff", self.n_eff.to(torch.float64))
        object.__setattr__(self, "sum_g", self.sum_g.to(dtype=self.n_eff.dtype, device=self.n_eff.device))
        object.__setattr__(
            self,
            "active",
            torch.clamp(self.active.to(dtype=self.n_eff.dtype, device=self.n_eff.device), min=0.0, max=1.0),
        )

    @property
    def n_arms(self) -> int:
        return int(self.n_eff.shape[0])

    @property
    def n_metrics(self) -> int:
        return int(self.sum_g.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.n_eff.dtype

    @property
    def device(self) -> torch.device:
        return self.n_eff.device

    def clone(self) -> "GaussianMetricState":
        return GaussianMetricState(
            n_eff=self.n_eff.clone(),
            sum_g=self.sum_g.clone(),
            active=self.active.clone(),
            t=self.t,
        )

    def replace(self, **kwargs) -> "GaussianMetricState":
        return replace(self, **kwargs)
