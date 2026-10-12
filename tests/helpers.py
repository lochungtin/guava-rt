"""Synthetic masks shared by the tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def shape_for(ndim: int) -> tuple[int, ...]:
    return (48, 48) if ndim == 2 else (32, 32, 32)


def coords(shape: Sequence[int]) -> Tensor:
    """Voxel coordinates ``(ndim, *shape)`` as float64."""
    axes = [torch.arange(n, dtype=torch.float64) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def ball(shape: Sequence[int], centre: Sequence[float], radius: float) -> Tensor:
    """Boolean disc/sphere."""
    c = torch.tensor(centre, dtype=torch.float64).view(-1, *[1] * len(shape))
    return (coords(shape) - c).norm(dim=0) <= radius


def box(shape: Sequence[int], lo: Sequence[int], hi: Sequence[int]) -> Tensor:
    """Boolean axis-aligned box covering ``[lo, hi)`` on every axis."""
    out = torch.zeros(tuple(shape), dtype=torch.bool)
    out[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))] = True
    return out


def shifted(mask: Tensor, shift: Sequence[int]) -> Tensor:
    """``mask`` translated by an integer ``shift`` (content must stay inside the grid)."""
    return torch.roll(mask, shifts=tuple(shift), dims=tuple(range(mask.ndim)))


def organs(ndim: int) -> list[Tensor]:
    """``[anchor, target, oar_near, oar_far]`` on :func:`shape_for`'s grid, all disjoint."""
    shape = shape_for(ndim)
    mid = [n // 2 for n in shape]

    def at(*offset: int) -> list[int]:
        return [m + o for m, o in zip(mid, (*offset, 0, 0)[:ndim], strict=True)]

    return [
        ball(shape, at(-11), 3),  # anchor ("spine")
        ball(shape, at(0), 5),  # target
        ball(shape, at(0, 10), 3),  # OAR beside the target
        ball(shape, at(10, -8), 3),  # OAR further away
    ]


LABELS = ["spine", "target", "near", "far"]
