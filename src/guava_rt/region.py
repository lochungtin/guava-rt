"""A set of masks from one study, measured relative to a target structure."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor

from guava_rt._deprecation import alias, alias_property, renamed_kwargs
from guava_rt.mask import ArrayLike, Device, Mask

T = TypeVar("T")


class Region:
    """Ordered collection of masks (e.g. the OARs of one patient position).

    ``target`` selects the structure receiving radiation; ``anchor`` is the rigid reference
    point used for displacement vectors, given as a mask (index or label, using its centre of
    mass) or as explicit coordinates. Per-structure results are dicts keyed by label when
    ``labels`` is given, otherwise lists in mask order.
    """

    def __init__(
        self,
        *masks: Mask | ArrayLike,
        target: int | str,
        anchor: int | str | ArrayLike,
        labels: list[str] | None = None,
        dev: Device = "cpu",
    ) -> None:
        self.masks = [m if isinstance(m, Mask) else Mask(m, dev) for m in masks]
        self.use_labels = labels is not None
        self.labels = labels
        self.dev = dev

        self.target = target
        self.target_idx = self._index(target)
        self.target_mask = self.masks[self.target_idx]

        if isinstance(anchor, np.ndarray):
            self.anchor = torch.from_numpy(anchor).float().to(self.dev)
        elif isinstance(anchor, Tensor):
            self.anchor = anchor.float().to(self.dev)
        else:
            self.anchor = self.masks[self._index(anchor)].center_of_mass

        self.target_dmap = self.target_mask.dmap()

    def __getitem__(self, key: int | str) -> Mask | None:
        """Mask by index or label; ``None`` for an unknown label."""
        if isinstance(key, str) and (self.labels is None or key not in self.labels):
            return None
        return self.masks[self._index(key)]

    def __len__(self) -> int:
        return len(self.masks)

    def _index(self, key: int | str) -> int:
        if isinstance(key, str):
            if self.labels is None:
                raise ValueError(f"cannot look up {key!r}: the region has no labels")
            return self.labels.index(key)
        return key

    def _keyed(self, values: list[T], exclude_target: bool = False) -> dict[str, T] | list[T]:
        """Key per-structure ``values`` by label (if labelled); target omitted if excluded."""
        if self.labels is None:
            return values
        labels = [
            label
            for i, label in enumerate(self.labels)
            if not (exclude_target and i == self.target_idx)
        ]
        return dict(zip(labels, values, strict=True))

    def _non_target(self) -> list[Mask]:
        return [m for i, m in enumerate(self.masks) if i != self.target_idx]

    # ------------------------------------------------------------------
    def _get_displacement_vector(self, mask: Mask, use_anchor: bool) -> Tensor:
        ref = self.anchor if use_anchor else self.target_mask.center_of_mass
        return mask.center_of_mass - ref

    @renamed_kwargs(useAnchor="use_anchor", useLabels="use_labels")
    def get_displacement_vectors(
        self, use_anchor: bool = False, use_labels: bool = True
    ) -> dict[str, Tensor] | list[Tensor]:
        """Centre of mass of every mask relative to the anchor (or the target's centre of mass)."""
        out = [self._get_displacement_vector(m, use_anchor) for m in self.masks]
        return self._keyed(out) if use_labels else out

    # ------------------------------------------------------------------
    def _masked_dmap(self, mask: Mask, mode: str, chunk_size: int | None) -> tuple[Tensor, ...]:
        """Target dmap restricted to the ``mode`` part of ``mask``: ``(values, masked, part)``."""
        if mode == "surface":
            part = mask.surface()
        elif mode == "rcvs":
            part = mask.get_rcvs(self.target_mask, chunk_size=chunk_size).mask
        else:
            part = mask.mask
        return *self.target_mask.mask_dmap(part), part

    # ------------------------------------------------------------------
    def get_separation_distances(
        self,
        mode: str = "volume",
        distances_only: bool = True,
        chunk_size: int | None = 1,
    ) -> dict[str, Any] | list[Any]:
        """Distances from the target to every voxel of each non-target mask.

        ``mode`` picks the voxels: ``"volume"`` (whole mask), ``"surface"`` or ``"rcvs"``.
        With ``distances_only=False`` each entry is ``(distances, masked dmap, voxel mask)``.
        """
        out: list[Any] = []
        for m in self._non_target():
            masked = self._masked_dmap(m, mode, chunk_size)
            out.append(masked[0] if distances_only else masked)
        return self._keyed(out, exclude_target=True)

    # ------------------------------------------------------------------
    def get_thresholded_overlap_percentages(
        self,
        mode: str = "volume",
        percentages_only: bool = True,
        chunk_size: int | None = 1,
    ) -> dict[str, Any] | list[Any]:
        """Fraction of each non-target mask within distance ``r`` of the target, per integer ``r``.

        Each entry covers ``r`` from the mask's minimum to maximum separation distance. With
        ``percentages_only=False`` it is ``[percs, full, bins, distances, masked dmap, voxel
        mask]``, where ``full`` / ``bins`` hold the fraction / voxel count for every ``r`` from 0
        to the target dmap's maximum.
        """
        max_dist = int(self.target_dmap.max().ceil().item())
        out: list[Any] = []

        for m in self._non_target():
            d_vals, masked, part = self._masked_dmap(m, mode, chunk_size)
            d_min = int(d_vals.min().ceil().item())
            d_max = int(d_vals.max().ceil().item())
            n_mask = part.float().sum()

            percs = torch.zeros(max(d_max - d_min + 1, 1), device=self.dev)
            full = torch.zeros(max_dist + 1, device=self.dev)
            bins = torch.zeros(max_dist + 1, device=self.dev)

            for j, thresh in enumerate(range(d_min, d_max + 1)):
                s = (d_vals <= thresh).float().sum()
                p = s / n_mask
                percs[j] = p
                if thresh <= max_dist:
                    full[thresh] = p
                    bins[thresh] = s

            if d_max <= max_dist:
                full[d_max:] = 1.0
                bins[d_max:] = n_mask

            out.append(percs if percentages_only else [percs, full, bins, d_vals, masked, part])

        return self._keyed(out, exclude_target=True)

    # ------------------------------------------------------------------
    # Deprecated camelCase API (guava-rt <= 0.5)
    useLabels = alias_property("use_labels", "useLabels")
    getDisplacementVectors = alias(get_displacement_vectors, "getDisplacementVectors")
    getSeparationDistances = alias(get_separation_distances, "getSeparationDistances")
    getThresholdedOverlapPercentages = alias(
        get_thresholded_overlap_percentages, "getThresholdedOverlapPercentages"
    )
