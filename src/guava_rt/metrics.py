"""Pairwise geometric comparison of two regions (e.g. upright vs. supine)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from guava_rt._deprecation import alias, alias_property
from guava_rt.mask import ArrayLike, Device, Mask
from guava_rt.region import Region
from guava_rt.utils import series_analysis

RegionLike = Region | Sequence[Mask | ArrayLike] | ArrayLike

# Summary statistics for get_separation_distance_diff; any other mode gives series_analysis
_STATISTICS: dict[str, Callable[[Tensor], Tensor]] = {
    "max": lambda x: x.max(),
    "min": lambda x: x.min(),
    "p5": lambda x: torch.quantile(x, 0.05),
    "p95": lambda x: torch.quantile(x, 0.95),
    "mean": lambda x: x.mean(),
    "median": lambda x: x.median(),
}


def _statistic(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """``f`` on a float tensor, NaN for an empty one (e.g. an OAR with no visible surface)."""

    def wrapped(x: Tensor) -> Tensor:
        x = x.float()
        return f(x) if x.numel() else torch.tensor(float("nan"), device=x.device)

    return wrapped


class Metrics:
    """Compares region ``A`` against region ``B`` structure by structure.

    ``A`` and ``B`` are :class:`Region` objects or sequences of masks, which are wrapped in a
    region using ``target`` / ``anchor`` (or the per-side ``target_A`` / ``anchor_B`` etc.)
    and ``labels``. Structures are paired by position, so both sides must list them in the same
    order. Every difference is ``A - B``; results are keyed by ``A``'s labels when it has them.
    """

    def __init__(
        self,
        A: RegionLike,
        B: RegionLike,
        target: int | str | None = None,
        target_A: int | str | None = None,
        target_B: int | str | None = None,
        anchor: int | str | ArrayLike | None = None,
        anchor_A: int | str | ArrayLike | None = None,
        anchor_B: int | str | ArrayLike | None = None,
        labels: list[str] | None = None,
        dev: Device = "cpu",
    ) -> None:
        self.A = _as_region(A, target, target_A, anchor, anchor_A, labels, dev)
        self.B = _as_region(B, target, target_B, anchor, anchor_B, labels, dev)
        self.use_labels = self.A.use_labels
        self.labels = self.A.labels
        self.dev = dev

        self._bsd: list[Tensor] | None = None

    def _pairs(self) -> zip[tuple[Mask, Mask]]:
        return zip(self.A.masks, self.B.masks, strict=True)

    # ------------------------------------------------------------------
    def get_vol_diff(self) -> dict[str, Any] | list[Any]:
        """Per structure ``(V_A - V_B, relative to V_B, V_A, V_B)``."""
        return self.A._keyed([a.get_vol_diff(b) for a, b in self._pairs()])

    def get_sa_diff(self) -> dict[str, Any] | list[Any]:
        """Per structure ``(S_A - S_B, relative to S_B, S_A, S_B)`` for surface areas."""
        return self.A._keyed([a.get_sa_diff(b) for a, b in self._pairs()])

    # ------------------------------------------------------------------
    def get_roi_displacement_diff(self) -> dict[str, Tensor] | list[Tensor]:
        """Displacement of each structure between A and B after aligning the anchors."""
        vec_a = torch.stack(_as_list(self.A.get_displacement_vectors(use_anchor=True)))
        vec_b = torch.stack(_as_list(self.B.get_displacement_vectors(use_anchor=True)))
        return self.A._keyed(list(vec_a - vec_b))

    def align_a_to_b(self) -> tuple[Region, Region]:
        """Translate every mask in A so that A's anchor lands on B's; returns ``(A', B)``."""
        shift_vec = self.B.anchor - self.A.anchor
        return (
            Region(
                *[m.align_to(shift_vec=shift_vec) for m in self.A.masks],
                target=self.A.target,
                anchor=self.A.anchor + shift_vec,
                labels=self.A.labels,
                dev=self.A.dev,
            ),
            self.B,
        )

    # ------------------------------------------------------------------
    def get_bsd_diff(self, mode: str = "all") -> dict[str, Tensor] | list[Tensor]:
        """Bidirectional surface distance between each A/B pair, centre-of-mass aligned.

        ``mode``: ``"asd"``, ``"hd"``, ``"hd95"``, or ``"all"`` for the full
        :func:`series_analysis` vector.
        """
        if self._bsd is None:
            self._bsd = [a.get_bsd(b)[0] for a, b in self._pairs()]

        stats = [series_analysis(b) for b in self._bsd]
        index = {"asd": 3, "hd": -1, "hd95": -2}.get(mode.lower())
        return self.A._keyed(stats if index is None else [s[index] for s in stats])

    # ------------------------------------------------------------------
    def get_separation_distance_diff(
        self,
        mask: str = "volume",
        mode: str = "all",
        distances_only: bool = True,
        chunk_size: int | None = 1,
    ) -> dict[str, Any] | list[Any]:
        """Target-to-structure separation distances, summarised, for every non-target structure.

        ``mask`` picks the voxels (``"volume"``, ``"surface"``, ``"rcvs"``); ``mode`` the
        statistic (``"min"``, ``"max"``, ``"mean"``, ``"median"``, ``"p5"``, ``"p95"``, or
        ``"all"``). Each entry is ``[f(A) - f(B), f(A), f(B)]``, followed by both sides' raw
        :meth:`Region.get_separation_distances` outputs when ``distances_only=False``.
        """
        dist_a = _as_list(self.A.get_separation_distances(mask, distances_only, chunk_size))
        dist_b = _as_list(self.B.get_separation_distances(mask, distances_only, chunk_size))
        f = _statistic(_STATISTICS.get(mode.lower(), series_analysis))

        out: list[Any] = []
        for a, b in zip(dist_a, dist_b, strict=True):
            d_a, d_b = (a, b) if distances_only else (a[0], b[0])
            extra = [] if distances_only else [*a, *b]
            out.append([f(d_a) - f(d_b), f(d_a), f(d_b), *extra])
        return self.A._keyed(out, exclude_target=True)

    # ------------------------------------------------------------------
    def get_percentage_overlap_diff(
        self,
        mode: str = "volume",
        percentages_only: bool = True,
        chunk_size: int | None = 1,
    ) -> dict[str, Any] | list[Any]:
        """Summary of each structure's thresholded overlap curve, A vs. B.

        ``mode`` picks the voxels (``"volume"``, ``"surface"``, ``"rcvs"``). Each entry is
        ``[s(A) - s(B), s(A), s(B)]`` with ``s`` = :func:`series_analysis`, followed by both
        sides' raw :meth:`Region.get_thresholded_overlap_percentages` outputs when
        ``percentages_only=False``.
        """
        perc_a = _as_list(
            self.A.get_thresholded_overlap_percentages(mode, percentages_only, chunk_size)
        )
        perc_b = _as_list(
            self.B.get_thresholded_overlap_percentages(mode, percentages_only, chunk_size)
        )

        out: list[Any] = []
        for a, b in zip(perc_a, perc_b, strict=True):
            p_a, p_b = (a, b) if percentages_only else (a[0], b[0])
            s_a, s_b = series_analysis(p_a), series_analysis(p_b)
            extra = [] if percentages_only else [*a, *b]
            out.append([s_a - s_b, s_a, s_b, *extra])
        return self.A._keyed(out, exclude_target=True)

    # ------------------------------------------------------------------
    # Deprecated camelCase API (guava-rt <= 0.5)
    useLabels = alias_property("use_labels", "useLabels")
    getVolDiff = alias(get_vol_diff, "getVolDiff")
    getSADiff = alias(get_sa_diff, "getSADiff")
    getROIDisplacementDiff = alias(get_roi_displacement_diff, "getROIDisplacementDiff")
    alignA2B = alias(align_a_to_b, "alignA2B")
    getBSDDiff = alias(get_bsd_diff, "getBSDDiff")
    getSeparationDistanceDiff = alias(get_separation_distance_diff, "getSeparationDistanceDiff")
    getPercentageOverlapDiff = alias(get_percentage_overlap_diff, "getPercentageOverlapDiff")


def _as_region(
    masks: RegionLike,
    target: int | str | None,
    side_target: int | str | None,
    anchor: int | str | ArrayLike | None,
    side_anchor: int | str | ArrayLike | None,
    labels: list[str] | None,
    dev: Device,
) -> Region:
    if isinstance(masks, Region):
        return masks
    target = target if target is not None else side_target
    anchor = anchor if anchor is not None else side_anchor
    if target is None or anchor is None:
        raise ValueError("building a Region from masks needs a target and an anchor")
    # a stacked array/tensor is split along its first axis, one mask per structure
    items = list(masks) if isinstance(masks, (np.ndarray, Tensor)) else masks
    return Region(*items, target=target, anchor=anchor, labels=labels, dev=dev)


def _as_list(values: dict[str, Any] | list[Any]) -> list[Any]:
    return list(values.values()) if isinstance(values, dict) else values
