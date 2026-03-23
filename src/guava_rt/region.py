from __future__ import annotations

import numpy as np
import torch

from .mask import Mask


class Region:
    def __init__(
        self,
        *masks,
        target: int | str,
        anchor: int | str | np.ndarray | torch.Tensor,
        labels: list[str] | None = None,
        dev: str,
    ):
        self.masks = [m if isinstance(m, Mask) else Mask(m, dev) for m in masks]
        self.useLabels = labels is not None
        self.labels = labels
        self.dev = dev

        self.target_idx = self.labels.index(target) if self.useLabels else target
        self.target = self.target_idx
        self.target_mask: Mask = self.masks[self.target]

        if isinstance(anchor, np.ndarray):
            self.anchor = torch.from_numpy(anchor).float().to(self.dev)
        elif isinstance(anchor, torch.Tensor):
            self.anchor = anchor.float().to(self.dev)
        else:
            anchor_idx = self.labels.index(anchor) if self.useLabels else anchor
            self.anchor = self.masks[anchor_idx].center_of_mass

        self.target_dmap = self.target_mask.dmap()

    # ------------------------------------------------------------------
    def _getDisplacementVector(self, mask: Mask, useAnchor: bool) -> torch.Tensor:
        ref = self.anchor if useAnchor else self.target_mask.center_of_mass
        return mask.center_of_mass - ref

    def getDisplacementVectors(self, useAnchor: bool = False):
        out = [self._getDisplacementVector(m, useAnchor) for m in self.masks]
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    # ------------------------------------------------------------------
    def _maskedDMap(self, mask: Mask, mode: str, chunk_size: int):
        if mode == "surface":
            _mask = mask.surface()
        elif mode == "rcvs":
            _mask = mask.getRCVS(self.target_mask, chunk_size=chunk_size).mask
        else:
            _mask = mask.mask
        return *self.target_mask.maskDMap(_mask), _mask

    # ------------------------------------------------------------------
    def getSeparationDistances(
        self,
        mode: str = "volume",
        distances_only: bool = True,
        chunk_size: int = 1,
    ):
        out = [
            (
                self._maskedDMap(m, mode, chunk_size)[0]
                if distances_only
                else self._maskedDMap(m, mode, chunk_size)
            )
            for i, m in enumerate(self.masks)
            if i != self.target_idx
        ]
        if self.useLabels:
            non_target = [l for i, l in enumerate(self.labels) if i != self.target_idx]
            return dict(zip(non_target, out))
        return out

    # ------------------------------------------------------------------
    def getThresholdedOverlapPercentages(
        self,
        mode: str = "volume",
        percentages_only: bool = True,
        chunk_size: int = 1,
    ):
        max_dist = int(self.target_dmap.max().ceil().item())
        out = []

        for i, m in enumerate(self.masks):
            if i == self.target_idx:
                continue

            d_vals, _masked, _mask = self._maskedDMap(m, mode, chunk_size)
            d_min = int(d_vals.min().ceil().item())
            d_max = int(d_vals.max().ceil().item())
            n_mask = _mask.float().sum()

            percs = torch.zeros(max(d_max - d_min + 1, 1), device=self.dev)
            full = torch.zeros(max_dist + 1, device=self.dev)

            for j, thresh in enumerate(range(d_min, d_max + 1)):
                p = (d_vals <= thresh).float().sum() / n_mask
                percs[j] = p
                if thresh <= max_dist:
                    full[thresh] = p

            if d_max <= max_dist:
                full[d_max:] = 1.0

            out.append(percs if percentages_only else [percs, full, _masked, _mask])

        if self.useLabels:
            non_target = [l for i, l in enumerate(self.labels) if i != self.target_idx]
            return dict(zip(non_target, out))
        return out
        return out
        return out
        return out
        return out
