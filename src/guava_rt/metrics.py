from __future__ import annotations

import numpy as np
import torch

from .mask import Mask
from .region import Region
from .utils import seriesAnalysis


class Metrics:
    def __init__(
        self,
        A: list[Mask | torch.Tensor | np.ndarray] | np.ndarray | torch.Tensor | Region,
        B: list[Mask | torch.Tensor | np.ndarray] | np.ndarray | torch.Tensor | Region,
        target: int | str | None = None,
        target_A: int | str | None = None,
        target_B: int | str | None = None,
        anchor: int | str | np.ndarray | torch.Tensor | None = None,
        anchor_A: int | str | np.ndarray | torch.Tensor | None = None,
        anchor_B: int | str | np.ndarray | torch.Tensor | None = None,
        labels: list[str] | None = None,
        dev: str = "cpu",
    ):
        if isinstance(A, list) or isinstance(A, np.ndarray):
            A = Region(
                *A,
                target=target if target is not None else target_A,
                anchor=anchor if anchor is not None else anchor_A,
                labels=labels,
                dev=dev,
            )
        self.A = A

        if isinstance(B, list) or isinstance(B, np.ndarray):
            B = Region(
                *B,
                target=target if target is not None else target_B,
                anchor=anchor if anchor is not None else anchor_B,
                labels=labels,
                dev=dev,
            )
        self.B = B

        self.useLabels = A.useLabels
        self.labels = A.labels

        self.A.useLabels = False
        self.B.useLabels = False

        self.bsd = None

        self.dev = dev

    # ------------------------------------------------------------------
    def getVolDiff(self):
        out = [a.getVolDiff(b) for a, b in zip(self.A.masks, self.B.masks)]
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getSADiff(self):
        out = [a.getSADiff(b) for a, b in zip(self.A.masks, self.B.masks)]
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    # ------------------------------------------------------------------
    def getROIDisplacementDiff(self):
        dVecA = torch.stack(self.A.getDisplacementVectors(useAnchor=True))
        dVecB = torch.stack(self.B.getDisplacementVectors(useAnchor=True))
        out = dVecA - dVecB
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    # ------------------------------------------------------------------
    def getBSDDiff(self, mode: str = "all"):
        if self.bsd is None:
            self.bsd = [a.getBSD(b)[0] for a, b in zip(self.A.masks, self.B.masks)]
        ml = mode.lower()
        if ml == "asd":
            out = [seriesAnalysis(b)[3] for b in self.bsd]
        elif ml == "hd":
            out = [seriesAnalysis(b)[-1] for b in self.bsd]
        elif ml == "hd95":
            out = [seriesAnalysis(b)[-2] for b in self.bsd]
        else:
            out = [seriesAnalysis(b) for b in self.bsd]

        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    # ------------------------------------------------------------------
    def getSeparationDistanceDiff(
        self,
        mask: str = "volume",
        mode: str = "all",
        distances_only: bool = True,
        chunk_size: int = 1,
    ):
        distA = self.A.getSeparationDistances(
            mode=mask, distances_only=distances_only, chunk_size=chunk_size
        )
        distB = self.B.getSeparationDistances(
            mode=mask, distances_only=distances_only, chunk_size=chunk_size
        )

        ml = mode.lower()
        if ml == "max":
            f = lambda x: x.float().max()
        elif ml == "min":
            f = lambda x: x.float().min()
        elif ml == "p5":
            f = lambda x: torch.quantile(x.float(), 0.05)
        elif ml == "p95":
            f = lambda x: torch.quantile(x.float(), 0.95)
        elif ml == "mean":
            f = lambda x: x.float().mean()
        elif ml == "median":
            f = lambda x: x.float().median()
        else:
            f = seriesAnalysis

        if distances_only:
            out = [[f(dA) - f(dB), f(dA), f(dB)] for dA, dB in zip(distA, distB)]
        else:
            out = [
                [f(a[0]) - f(b[0]), f(a[0]), f(b[0]), *a, *b]
                for a, b in zip(distA, distB)
            ]

        if self.useLabels:
            non_target = [
                l for i, l in enumerate(self.labels) if i != self.A.target_idx
            ]
            return dict(zip(non_target, out))
        return out

    # ------------------------------------------------------------------
    def getPercentageOverlapDiff(
        self,
        mode: str = "volume",
        percentages_only: bool = True,
        chunk_size: int = 1,
    ):
        percA = self.A.getThresholdedOverlapPercentages(
            mode=mode, percentages_only=percentages_only, chunk_size=chunk_size
        )
        percB = self.B.getThresholdedOverlapPercentages(
            mode=mode, percentages_only=percentages_only, chunk_size=chunk_size
        )

        if percentages_only:
            out = [
                [
                    seriesAnalysis(pA) - seriesAnalysis(pB),
                    seriesAnalysis(pA),
                    seriesAnalysis(pB),
                ]
                for pA, pB in zip(percA, percB)
            ]
        else:
            out = [
                [
                    seriesAnalysis(a[0]) - seriesAnalysis(b[0]),
                    seriesAnalysis(a[0]),
                    seriesAnalysis(b[0]),
                    *a,
                    *b,
                ]
                for a, b in zip(percA, percB)
            ]

        if self.useLabels:
            non_target = [
                l for i, l in enumerate(self.labels) if i != self.A.target_idx
            ]
            return dict(zip(non_target, out))
        return out
