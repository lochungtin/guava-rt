from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, center_of_mass, distance_transform_edt, shift

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Metric functions using: {DEV}")

SERIES_ANALYSIS_LABELS = ["MIN", "P05", "P10", "AVG", "MDN", "P90", "P95", "MAX"]


class Mask:
    def __init__(
        self,
        mask: np.array,
    ):
        self.mask = mask.astype(bool)
        if len(self.mask.shape) == 3:
            erosion_struct = np.zeros((3, 3, 3))
            erosion_struct[1, :, :] = 1
            erosion_struct[:, 1, :] = 1
            erosion_struct[:, :, 1] = 1
        else:
            erosion_struct = np.zeros((3, 3))
            erosion_struct[1, :] = 1
            erosion_struct[:, 1] = 1
        self.surface = self.mask & ~binary_erosion(self.mask, structure=erosion_struct)
        self.volume = np.sum(self.mask).astype(np.int32)
        self.surface_area = np.sum(self.surface).astype(np.int32)
        self.center_of_mass = np.asarray(center_of_mass(self.mask))
        self.SHAPE = self.mask.shape
        self.dmap = None

    def getVolDiff(self, mask: Mask, diff_only=True):
        if diff_only:
            return mask.volume - self.volume
        return mask.volume - self.volume, self.volume, mask.volume

    def getSADiff(self, mask: Mask, diff_only=True):
        return (
            mask.surface_area - self.surface_area,
            self.surface_area,
            mask.surface_area,
        )

    def getDMap(self):
        if self.dmap is None:
            self.dmap = distance_transform_edt(~self.mask)
        return self.dmap

    def maskDMap(self, mask: Mask):
        m = mask
        if isinstance(mask, Mask):
            m = mask.mask
        return self.getDMap()[np.where(m)], self.getDMap() * m.astype(int)

    def getMaskCoordinates(self):
        return np.where(self.mask)

    def getSurfaceCoordinates(self):
        return np.argwhere(self.surface)

    def alignTo(self, target: Mask):
        tVec = target.center_of_mass - self.center_of_mass
        return Mask(shift(self.mask.astype(float), tVec, order=0) > 0.5)

    def getBSD(self, mask: Mask, bi_only: bool = True):
        selfAligned = self.alignTo(mask)
        stD, _ = selfAligned.maskDMap(mask.surface)
        tsD, _ = mask.maskDMap(selfAligned.surface)

        print(selfAligned.center_of_mass)
        print(mask.center_of_mass)

        if bi_only:
            return np.concatenate([stD, tsD])
        return np.concatenate([stD, tsD]), stD, tsD

    def getRCVS(self, raySource: Mask, cutAwayDist: int = 8, N: int = 32):
        _raySource = Mask(
            (self.getDMap() * raySource.mask > cutAwayDist) & raySource.mask
        ).surface
        _raySource |= raySource.surface

        sourceSurfaceCoords = np.argwhere(_raySource)
        selfSurfaceCoords = self.getSurfaceCoordinates()

        visible = np.zeros(len(selfSurfaceCoords), dtype=bool)

        sourceMaskTensor = torch.tensor(_raySource, dtype=torch.bool, device=DEV)
        selfMaskTensor = torch.tensor(self.mask, dtype=torch.bool, device=DEV)

        T = torch.linspace(2.0 / (N + 1), (N - 1.0) / (N + 1), N, device=DEV)

        for src in sourceSurfaceCoords:
            todo = np.where(~visible)[0]
            if len(todo) == 0:
                break

            _sA = torch.tensor(src, dtype=torch.float32, device=DEV)
            _sB = torch.tensor(selfSurfaceCoords[todo], dtype=torch.float32, device=DEV)

            pts = _sA[None, None, :] + T[None, :, None] * (_sB - _sA)[:, None, :]

            idx = pts.round().long()

            idx[..., 0].clamp_(0, self.SHAPE[0] - 1)
            idx[..., 1].clamp_(0, self.SHAPE[1] - 1)
            idx[..., 2].clamp_(0, self.SHAPE[2] - 1)

            x, y, z = idx[..., 0], idx[..., 1], idx[..., 2]
            blocked = (selfMaskTensor[x, y, z] | sourceMaskTensor[x, y, z]).any(dim=1)
            clear = (~blocked).cpu().numpy()

            visible[todo[clear]] = True

        face = np.zeros(self.SHAPE, dtype=bool)
        face[tuple(selfSurfaceCoords[visible].T)] = True
        return Mask(face)


class Region:
    def __init__(
        self,
        *masks: Mask,
        target: int | str,
        anchor: int | str | np.ndarray,
        labels: list[str] | None = None,
    ):
        self.masks = [m if isinstance(m, Mask) else Mask(m) for m in masks]
        self.useLabels = labels != None
        self.labels = labels

        self.target_idx = self.labels.index(target)
        if self.useLabels:
            target = self.target_idx
        self.target = target
        self.target_mask = self.masks[self.target]

        if isinstance(anchor, np.ndarray):
            self.anchor = anchor
        else:
            if self.useLabels:
                anchor = self.labels.index(anchor)
            self.anchor = self.masks[anchor].center_of_mass

        self.target_dmap = self.target_mask.getDMap()

    def _getDisplacementVector(self, mask, useAnchor=False):
        if useAnchor:
            return mask.center_of_mass - self.anchor
        return mask.center_of_mass - self.target_mask.center_of_mass

    def getDisplacementVectors(self, useAnchor=False):
        out = [self._getDisplacementVector(m, useAnchor) for m in self.masks]
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def _maskedDMap(self, mask: Mask, mode: str = "volume"):
        if mode == "surface":
            _mask = mask.surface
        elif mode == "rcvs":
            _mask = mask.getRCVS(self.target_mask).mask
        else:
            _mask = mask.mask

        return *self.target_mask.maskDMap(_mask), _mask

    def getSeparationDistances(self, mode: str = "volume", distances_only: bool = True):
        _slice = 1 if distances_only else 3

        out = [
            self._maskedDMap(m, mode)[:_slice]
            for i, m in enumerate(self.masks)
            if i != self.target_idx
        ]
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getThresholdedOverlapPercentages(
        self, r="sweep", mode="volume", percentages_only=True
    ):
        maxDist = np.ceil(np.max(self.target_dmap)).astype(int)
        out = []
        for i, m in enumerate(self.masks):
            if i == self.target_idx:
                continue

            dVals, _masked, _mask = self._maskedDMap(m, mode)
            dMin = np.ceil(np.min(dVals)).astype(int)
            dMax = np.ceil(np.max(dVals)).astype(int)

            percs = np.zeros(dMax - dMin)
            full = np.zeros(maxDist + 1)
            for j, r in enumerate(range(dMin, dMax)):
                p = np.sum(dVals <= r) / np.sum(_mask)
                percs[j] = p
                full[r] = p
            full[dMax:] = 1

            if percentages_only:
                out.append(percs)
            else:
                out.append([percs, full, _masked, _mask])
        if self.useLabels:
            return dict(zip(self.labels, out))
        return out


class Compare:
    def __init__(
        self,
        A: Region,
        B: Region,
    ):
        self.A = A
        self.B = B
        self.useLabels = A.useLabels
        self.labels = A.labels

        self.A.useLabels = False
        self.B.useLabels = False

        self.bsd = None

    def getVolDiff(self, diff_only=False):
        out = [
            a.getVolDiff(b, diff_only=diff_only)
            for a, b in zip(self.A.masks, self.B.masks)
        ]

        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getSADiff(self, diff_only=False):
        out = [
            a.getSADiff(b, diff_only=diff_only)
            for a, b in zip(self.A.masks, self.B.masks)
        ]

        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getROIDisplacementDiff(self):
        dVecA = np.array(self.A.getDisplacementVectors(useAnchor=True))
        dVecB = np.array(self.B.getDisplacementVectors(useAnchor=True))

        out = [[*vec, np.linalg.norm(vec)] for vec in dVecA - dVecB]

        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getBSDDiff(self, mode="all"):
        if self.bsd == None:
            self.bsd = [
                a.getBSD(b, bi_only=True) for a, b in zip(self.A.masks, self.B.masks)
            ]

        if mode.lower() == "asd":
            out = [seriesAnalysis(b)[3] for b in self.bsd]
        elif mode.lower() == "hd":
            out = [seriesAnalysis(b)[-1] for b in self.bsd]
        elif mode.lower() == "hd95":
            out = [seriesAnalysis(b)[-2] for b in self.bsd]
        else:
            out = [seriesAnalysis(b) for b in self.bsd]

        if self.useLabels:
            return dict(zip(self.labels, out))
        return out

    def getSeparationDistanceDiff(self, mask="volume", mode="all", distances_only=True):
        distA = self.A.getSeparationDistances(mode=mask, distances_only=distances_only)
        distB = self.B.getSeparationDistances(mode=mask, distances_only=distances_only)

        if mode.lower() == "max":
            f = np.max
        elif mode.lower() == "min":
            f = np.min
        elif mode.lower() == "p5":
            f = lambda x: np.percentile(x, 5)
        elif mode.lower() == "p95":
            f = lambda x: np.percentile(x, 95)
        elif mode.lower() == "mean":
            f = np.mean
        elif mode.lower() == "median":
            f = np.median
        else:
            f = seriesAnalysis

        if distances_only:
            out = [[f(dA) - f(dB), f(dA), f(dB)] for dA, dB in zip(distA, distB)]
        else:
            out = [
                [f(dA) - f(dB), f(dA), f(dB), dA, dmA, mA, dB, dmB, mB]
                for (dA, dmA, mA), (dB, dmB, mB) in zip(distA, distB)
            ]

        if self.useLabels:
            labels = [*self.labels]
            labels.pop(self.A.target_idx)
            return dict(zip(labels, out))
        return out

    def getPercentageOverlapDiff(self, mode="volume", percentages_only=True):
        percA = self.A.getThresholdedOverlapPercentages(
            mode=mode, percentages_only=percentages_only
        )
        percB = self.B.getThresholdedOverlapPercentages(
            mode=mode, percentages_only=percentages_only
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
                    seriesAnalysis(pA) - seriesAnalysis(pB),
                    seriesAnalysis(pA),
                    seriesAnalysis(pB),
                    pA,
                    fA,
                    dmA,
                    mA,
                    pB,
                    fB,
                    dmB,
                    mB,
                ]
                for (pA, fA, dmA, mA), (pB, fB, dmB, mB) in zip(percA, percB)
            ]

        if self.useLabels:
            labels = [*self.labels]
            labels.pop(self.A.target_idx)
            return dict(zip(labels, out))
        return out


# === Helper Functions ========================================================


def seriesAnalysis(arr, labels=False):
    out = np.array(
        [
            np.min(arr),
            np.percentile(arr, 5),
            np.percentile(arr, 10),
            np.mean(arr),
            np.median(arr),
            np.percentile(arr, 90),
            np.percentile(arr, 95),
            np.max(arr),
        ]
    )
    if labels:
        return out, SERIES_ANALYSIS_LABELS
    return out


def prettyPrintTable(rowNames, rows, labels, rounding=3, minWidth=0):
    def pad(data, gaps):
        return [str(d).ljust(g + 1) for d, g in zip(data, gaps)]

    for r in range(len(rows)):
        rows[r] = [round(i, rounding) if isinstance(i, float) else i for i in rows[r]]

    gaps = [max(map(len, rowNames))]
    for i in range(len(labels)):
        colMax = len(labels[i])
        for row in rows:
            colMax = max(len(str(row[i])), colMax, minWidth)
        gaps.append(colMax)

    rows = [list(map(str, row)) for row in rows]

    rowSep = "+" + "+".join("-" * (i + 2) for i in gaps) + "+"
    print(rowSep)
    print("| " + "| ".join(pad([""] + labels, gaps)) + "|")
    print(rowSep)
    for name, row in zip(rowNames, rows):
        print("| " + "| ".join(pad([name] + row, gaps)) + "|")
        print(rowSep)
