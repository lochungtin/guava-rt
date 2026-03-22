from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from .distance_transform_edt import distance_transform_edt


class Mask:
    def __init__(self, mask, dev):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        self.dev = dev

        self.mask = mask.bool().to(self.dev)
        self.volume = self.mask.sum().to(torch.int32)

        self.center_of_mass = (
            torch.stack(torch.where(mask), dim=0).float().mean(dim=1).to(self.dev)
        )

        self.SHAPE = tuple(self.mask.shape)

        self._surface = None
        self._surface_area = None
        self._dmap = None
        self._sdmap = None

    def surface(self):
        if self._surface is None:
            self._surface = self.mask & ~_binary_erosion(self.mask)
        return self._surface

    def surface_area(self):
        if self._surface_area is None:
            self._surface_area = self.surface().sum().to(torch.int32)
        return self._surface_area

    def dmap(self):
        if self._dmap is None:
            self._dmap = distance_transform_edt(~self.mask)
        return self._dmap

    def sdmap(self):
        if self._sdmap is None:
            self._sdmap = distance_transform_edt(~self.surface())
        return self._sdmap

    # ------------------------------------------------------------------
    def getVolDiff(self, mask: Mask):
        val = (self.volume - mask.volume).item()
        return (
            val,
            val / mask.volume.item(),
            self.volume.item(),
            mask.volume.item(),
        )

    def getSADiff(self, mask: Mask):
        val = (self.surface_area() - mask.surface_area()).item()
        return (
            val,
            val / mask.surface_area().item(),
            self.surface_area().item(),
            mask.surface_area().item(),
        )

    # ------------------------------------------------------------------
    def maskDMap(self, mask):
        if isinstance(mask, Mask):
            m = mask.mask
        elif isinstance(mask, np.ndarray):
            m = torch.from_numpy(mask).bool().to(self.dev)
        else:
            m = mask.bool().to(self.dev)
        return self.dmap()[torch.where(m)], self.dmap() * m.float()

    # ------------------------------------------------------------------
    def getMaskCoordinates(self):
        return torch.where(self.mask)

    def getSurfaceCoordinates(self) -> torch.Tensor:
        return torch.argwhere(self.surface())

    # ------------------------------------------------------------------
    def alignTo(self, target: Mask):
        shift_vec = target.center_of_mass - self.center_of_mass

        x = self.mask.float().unsqueeze(0).unsqueeze(0)

        if self.mask.ndim == 3:
            D, H, W = self.SHAPE
            theta = torch.eye(3, 4, device=self.dev, dtype=torch.float32).unsqueeze(0)
            theta[0, 0, 3] = -2.0 * shift_vec[2].item() / max(W - 1, 1)  # W / x-axis
            theta[0, 1, 3] = -2.0 * shift_vec[1].item() / max(H - 1, 1)  # H / y-axis
            theta[0, 2, 3] = -2.0 * shift_vec[0].item() / max(D - 1, 1)  # D / z-axis
            grid = F.affine_grid(theta, (1, 1, D, H, W), align_corners=True)
        else:
            H, W = self.SHAPE
            theta = torch.eye(2, 3, device=self.dev, dtype=torch.float32).unsqueeze(0)
            theta[0, 0, 2] = -2.0 * shift_vec[1].item() / max(W - 1, 1)
            theta[0, 1, 2] = -2.0 * shift_vec[0].item() / max(H - 1, 1)
            grid = F.affine_grid(theta, (1, 1, H, W), align_corners=True)

        out = F.grid_sample(
            x, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )
        return Mask(out.reshape(self.mask.shape) > 0.5, self.dev)

    def getBSD(self, mask: Mask):
        self_aligned = self.alignTo(mask)
        st_d = self_aligned.sdmap()[torch.where(mask.surface())]
        ts_d = mask.sdmap()[torch.where(self_aligned.surface())]
        return torch.cat([st_d]), st_d, ts_d

    # ------------------------------------------------------------------
    def getRCVS(
        self,
        raySource: Mask,
        cutAwayDist: int = 8,
        N: int = 32,
        chunk_size: int | None = 1,
    ):
        ndim = self.mask.ndim

        cut_mask = (self.dmap() * raySource.mask.float() > cutAwayDist) & raySource.mask
        _ray_surface = Mask(cut_mask, self.dev).surface() | raySource.surface()

        source_coords = torch.argwhere(_ray_surface).float()
        self_coords = self.getSurfaceCoordinates().float()
        S, M = source_coords.shape[0], self_coords.shape[0]

        T = torch.linspace(2.0 / (N + 1), (N - 1.0) / (N + 1), N, device=self.dev)

        occupied = self.mask | _ray_surface

        def _check_chunk(src_chunk: torch.Tensor) -> torch.Tensor:
            sA = src_chunk[:, None, None, :]
            sB = self_coords[None, :, None, :]
            T_ = T[None, None, :, None]
            pts = sA + T_ * (sB - sA)

            idx = pts.round().long()
            for dim, size in enumerate(self.SHAPE):
                idx[..., dim].clamp_(0, size - 1)

            index_tuple = tuple(idx[..., dim] for dim in range(ndim))
            hit = occupied[index_tuple]

            blocked = hit.any(dim=2)
            return (~blocked).any(dim=0)

        if chunk_size is None or chunk_size >= S:
            visible = _check_chunk(source_coords)
        else:
            visible = torch.zeros(M, dtype=torch.bool, device=self.dev)
            for start in range(0, S, chunk_size):
                visible |= _check_chunk(source_coords[start : start + chunk_size])
                if visible.all():
                    break

        face = torch.zeros(self.SHAPE, dtype=torch.bool, device=self.dev)
        if visible.any():
            vc = self_coords[visible].long()
            index_tuple = tuple(vc[:, dim] for dim in range(ndim))
            face[index_tuple] = True
        return Mask(face, self.dev)


def _binary_erosion(mask: torch.BoolTensor):
    ndim = mask.ndim
    shape = mask.shape
    m = mask.float().unsqueeze(0).unsqueeze(0)
    if ndim == 3:
        kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device)
        kernel[0, 0, 1, :, :] = 1.0
        kernel[0, 0, :, 1, :] = 1.0
        kernel[0, 0, :, :, 1] = 1.0
        n_ones = int(kernel.sum().item())
        result = F.conv3d(m, kernel, padding=1).reshape(shape)
    else:
        kernel = torch.zeros(1, 1, 3, 3, device=mask.device)
        kernel[0, 0, 1, :] = 1.0
        kernel[0, 0, :, 1] = 1.0
        n_ones = int(kernel.sum().item())
        result = F.conv2d(m, kernel, padding=1).reshape(shape)
    return (result >= n_ones) & mask
