"""A single binary mask and the geometric quantities derived from it."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

from guava_rt._deprecation import alias, alias_property, renamed_kwargs
from guava_rt.distance_transform_edt import distance_transform_edt

ArrayLike = npt.NDArray[Any] | Tensor
Device = str | torch.device


class Mask:
    """Binary segmentation mask (2-D or 3-D) on one device.

    ``volume`` and ``center_of_mass`` are computed on construction; ``surface``,
    ``surface_area``, ``dmap`` and ``sdmap`` are computed on first call and cached.
    Coordinates are in voxels, ordered like the tensor dimensions (``(z, y, x)`` or ``(y, x)``).
    """

    def __init__(self, mask: ArrayLike, dev: Device = "cpu") -> None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        self.dev = dev
        self.mask: Tensor = mask.bool().to(self.dev)
        self.shape: tuple[int, ...] = tuple(self.mask.shape)

        self.volume: Tensor = self.mask.sum().to(torch.int32)
        self.center_of_mass: Tensor = torch.argwhere(self.mask).float().mean(dim=0)

        self._surface: Tensor | None = None
        self._surface_area: Tensor | None = None
        self._dmap: Tensor | None = None
        self._sdmap: Tensor | None = None

    def surface(self) -> Tensor:
        """Boundary voxels: ``X ∩ ¬Erode(X)`` (see :func:`_binary_erosion`)."""
        if self._surface is None:
            self._surface = self.mask & ~_binary_erosion(self.mask)
        return self._surface

    def surface_area(self) -> Tensor:
        """Number of boundary voxels."""
        if self._surface_area is None:
            self._surface_area = self.surface().sum().to(torch.int32)
        return self._surface_area

    def dmap(self) -> Tensor:
        """Distance from every voxel to the nearest mask voxel (0 inside the mask)."""
        if self._dmap is None:
            self._dmap = distance_transform_edt(~self.mask)
        return self._dmap

    def sdmap(self) -> Tensor:
        """Distance from every voxel to the nearest surface voxel."""
        if self._sdmap is None:
            self._sdmap = distance_transform_edt(~self.surface())
        return self._sdmap

    def numpy(self) -> npt.NDArray[np.bool_]:
        return self.mask.cpu().numpy()

    # ------------------------------------------------------------------
    def get_vol_diff(self, mask: Mask) -> tuple[int, float, int, int]:
        """``(V_self - V_mask, relative to V_mask, V_self, V_mask)``."""
        val = int((self.volume - mask.volume).item())
        return val, val / mask.volume.item(), int(self.volume.item()), int(mask.volume.item())

    def get_sa_diff(self, mask: Mask) -> tuple[int, float, int, int]:
        """``(S_self - S_mask, relative to S_mask, S_self, S_mask)`` for surface areas."""
        sa_self, sa_mask = int(self.surface_area().item()), int(mask.surface_area().item())
        val = sa_self - sa_mask
        return val, val / sa_mask, sa_self, sa_mask

    # ------------------------------------------------------------------
    def mask_dmap(self, mask: Mask | ArrayLike) -> tuple[Tensor, Tensor]:
        """This mask's :meth:`dmap` at ``mask``'s voxels, and the dmap zeroed outside ``mask``."""
        if isinstance(mask, Mask):
            m = mask.mask
        elif isinstance(mask, np.ndarray):
            m = torch.from_numpy(mask).bool().to(self.dev)
        else:
            m = mask.bool().to(self.dev)
        return self.dmap()[m], self.dmap() * m.float()

    # ------------------------------------------------------------------
    def get_mask_coordinates(self) -> Tensor:
        """Coordinates of all mask voxels, shape ``(N, ndim)``."""
        return torch.argwhere(self.mask)

    def get_surface_coordinates(self) -> Tensor:
        """Coordinates of all surface voxels, shape ``(N, ndim)``."""
        return torch.argwhere(self.surface())

    # ------------------------------------------------------------------
    def align_to(
        self,
        target: Mask | None = None,
        shift_vec: ArrayLike | None = None,
    ) -> Mask:
        """Translate this mask by ``shift_vec`` (voxels), or onto ``target``'s centre of mass.

        Uses nearest-neighbour resampling; content shifted outside the grid is lost.
        """
        if shift_vec is None:
            if target is None:
                raise ValueError("align_to needs either a target mask or a shift_vec")
            shift_vec = target.center_of_mass - self.center_of_mass
        shift = [float(s) for s in shift_vec]

        # affine_grid works in normalised [-1, 1] coordinates with axes in x, y(, z) order,
        # i.e. the reverse of the tensor dimensions
        ndim = self.mask.ndim
        theta = torch.eye(ndim, ndim + 1, device=self.dev, dtype=torch.float32).unsqueeze(0)
        for i, size in enumerate(reversed(self.shape)):
            theta[0, i, ndim] = -2.0 * shift[ndim - 1 - i] / max(size - 1, 1)
        grid = F.affine_grid(theta, [1, 1, *self.shape], align_corners=True)

        x = self.mask.float()[None, None]
        out = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return Mask(out.reshape(self.shape) > 0.5, self.dev)

    def get_bsd(self, mask: Mask) -> tuple[Tensor, Tensor, Tensor]:
        """Bidirectional surface distances to ``mask`` after centre-of-mass alignment.

        Returns ``(BSD, SD_mask, SD_self)``: the union of both directions, the distances from
        ``mask``'s surface to this (aligned) surface, and from this surface to ``mask``'s.
        """
        self_aligned = self.align_to(mask)
        st_d = self_aligned.sdmap()[mask.surface()]
        ts_d = mask.sdmap()[self_aligned.surface()]
        return torch.cat([st_d, ts_d]), st_d, ts_d

    # ------------------------------------------------------------------
    @renamed_kwargs(raySource="ray_source", cutAwayDist="cut_away_dist", N="n_steps")
    def get_rcvs(
        self,
        ray_source: Mask,
        cut_away_dist: float = 8,
        n_steps: int = 32,
        chunk_size: int | None = 1,
    ) -> Mask:
        """Ray-cast visible surface: the part of this surface with line of sight to ``ray_source``.

        Rays start on ``ray_source``'s surface plus an inner shell ``cut_away_dist`` voxels deep.
        Each ray is sampled at ``n_steps`` evenly spaced interior points, each rounded to its
        voxel. The ray is blocked if any sample lands in this mask or on the ray-source shells,
        other than in the ray's own start or end voxel. Source points are processed
        ``chunk_size`` at a time (``None``: all at once) to bound memory, which is
        ``O(chunk_size * |surface| * n_steps)``.
        """
        ndim = self.mask.ndim

        cut_mask = (self.dmap() * ray_source.mask.float() > cut_away_dist) & ray_source.mask
        ray_surface = Mask(cut_mask, self.dev).surface() | ray_source.surface()

        source_coords = torch.argwhere(ray_surface)
        self_coords = self.get_surface_coordinates()
        n_source, n_self = source_coords.shape[0], self_coords.shape[0]

        t = torch.arange(1, n_steps + 1, device=self.dev, dtype=torch.float32) / (n_steps + 1)

        occupied = self.mask | ray_surface

        def check_chunk(src_chunk: Tensor) -> Tensor:
            # (source, target, step, ndim) sample voxels along every source → target ray
            src = src_chunk[:, None, None, :]
            dst = self_coords[None, :, None, :]
            idx = (src + t[None, None, :, None] * (dst - src)).round().long()

            # samples rounding onto the ray's own end voxels are not obstructions: the start
            # voxel is on the source shell and the end voxel is in this mask by definition
            at_end = (idx == src).all(dim=-1) | (idx == dst).all(dim=-1)
            hit = occupied[tuple(idx[..., dim] for dim in range(ndim))] & ~at_end
            blocked = hit.any(dim=2)
            return (~blocked).any(dim=0)

        if chunk_size is None or chunk_size >= n_source:
            visible = check_chunk(source_coords)
        else:
            visible = torch.zeros(n_self, dtype=torch.bool, device=self.dev)
            for start in range(0, n_source, chunk_size):
                visible |= check_chunk(source_coords[start : start + chunk_size])
                if visible.all():
                    break

        face = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        if visible.any():
            vc = self_coords[visible]
            face[tuple(vc[:, dim] for dim in range(ndim))] = True
        return Mask(face, self.dev)

    # ------------------------------------------------------------------
    # Deprecated camelCase API (guava-rt <= 0.5)
    SHAPE = alias_property("shape", "SHAPE")
    getVolDiff = alias(get_vol_diff, "getVolDiff")
    getSADiff = alias(get_sa_diff, "getSADiff")
    maskDMap = alias(mask_dmap, "maskDMap")
    getSurfaceCoordinates = alias(get_surface_coordinates, "getSurfaceCoordinates")
    alignTo = alias(align_to, "alignTo")
    getBSD = alias(get_bsd, "getBSD")
    getRCVS = alias(get_rcvs, "getRCVS")

    def getMaskCoordinates(self) -> tuple[Tensor, ...]:
        """Deprecated: use :meth:`get_mask_coordinates`, which returns an ``(N, ndim)`` tensor."""
        warnings.warn(
            "getMaskCoordinates is deprecated; use get_mask_coordinates instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.where(self.mask)


def _binary_erosion(mask: Tensor) -> Tensor:
    """Erosion by the union of the central axis-aligned planes of a 3^ndim cube.

    That is the 18-neighbourhood in 3-D and the 4-neighbourhood (cross) in 2-D.
    """
    ndim = mask.ndim
    kernel = torch.zeros((1, 1) + (3,) * ndim, device=mask.device)
    for dim in range(ndim):
        idx: list[int | slice] = [0, 0, *([slice(None)] * ndim)]
        idx[2 + dim] = 1
        kernel[tuple(idx)] = 1.0
    n_ones = int(kernel.sum().item())

    conv = F.conv3d if ndim == 3 else F.conv2d
    result = conv(mask.float()[None, None], kernel, padding=1).reshape(mask.shape)
    return (result >= n_ones) & mask
