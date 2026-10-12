"""Exact Euclidean distance transform in PyTorch (a drop-in for ``scipy.ndimage``'s)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def distance_transform_edt(
    input: Tensor,
    sampling: float | Sequence[float] | None = None,
) -> Tensor:
    """Distance from every non-zero voxel to the nearest zero voxel, like SciPy's.

    Supports 2-D and 3-D inputs on any device; ``sampling`` is the voxel spacing per axis (or one
    value for all axes). Computed in float64, returned as float32. Voxels with no zero voxel
    anywhere in the input are ``inf``.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)!r}")

    ndim = input.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Only 2-D and 3-D inputs are supported, got ndim={ndim}")

    if sampling is None:
        spacing = [1.0] * ndim
    elif isinstance(sampling, (int, float)):
        spacing = [float(sampling)] * ndim
    else:
        spacing = [float(s) for s in sampling]

    if len(spacing) != ndim:
        raise ValueError(f"len(sampling)={len(spacing)} does not match input.ndim={ndim}")

    # squared distances: 0 at background (zero) voxels, inf elsewhere until resolved
    sq = torch.where(
        input == 0,
        input.new_zeros(input.shape, dtype=torch.float64),
        input.new_full(input.shape, float("inf"), dtype=torch.float64),
    )

    # separable: an exact 1-D pass along the first axis, then lower envelopes of parabolas
    for axis, sp in enumerate(spacing):
        sq = _edt_axis_binary(sq, axis, sp) if axis == 0 else _edt_axis(sq, axis, sp)

    return sq.sqrt().to(torch.float32)


def _edt_axis_binary(sq: Tensor, axis: int, sp: float) -> Tensor:
    """First pass on a binary 0/inf input: squared distance to the nearest 0 along ``axis``."""
    sq = sq.movedim(axis, -1)
    shape = sq.shape
    L = shape[-1]
    M = sq.numel() // L
    g = sq.reshape(M, L)

    device = g.device
    sp2 = sp * sp
    INF = float("inf")

    q = torch.arange(L, dtype=torch.float64, device=device)

    # index of the nearest background voxel to the left (cummax) and right (reversed cummin)
    bg = torch.isfinite(g)
    bg_pos_l = torch.where(bg, q.unsqueeze(0).expand(M, L), g.new_full((M, L), -INF))
    nearest_l = torch.cummax(bg_pos_l, dim=1).values
    d_left_sq = torch.where(
        nearest_l == -INF,
        g.new_full((M, L), INF),
        sp2 * (q - nearest_l) ** 2,
    )

    bg_pos_r = torch.where(bg, q.unsqueeze(0).expand(M, L), g.new_full((M, L), INF))
    nearest_r = torch.cummin(bg_pos_r.flip(1), dim=1).values.flip(1)
    d_right_sq = torch.where(
        nearest_r == INF,
        g.new_full((M, L), INF),
        sp2 * (nearest_r - q) ** 2,
    )

    result = torch.minimum(d_left_sq, d_right_sq)
    return result.reshape(shape).movedim(-1, axis)


def _edt_axis(sq: Tensor, axis: int, sp: float) -> Tensor:
    """Later passes: combine the squared distances so far along ``axis`` (Meijster)."""
    sq = sq.movedim(axis, -1).contiguous()
    shape = sq.shape
    L = shape[-1]
    M = sq.numel() // L

    out = _run_meijster(sq.reshape(M, L), sp)
    return out.reshape(shape).movedim(-1, axis)


def _meijster(g: Tensor, sp: float) -> Tensor:
    """1-D squared EDT of every row of ``g`` (``(M, L)``), batched over rows.

    Builds each row's lower envelope of parabolas ``sp² (x - q)² + g[q]`` (``v``: parabola
    positions, ``z``: envelope breakpoints, ``ks``: index of the last parabola), then evaluates
    the envelope at every position.
    """
    M, L = g.shape
    device = g.device
    sp2 = sp * sp
    INF = float("inf")

    if L == 0:
        return g.clone()

    v = g.new_zeros((M, L), dtype=torch.long)
    z = g.new_full((M, L + 1), INF)
    z[:, 0] = -INF
    ks = g.new_full((M,), -1, dtype=torch.long)

    bidx = torch.arange(M, dtype=torch.long, device=device)

    active_cols: list[int] = torch.isfinite(g).any(dim=0).nonzero(as_tuple=True)[0].tolist()

    # rows advance together; ``active`` marks rows whose parabola at q is still being inserted
    for q in active_cols:
        gq_all = g[:, q]
        active = torch.isfinite(gq_all)
        if not active.any():
            continue

        while active.any():
            bi = bidx[active]
            ki = ks[bi]

            is_empty = ki < 0
            if is_empty.any():
                be = bi[is_empty]
                ks[be] = 0
                v[be, 0] = q
                z[be, 1] = INF
                active[be] = False

            ne = ~is_empty
            if not ne.any():
                break

            bn = bi[ne]
            kn = ki[ne]
            vk = v[bn, kn]
            vk_f = vk.double()
            q_f = float(q)
            num = sp2 * (q_f * q_f - vk_f * vk_f) + gq_all[bn] - g[bn, vk]
            den = 2.0 * sp2 * (q_f - vk_f)
            s = num / den

            pop = s <= z[bn, kn]
            no_pop = ~pop

            if pop.any():
                popped = bn[pop]
                ks[popped] -= 1
                z[popped, ks[popped] + 1] = INF

            if no_pop.any():
                bp = bn[no_pop]
                nk = ks[bp] + 1
                ks[bp] = nk
                v[bp, nk] = q
                z[bp, nk] = s[no_pop]
                z[bp, nk + 1] = INF
                active[bp] = False

    has_bg = ks >= 0

    if not has_bg.any():
        return g.new_full((M, L), INF)

    # evaluate: for each position, find its envelope segment via the breakpoints
    q_pos = torch.arange(L, dtype=z.dtype, device=device)

    kp = (
        torch.searchsorted(
            z,
            q_pos.unsqueeze(0).expand(M, L).contiguous(),
            right=True,
        )
        .sub_(1)
        .clamp_(0, L - 1)
    )

    vi = torch.gather(v, 1, kp)
    g_vi = torch.gather(g, 1, vi)

    result = q_pos.unsqueeze(0).double().expand(M, L).clone()
    result.sub_(vi.double()).pow_(2).mul_(sp2).add_(g_vi)

    result[~has_bg] = INF
    return result


# torch.compile speeds up the Python loop in _meijster considerably. It is attempted once at
# import, and if compiling or running the compiled function fails, eager mode is used from then on.
_meijster_impl: Callable[[Tensor, float], Tensor] = _meijster
try:
    _meijster_impl = torch.compile(_meijster)
except Exception:
    pass


def _run_meijster(g: Tensor, sp: float) -> Tensor:
    global _meijster_impl
    if _meijster_impl is _meijster:
        return _meijster(g, sp)
    try:
        return _meijster_impl(g, sp)
    except Exception:
        _meijster_impl = _meijster
        return _meijster(g, sp)
