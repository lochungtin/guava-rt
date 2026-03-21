from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

__all__ = ["distance_transform_edt"]


def distance_transform_edt(
    input: torch.Tensor,
    sampling: Optional[Union[float, Sequence[float]]] = None,
) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)!r}")

    ndim = input.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Only 2-D and 3-D inputs are supported, got ndim={ndim}")

    if sampling is None:
        sampling = [1.0] * ndim
    elif isinstance(sampling, (int, float)):
        sampling = [float(sampling)] * ndim
    else:
        sampling = [float(s) for s in sampling]

    if len(sampling) != ndim:
        raise ValueError(
            f"len(sampling)={len(sampling)} does not match input.ndim={ndim}"
        )

    sq = torch.where(
        input == 0,
        input.new_zeros(input.shape, dtype=torch.float64),
        input.new_full(input.shape, float("inf"), dtype=torch.float64),
    )

    for axis in range(ndim):
        sp = sampling[axis]
        if axis == 0:
            sq = _edt_axis_binary(sq, axis, sp)
        else:
            sq = _edt_axis(sq, axis, sp)

    return sq.sqrt().to(torch.float32)


def _edt_axis_binary(sq: torch.Tensor, axis: int, sp: float) -> torch.Tensor:
    sq = sq.movedim(axis, -1)
    shape = sq.shape
    L = shape[-1]
    M = sq.numel() // L
    g = sq.reshape(M, L)

    device = g.device
    sp2 = sp * sp
    INF = float("inf")

    q = torch.arange(L, dtype=torch.float64, device=device)

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


def _edt_axis(sq: torch.Tensor, axis: int, sp: float) -> torch.Tensor:
    sq = sq.movedim(axis, -1).contiguous()
    shape = sq.shape
    L = shape[-1]
    M = sq.numel() // L

    out = _meijster_compiled(sq.reshape(M, L), sp)
    return out.reshape(shape).movedim(-1, axis)


def _meijster(g: torch.Tensor, sp: float) -> torch.Tensor:
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

    active_cols: list[int] = (
        torch.isfinite(g).any(dim=0).nonzero(as_tuple=True)[0].tolist()
    )

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


try:
    _compiled = torch.compile(_meijster)

    def _meijster_compiled(g: "torch.Tensor", sp: float) -> "torch.Tensor":
        global _meijster_compiled, _compiled
        try:
            return _compiled(g, sp)
        except Exception:
            _meijster_compiled = _meijster
            return _meijster(g, sp)

except Exception:
    _meijster_compiled = _meijster
