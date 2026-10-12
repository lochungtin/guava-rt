import numpy as np
import pytest
import torch

from guava_rt import SERIES_ANALYSIS_LABELS, Metrics, Region
from helpers import LABELS, organs, shifted


def make(a: list[torch.Tensor], b: list[torch.Tensor], **kwargs: object) -> Metrics:
    kwargs = {"target": "target", "anchor": "spine", "labels": LABELS} | kwargs
    return Metrics(A=a, B=b, **kwargs)  # type: ignore[arg-type]


def moved(masks: list[torch.Tensor], shift: list[int]) -> list[torch.Tensor]:
    return [shifted(m, shift) for m in masks]


def test_identical_regions_have_no_difference(ndim: int) -> None:
    x = organs(ndim)
    m = make(x, x)
    for diff in m.get_vol_diff().values():  # type: ignore[union-attr]
        assert diff[0] == 0 and diff[1] == 0
    for diff in m.get_sa_diff().values():  # type: ignore[union-attr]
        assert diff[0] == 0
    for v in m.get_roi_displacement_diff().values():  # type: ignore[union-attr]
        assert v.abs().max() == 0
    for s in m.get_bsd_diff().values():  # type: ignore[union-attr]
        assert s.abs().max() == 0
    for mask in ("volume", "surface"):
        for d in m.get_separation_distance_diff(mask=mask).values():  # type: ignore[union-attr]
            assert d[0].abs().max() == 0
            assert torch.equal(d[1], d[2])
        for p in m.get_percentage_overlap_diff(mode=mask).values():  # type: ignore[union-attr]
            assert p[0].abs().max() == 0


def test_rigid_translation_is_removed_by_the_anchor(ndim: int) -> None:
    """Moving the whole region shifts every COM equally, so anchored displacements cancel."""
    x = organs(ndim)
    m = make(x, moved(x, [1, -2, 1][:ndim]))
    for v in m.get_roi_displacement_diff().values():  # type: ignore[union-attr]
        assert torch.allclose(v, torch.zeros(ndim))
    for diff in m.get_vol_diff().values():  # type: ignore[union-attr]
        assert diff[0] == 0
    for s in m.get_bsd_diff().values():  # type: ignore[union-attr]
        assert s.abs().max() == 0


def test_roi_displacement_of_one_moved_structure(ndim: int) -> None:
    x = organs(ndim)
    shift = [0, 2, -1][:ndim]
    y = list(x)
    y[2] = shifted(x[2], shift)
    disp = make(x, y).get_roi_displacement_diff()
    assert torch.allclose(disp["near"], -torch.tensor(shift, dtype=torch.float32))  # type: ignore[call-overload]
    for name in ("spine", "target", "far"):
        assert torch.allclose(disp[name], torch.zeros(ndim))  # type: ignore[call-overload]


def test_vol_diff_is_a_minus_b(ndim: int) -> None:
    x, y = organs(ndim), organs(ndim)
    y[3] = torch.zeros_like(y[3])
    y[3][(slice(0, 2),) * ndim] = True
    va, vb = int(x[3].sum()), 2**ndim
    assert make(x, y).get_vol_diff()["far"] == (va - vb, (va - vb) / vb, va, vb)  # type: ignore[call-overload]


def test_bsd_modes_index_the_full_statistics(ndim: int) -> None:
    x = organs(ndim)
    y = list(x)
    y[1] = organs(ndim)[2]  # different shape for the target
    m = make(x, y)
    full = m.get_bsd_diff("all")
    assert full["target"].shape == (len(SERIES_ANALYSIS_LABELS),)  # type: ignore[call-overload]
    for mode, label in (("asd", "AVG"), ("hd", "MAX"), ("HD95", "P95")):
        i = SERIES_ANALYSIS_LABELS.index(label)
        for name, value in m.get_bsd_diff(mode).items():  # type: ignore[union-attr]
            assert value == full[name][i]  # type: ignore[call-overload]


def test_separation_distance_statistics(ndim: int) -> None:
    x = organs(ndim)
    y = moved(x, [0, 1, 0][:ndim])
    y[1] = x[1]  # keep the target in place so OARs move relative to it
    m = make(x, y)
    raw_a = m.A.get_separation_distances()
    raw_b = m.B.get_separation_distances()
    stats = {
        "min": lambda t: t.min(),
        "max": lambda t: t.max(),
        "mean": lambda t: t.mean(),
        "median": lambda t: t.median(),
        "p5": lambda t: torch.quantile(t, 0.05),
        "p95": lambda t: torch.quantile(t, 0.95),
    }
    for mode, f in stats.items():
        for name, (diff, a, b) in m.get_separation_distance_diff(mode=mode).items():  # type: ignore[union-attr]
            assert torch.allclose(a, f(raw_a[name]))  # type: ignore[call-overload]
            assert torch.allclose(b, f(raw_b[name]))  # type: ignore[call-overload]
            assert torch.allclose(diff, a - b)
    out = m.get_separation_distance_diff(mode="all", distances_only=False)
    assert len(out["near"]) == 3 + 2 * 3  # type: ignore[call-overload]


def test_percentage_overlap_details(ndim: int) -> None:
    x = organs(ndim)
    out = make(x, x).get_percentage_overlap_diff(percentages_only=False)
    assert list(out) == ["spine", "near", "far"]
    assert len(out["near"]) == 3 + 2 * 6  # type: ignore[call-overload]


def test_align_a_to_b_moves_the_anchor_onto_b(ndim: int) -> None:
    x = organs(ndim)
    shift = [2, -1, 1][:ndim]
    m = make(x, moved(x, shift))
    a2, b = m.align_a_to_b()
    assert b is m.B
    assert torch.allclose(a2.anchor, m.B.anchor)
    assert torch.allclose(a2.anchor, a2["spine"].center_of_mass)  # type: ignore[union-attr]
    for ma, mb in zip(a2.masks, m.B.masks, strict=True):
        assert torch.equal(ma.mask, mb.mask)


def test_inputs_regions_and_stacked_arrays(ndim: int) -> None:
    x = organs(ndim)
    ref = make(x, x).get_vol_diff()
    stacked = torch.stack(x)
    for a, b in (
        (stacked, stacked.numpy()),
        (Region(*x, target="target", anchor="spine", labels=LABELS), x),
    ):
        assert make(a, b).get_vol_diff() == ref  # type: ignore[arg-type]


def test_per_side_target_and_anchor(ndim: int) -> None:
    x = organs(ndim)
    m = Metrics(A=x, B=x, target_A=1, target_B=1, anchor_A=0, anchor_B=np.zeros(ndim))
    assert m.A.target_idx == m.B.target_idx == 1
    assert torch.equal(m.B.anchor, torch.zeros(ndim))
    assert isinstance(m.get_vol_diff(), list)


def test_missing_target_or_anchor(ndim: int) -> None:
    x = organs(ndim)
    with pytest.raises(ValueError):
        Metrics(A=x, B=x, anchor=0)
    with pytest.raises(ValueError):
        Metrics(A=x, B=x, target=1)


def test_mismatched_structure_counts(ndim: int) -> None:
    x = organs(ndim)
    m = Metrics(A=x, B=x[:3], target=1, anchor=0)
    with pytest.raises(ValueError):
        m.get_vol_diff()


@pytest.mark.gpu
def test_cuda_matches_cpu(ndim: int) -> None:
    x = organs(ndim)
    y = moved(x, [1, 0, -1][:ndim])
    cpu, gpu = make(x, y), make(x, y, dev="cuda")
    assert gpu.A.masks[0].mask.is_cuda
    for name, s in gpu.get_bsd_diff().items():  # type: ignore[union-attr]
        assert torch.allclose(s.cpu(), cpu.get_bsd_diff()[name])  # type: ignore[call-overload]
    for name, d in gpu.get_separation_distance_diff().items():  # type: ignore[union-attr]
        assert torch.allclose(d[1].cpu(), cpu.get_separation_distance_diff()[name][1])  # type: ignore[call-overload]
