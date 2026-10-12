import numpy as np
import pytest
import torch

from guava_rt import Mask, Region
from helpers import LABELS, box, organs, shape_for


def make(ndim: int, **kwargs: object) -> Region:
    kwargs = {"target": "target", "anchor": "spine", "labels": LABELS} | kwargs
    return Region(*organs(ndim), **kwargs)  # type: ignore[arg-type]


def test_target_and_anchor_by_label_or_index(ndim: int) -> None:
    by_label = make(ndim)
    by_index = make(ndim, target=1, anchor=0)
    unlabelled = make(ndim, target=1, anchor=0, labels=None)
    for r in (by_label, by_index, unlabelled):
        assert r.target_idx == 1
        assert r.target_mask is r.masks[1]
        assert torch.equal(r.anchor, r.masks[0].center_of_mass)
        assert torch.equal(r.target_dmap, r.masks[1].dmap())


def test_anchor_as_coordinates(ndim: int) -> None:
    point = [3.0] * ndim
    for anchor in (np.array(point), torch.tensor(point, dtype=torch.float64)):
        r = make(ndim, anchor=anchor)
        assert r.anchor.dtype == torch.float32
        assert torch.equal(r.anchor, torch.tensor(point))


def test_label_lookup_without_labels_fails(ndim: int) -> None:
    with pytest.raises(ValueError):
        make(ndim, labels=None)


def test_accepts_masks_and_arrays(ndim: int) -> None:
    x = organs(ndim)
    r = Region(Mask(x[0]), x[1].numpy(), x[2], x[3], target=1, anchor=0)
    assert all(isinstance(m, Mask) for m in r.masks)
    assert len(r) == 4


def test_getitem(ndim: int) -> None:
    r = make(ndim)
    assert r["near"] is r.masks[2]
    assert r[3] is r.masks[3]
    assert r["missing"] is None
    assert make(ndim, target=1, anchor=0, labels=None)["near"] is None


def test_results_keyed_by_label(ndim: int) -> None:
    r = make(ndim)
    assert list(r.get_displacement_vectors()) == LABELS
    assert isinstance(r.get_displacement_vectors(use_labels=False), list)
    non_target = ["spine", "near", "far"]
    assert list(r.get_separation_distances()) == non_target
    assert list(r.get_thresholded_overlap_percentages()) == non_target
    unlabelled = make(ndim, target=1, anchor=0, labels=None)
    assert len(unlabelled.get_separation_distances()) == 3


def test_displacement_vectors(ndim: int) -> None:
    r = make(ndim)
    com = [m.center_of_mass for m in r.masks]
    to_target = r.get_displacement_vectors(use_labels=False)
    to_anchor = r.get_displacement_vectors(use_anchor=True, use_labels=False)
    for i in range(4):
        assert torch.allclose(to_target[i], com[i] - com[1])
        assert torch.allclose(to_anchor[i], com[i] - com[0])
    assert torch.allclose(to_target[1], torch.zeros(ndim))


def test_separation_distances_between_boxes(ndim: int) -> None:
    """Two boxes ``gap`` voxels apart along axis 0: distances run gap+1 .. gap+depth."""
    shape = shape_for(ndim)
    target = box(shape, [4] * ndim, [10] * ndim)
    oar_lo, oar_hi = [14] + [4] * (ndim - 1), [17] + [10] * (ndim - 1)
    r = Region(target, box(shape, oar_lo, oar_hi), target=0, anchor=0)
    (d,) = r.get_separation_distances(mode="volume")
    assert d.numel() == 3 * 6 ** (ndim - 1)
    assert sorted(set(d.tolist())) == [5.0, 6.0, 7.0]
    (s,) = r.get_separation_distances(mode="surface")
    assert set(s.tolist()) <= {5.0, 6.0, 7.0}


def test_separation_distances_details(ndim: int) -> None:
    r = make(ndim)
    for mode in ("volume", "surface"):
        for d, (values, masked, part) in zip(
            r.get_separation_distances(mode=mode).values(),
            r.get_separation_distances(mode=mode, distances_only=False).values(),
            strict=True,
        ):
            assert torch.equal(d, values)
            assert torch.equal(values, r.target_dmap[part])
            assert torch.equal(masked, r.target_dmap * part.float())
    surf = r.get_separation_distances(mode="surface", distances_only=False)["near"][2]
    assert torch.equal(surf, r["near"].surface())  # type: ignore[union-attr]


def test_separation_distances_rcvs_is_subset_of_surface() -> None:
    shape = (48, 48)
    target = box(shape, (16, 4), (32, 12))
    oar = box(shape, (14, 26), (34, 34))
    r = Region(target, oar, target=0, anchor=0)
    (_, _, part) = r.get_separation_distances(mode="rcvs", distances_only=False)[0]
    assert part.any()
    assert (part <= Mask(oar).surface()).all()


def test_thresholded_overlap_curve(ndim: int) -> None:
    r = make(ndim)
    for mode in ("volume", "surface"):
        details = r.get_thresholded_overlap_percentages(mode=mode, percentages_only=False)
        for name, (percs, full, bins, d_vals, _, part) in details.items():  # type: ignore[union-attr]
            assert torch.all(percs[1:] >= percs[:-1]), name
            assert percs[-1] == 1.0
            d_min, d_max = int(d_vals.min().ceil()), int(d_vals.max().ceil())
            assert len(percs) == d_max - d_min + 1
            assert torch.allclose(full[d_min : d_max + 1], percs)
            assert full[d_max:].eq(1).all()
            assert bins[d_max] == part.sum()


def test_overlap_of_boxes(ndim: int) -> None:
    shape = shape_for(ndim)
    target = box(shape, [4] * ndim, [10] * ndim)
    oar_lo, oar_hi = [14] + [4] * (ndim - 1), [17] + [10] * (ndim - 1)
    r = Region(target, box(shape, oar_lo, oar_hi), target=0, anchor=0)
    (percs,) = r.get_thresholded_overlap_percentages()
    assert torch.allclose(percs, torch.tensor([1 / 3, 2 / 3, 1.0]))


def test_overlap_of_empty_mask(ndim: int) -> None:
    """A structure with no voxels in the chosen mode gets an empty curve, not an error."""
    x = organs(ndim)
    r = Region(x[1], torch.zeros_like(x[1]), target=0, anchor=0)
    ((percs, full, bins, d_vals, _, _),) = r.get_thresholded_overlap_percentages(
        percentages_only=False
    )
    assert percs.numel() == d_vals.numel() == 0
    assert full.eq(0).all() and bins.eq(0).all()
    assert len(full) == int(r.target_dmap.max().ceil()) + 1
