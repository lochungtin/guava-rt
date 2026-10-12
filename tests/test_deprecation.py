"""The camelCase API of guava-rt <= 0.5 still works, with a DeprecationWarning."""

import pytest
import torch

import guava_rt
from guava_rt import Mask, Metrics, Region
from helpers import LABELS, organs

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


@pytest.fixture
def metrics() -> Metrics:
    x = organs(2)
    return Metrics(A=x, B=x, labels=LABELS, target="target", anchor="spine")


def _same(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)  # type: ignore[arg-type]
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)  # type: ignore[attr-defined,index]
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(_same, a, b))  # type: ignore[arg-type]
    if isinstance(a, Region | Mask):
        return True
    return a == b


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("getVolDiff", "get_vol_diff"),
        ("getSADiff", "get_sa_diff"),
        ("getROIDisplacementDiff", "get_roi_displacement_diff"),
        ("alignA2B", "align_a_to_b"),
        ("getBSDDiff", "get_bsd_diff"),
        ("getSeparationDistanceDiff", "get_separation_distance_diff"),
        ("getPercentageOverlapDiff", "get_percentage_overlap_diff"),
    ],
)
def test_metrics_methods(metrics: Metrics, old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated; use {new}"):
        result = getattr(metrics, old)()
    assert _same(result, getattr(metrics, new)())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("getSeparationDistances", "get_separation_distances"),
        ("getThresholdedOverlapPercentages", "get_thresholded_overlap_percentages"),
    ],
)
def test_region_methods(metrics: Metrics, old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning):
        result = getattr(metrics.A, old)(mode="surface")
    assert _same(result, getattr(metrics.A, new)(mode="surface"))


def test_renamed_keyword_arguments(metrics: Metrics) -> None:
    r = metrics.A
    with pytest.warns(DeprecationWarning) as record:
        old = r.getDisplacementVectors(useAnchor=True, useLabels=False)
    messages = {str(w.message).split(" is deprecated")[0] for w in record}
    assert messages == {
        "getDisplacementVectors",
        "keyword argument 'useAnchor'",
        "keyword argument 'useLabels'",
    }
    assert _same(old, r.get_displacement_vectors(use_anchor=True, use_labels=False))

    a, b = r.masks[1], r.masks[2]
    with pytest.warns(DeprecationWarning):
        old_face = b.getRCVS(raySource=a, cutAwayDist=8, N=16, chunk_size=None)
    assert torch.equal(old_face.mask, b.get_rcvs(a, 8, 16, None).mask)

    with pytest.raises(TypeError, match="got both"):
        b.get_rcvs(ray_source=a, raySource=a)  # type: ignore[call-arg]


def test_mask_methods_and_attributes(metrics: Metrics) -> None:
    a, b = metrics.A.masks[1], metrics.B.masks[2]
    with pytest.warns(DeprecationWarning):
        assert a.SHAPE == a.shape
    with pytest.warns(DeprecationWarning):
        assert _same(a.getBSD(b), a.get_bsd(b))
    with pytest.warns(DeprecationWarning):
        assert torch.equal(a.alignTo(b).mask, a.align_to(b).mask)
    with pytest.warns(DeprecationWarning):
        assert _same(a.maskDMap(b), a.mask_dmap(b))
    with pytest.warns(DeprecationWarning):
        assert torch.equal(a.getSurfaceCoordinates(), a.get_surface_coordinates())
    with pytest.warns(DeprecationWarning):
        assert a.getVolDiff(b) == a.get_vol_diff(b)
    with pytest.warns(DeprecationWarning):
        assert a.getSADiff(b) == a.get_sa_diff(b)
    with pytest.warns(DeprecationWarning):
        old = a.getMaskCoordinates()  # old return type: a torch.where tuple
    assert torch.equal(torch.stack(old, dim=1), a.get_mask_coordinates())


def test_use_labels_attribute(metrics: Metrics) -> None:
    for obj in (metrics, metrics.A):
        with pytest.warns(DeprecationWarning):
            assert obj.useLabels is True


def test_utils(capsys) -> None:  # type: ignore[no-untyped-def]
    x = torch.arange(10.0)
    with pytest.warns(DeprecationWarning):
        assert torch.equal(guava_rt.seriesAnalysis(x), guava_rt.series_analysis(x))
    with pytest.warns(DeprecationWarning):
        guava_rt.prettyPrintTable(rowNames=["r"], rows=[[1]], labels=["c"], minWidth=3)
    assert "| c   |" in capsys.readouterr().out


def test_aliases_not_exported() -> None:
    assert "seriesAnalysis" not in guava_rt.__all__
