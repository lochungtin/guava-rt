import numpy as np
import pytest
import torch
from scipy import ndimage

from guava_rt import Mask
from helpers import ball, box, shape_for, shifted


def test_volume_and_center_of_mass(ndim: int) -> None:
    shape = shape_for(ndim)
    m = Mask(box(shape, [4] * ndim, [10] * ndim))
    assert m.volume.item() == 6**ndim
    assert m.volume.dtype == torch.int32
    assert torch.allclose(m.center_of_mass, torch.full((ndim,), 6.5))
    assert m.shape == shape


def test_accepts_numpy_and_non_bool_input(ndim: int) -> None:
    x = box(shape_for(ndim), [3] * ndim, [7] * ndim)
    a, b = Mask(x.numpy()), Mask(x.to(torch.uint8) * 5)
    assert a.mask.dtype == torch.bool
    assert torch.equal(a.mask, b.mask)
    assert np.array_equal(a.numpy(), x.numpy())


def test_surface_of_box(ndim: int) -> None:
    """A box's surface is its outer shell: the box minus its interior."""
    shape = shape_for(ndim)
    m = Mask(box(shape, [4] * ndim, [12] * ndim))
    interior = box(shape, [5] * ndim, [11] * ndim)
    assert torch.equal(m.surface(), m.mask & ~interior)
    assert m.surface_area().item() == 8**ndim - 6**ndim


def test_surface_matches_scipy_erosion(ndim: int) -> None:
    """The structuring element is the union of the central axis planes (18-/4-neighbourhood)."""
    shape = shape_for(ndim)
    x = ball(shape, [n / 2 for n in shape], 9)
    s = np.zeros((3,) * ndim, dtype=bool)
    for dim in range(ndim):
        idx = [slice(None)] * ndim
        idx[dim] = 1
        s[tuple(idx)] = True
    ref = x.numpy() & ~ndimage.binary_erosion(x.numpy(), structure=s)
    assert np.array_equal(Mask(x).surface().numpy(), ref)


def test_single_voxel_is_all_surface(ndim: int) -> None:
    x = torch.zeros(shape_for(ndim), dtype=torch.bool)
    x[(5,) * ndim] = True
    assert Mask(x).surface_area().item() == 1


def test_dmap_and_sdmap(ndim: int) -> None:
    shape = shape_for(ndim)
    x = ball(shape, [n / 2 for n in shape], 8)
    m = Mask(x)
    np.testing.assert_allclose(
        m.dmap().numpy(), ndimage.distance_transform_edt(~x.numpy()), atol=1e-5
    )
    assert (m.dmap()[x] == 0).all()
    assert (m.sdmap()[m.surface()] == 0).all()
    assert (m.sdmap()[x & ~m.surface()] > 0).all()


def test_derived_quantities_are_cached(ndim: int) -> None:
    m = Mask(box(shape_for(ndim), [2] * ndim, [6] * ndim))
    for f in (m.surface, m.surface_area, m.dmap, m.sdmap):
        assert f() is f()


def test_vol_and_sa_diff(ndim: int) -> None:
    shape = shape_for(ndim)
    a = Mask(box(shape, [0] * ndim, [4] * ndim))
    b = Mask(box(shape, [0] * ndim, [2] * ndim))
    va, vb = 4**ndim, 2**ndim
    assert a.get_vol_diff(b) == (va - vb, (va - vb) / vb, va, vb)
    sa, sb = 4**ndim - 2**ndim, 2**ndim
    assert a.get_sa_diff(b) == (sa - sb, (sa - sb) / sb, sa, sb)


def test_mask_dmap(ndim: int) -> None:
    shape = shape_for(ndim)
    a = Mask(box(shape, [0] * ndim, [4] * ndim))
    other = box(shape, [10] * ndim, [12] * ndim)
    for arg in (Mask(other), other, other.numpy()):
        values, masked = a.mask_dmap(arg)
        assert torch.equal(values, a.dmap()[other])
        assert torch.equal(masked, a.dmap() * other.float())


def test_coordinates(ndim: int) -> None:
    m = Mask(box(shape_for(ndim), [1] * ndim, [4] * ndim))
    assert m.get_mask_coordinates().shape == (3**ndim, ndim)
    assert m.get_surface_coordinates().shape == (int(m.surface_area()), ndim)
    assert torch.equal(m.mask[tuple(m.get_surface_coordinates().T)], m.surface()[m.surface()])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_align_to_integer_shift_is_exact(ndim: int, axis: int) -> None:
    """Shift component i must move array axis i (guards the affine_grid x/y/z flip)."""
    if axis >= ndim:
        pytest.skip("axis not present")
    shape = shape_for(ndim)
    x = ball(shape, [n / 2 for n in shape], 5)
    shift = [0] * ndim
    shift[axis] = 3
    out = Mask(x).align_to(shift_vec=torch.tensor(shift, dtype=torch.float32))
    assert torch.equal(out.mask, shifted(x, shift))
    out_np = Mask(x).align_to(shift_vec=np.array(shift, dtype=float))
    assert torch.equal(out_np.mask, out.mask)


def test_align_to_target_matches_center_of_mass(ndim: int) -> None:
    shape = shape_for(ndim)
    a = Mask(ball(shape, [10] * ndim, 4))
    b = Mask(ball(shape, [14, 17, 12][:ndim], 4))
    out = a.align_to(b)
    assert torch.allclose(out.center_of_mass, b.center_of_mass)
    assert torch.equal(out.mask, b.mask)


def test_align_to_needs_target_or_shift() -> None:
    with pytest.raises(ValueError):
        Mask(torch.ones(4, 4)).align_to()


def test_bsd_of_translated_copy_is_zero(ndim: int) -> None:
    """BSD aligns centres of mass first, so a pure translation gives no discrepancy."""
    shape = shape_for(ndim)
    x = ball(shape, [n / 2 for n in shape], 6)
    bsd, st_d, ts_d = Mask(x).get_bsd(Mask(shifted(x, [2, -3, 1][:ndim])))
    assert bsd.abs().max() == 0
    assert len(st_d) == len(ts_d) == int(Mask(x).surface_area())


def test_bsd_is_union_of_both_directions(ndim: int) -> None:
    shape = shape_for(ndim)
    c = [n / 2 for n in shape]
    small, large = Mask(ball(shape, c, 4)), Mask(ball(shape, c, 8))
    bsd, st_d, ts_d = small.get_bsd(large)
    assert torch.equal(bsd, torch.cat([st_d, ts_d]))
    assert len(st_d) == int(large.surface_area())
    assert len(ts_d) == int(small.surface_area())
    # concentric spheres: every distance is about the difference of the radii
    assert (bsd - 4).abs().max() <= 1.5


def test_rcvs_sees_only_the_facing_side() -> None:
    """A box beside the target: only the face towards the target is visible."""
    shape = (40, 40)
    target = Mask(ball(shape, (20, 8), 4))
    oar = Mask(box(shape, (14, 26), (27, 34)))
    face = oar.get_rcvs(target, chunk_size=None)
    assert face.volume > 0
    assert (face.mask <= oar.surface()).all()
    assert not face.mask[:, 30:].any()  # far half of the box is hidden behind itself
    assert face.mask[:, 26].any()


def test_rcvs_chunking_does_not_change_result() -> None:
    shape = (40, 40)
    target = Mask(ball(shape, (20, 8), 4))
    oar = Mask(box(shape, (14, 26), (27, 34)))
    full = oar.get_rcvs(target, chunk_size=None).mask
    for chunk_size in (1, 7, 10_000):
        assert torch.equal(oar.get_rcvs(target, chunk_size=chunk_size).mask, full)


def test_rcvs_occluded_by_another_part_of_itself() -> None:
    """The inner arm of a U-shaped OAR hides the outer arm from the target."""
    shape = (40, 60)
    target = Mask(ball(shape, (20, 6), 4))
    inner, outer, base = (
        box(shape, (8, 24), (32, 27)),
        box(shape, (8, 44), (32, 47)),
        box(shape, (8, 24), (11, 47)),
    )
    face = Mask(inner | outer | base).get_rcvs(target, chunk_size=None)
    assert face.mask[12:28, 24].any()
    assert not face.mask[12:28, 44:].any()


@pytest.mark.parametrize("gap", [0, 1, 2, 5])
def test_rcvs_of_nearby_structure(gap: int) -> None:
    """Short rays see the facing side: samples in a ray's own end voxels don't block it."""
    shape = (40, 40)
    target = Mask(box(shape, (10, 10), (30, 20)))
    oar = Mask(box(shape, (10, 20 + gap), (30, 24 + gap)))
    face = oar.get_rcvs(target, chunk_size=None).mask
    assert face[10:30, 20 + gap].all()
    assert not face[11:29, 23 + gap].any()  # far side stays hidden behind the OAR itself


def test_rcvs_of_nearby_structure_3d() -> None:
    shape = (24, 24, 24)
    target = Mask(box(shape, (6, 6, 4), (18, 18, 10)))
    oar = Mask(box(shape, (6, 6, 11), (18, 18, 14)))
    face = oar.get_rcvs(target, chunk_size=None).mask
    assert face[6:18, 6:18, 11].all()
    assert not face[7:17, 7:17, 13].any()


@pytest.mark.gpu
def test_cuda_matches_cpu(ndim: int) -> None:
    shape = shape_for(ndim)
    a = ball(shape, [n / 2 for n in shape], 6)
    b = ball(shape, [n / 2 + 1 for n in shape], 5)
    cpu_bsd = Mask(a).get_bsd(Mask(b))[0]
    gpu = Mask(a, "cuda")
    assert gpu.mask.is_cuda
    assert torch.allclose(gpu.get_bsd(Mask(b, "cuda"))[0].cpu(), cpu_bsd)
    assert torch.equal(gpu.surface().cpu(), Mask(a).surface())
