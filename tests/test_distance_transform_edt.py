import numpy as np
import pytest
import torch
from scipy import ndimage

from guava_rt import distance_transform_edt
from helpers import ball, shape_for


def test_matches_scipy_on_random_input(ndim: int) -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(shape_for(ndim), generator=g) > 0.97
    out = distance_transform_edt(~x)
    ref = ndimage.distance_transform_edt(~x.numpy())
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-5)


def test_matches_scipy_on_ball(ndim: int) -> None:
    shape = shape_for(ndim)
    x = ball(shape, [n / 2 for n in shape], 7)
    np.testing.assert_allclose(
        distance_transform_edt(x).numpy(), ndimage.distance_transform_edt(x.numpy()), atol=1e-5
    )


@pytest.mark.parametrize("sampling", [2.5, (3.0, 1.0, 0.5)])
def test_anisotropic_sampling(ndim: int, sampling: float | tuple[float, ...]) -> None:
    if isinstance(sampling, tuple):
        sampling = sampling[:ndim]
    g = torch.Generator().manual_seed(1)
    x = torch.rand(shape_for(ndim), generator=g) > 0.98
    out = distance_transform_edt(~x, sampling=sampling)
    ref = ndimage.distance_transform_edt(~x.numpy(), sampling=sampling)
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)


def test_single_background_voxel_gives_euclidean_norm(ndim: int) -> None:
    shape = shape_for(ndim)
    x = torch.ones(shape, dtype=torch.bool)
    x[(0,) * ndim] = False
    out = distance_transform_edt(x)
    idx = torch.stack(torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij"))
    assert torch.allclose(out, idx.float().norm(dim=0))


def test_without_background_everything_is_inf(ndim: int) -> None:
    out = distance_transform_edt(torch.ones(shape_for(ndim), dtype=torch.bool))
    assert torch.isinf(out).all()


def test_all_background_is_zero(ndim: int) -> None:
    out = distance_transform_edt(torch.zeros(shape_for(ndim), dtype=torch.bool))
    assert torch.equal(out, torch.zeros(shape_for(ndim)))


def test_accepts_non_bool_input() -> None:
    x = torch.zeros(10, 10)
    x[2:5, 2:5] = 3.0
    assert torch.equal(distance_transform_edt(x), distance_transform_edt(x != 0))


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((np.zeros((4, 4)),), {}, TypeError),
        ((torch.zeros(4),), {}, ValueError),
        ((torch.zeros(2, 2, 2, 2),), {}, ValueError),
        ((torch.zeros(4, 4),), {"sampling": (1.0, 1.0, 1.0)}, ValueError),
    ],
)
def test_invalid_input(args: tuple[object, ...], kwargs: dict[str, object], error: type) -> None:
    with pytest.raises(error):
        distance_transform_edt(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.gpu
def test_cuda_matches_cpu(ndim: int) -> None:
    g = torch.Generator().manual_seed(2)
    x = torch.rand(shape_for(ndim), generator=g) > 0.97
    out = distance_transform_edt(~x.cuda())
    assert out.is_cuda
    assert torch.allclose(out.cpu(), distance_transform_edt(~x))
