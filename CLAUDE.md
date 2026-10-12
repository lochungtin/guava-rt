# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

guava-rt (`guava_rt`) is a PyTorch library of geometric metrics for comparing binary segmentation masks (2-D or 3-D). It was built to compare upright and supine radiotherapy anatomy. The README is the user-facing maths spec for every metric, so keep it in sync when behaviour changes.

## Commands

- `uv venv && uv pip install --torch-backend=cpu -e ".[dev]"`: install with CPU torch (pytest, ruff, mypy, plus scipy as a test reference). `uv sync --extra dev` also works but pulls CUDA torch.
- `uv run pytest`: all tests; `uv run pytest tests/test_mask.py::test_bsd_is_union_of_both_directions`: a single test.
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict, `src` only). Check `.`, not just `src tests`. Markdown is excluded from formatting in `pyproject.toml`.
- `uv build`: wheel + sdist into `dist/`.
- **Don't run tests (or anything else that uses the GPU) without the user's go-ahead**: the GPU is often busy with other jobs. `CUDA_VISIBLE_DEVICES=` forces CPU, and the `gpu`-marked tests then skip. `slow`-marked tests only run with `GUAVA_SLOW=1`.

## Release / CI

The project setup, including `.github/workflows`, mirrors `../pyNiftiReg`.

- **Versioning**: the version comes from git tags via hatch-vcs (`no-guess-dev`: untagged commits build as `X.Y.Z.postN.devM`). Never hard-code it in `pyproject.toml`.
- **Commits must be conventional**: Commitizen derives the version bump and `CHANGELOG.md` from them. `fix:` releases a patch and `feat:` a minor version, each publishing to PyPI. `docs:`, `ci:`, `chore:`, `test:` and `refactor:` don't release, so don't use `fix:` for non-code changes. A push with no release-worthy commits makes the auto-bump job fail with "no commits to bump"; that's expected, not a CI bug.
- **The bot pushes to the default branch**: after a release, `cz bump` adds a bump commit and a changelog commit. Run `git pull --ff-only` before committing.
- **Don't edit `[tool.commitizen] version` or `CHANGELOG.md` by hand**: `cz bump` maintains both.
- `branch-ci.yml` (push to `main`/`master`): commit lint and tests (CPU torch, Python 3.10–3.13; ruff and mypy on 3.12), build, publish to TestPyPI, then `cz bump` tags `vX.Y.Z` and pushes. It uses `cancel-in-progress`, so don't push while a release run is still going. `release.yml` (on a `v*.*.*` tag): publish to PyPI and create a GitHub Release.

## Architecture

Three layers, each wrapping the one below:

1. **`Mask`** (`mask.py`): one binary mask on one device.
   - `volume` and `center_of_mass` are computed eagerly. `surface()`, `surface_area()`, `dmap()` (EDT of `~mask`) and `sdmap()` (EDT of `~surface`) are computed lazily and cached.
   - The surface is the mask minus its erosion by the union of the central axis planes. In 3-D that is an 18-neighbourhood, not the 6-connected cross.
   - `align_to` translates by a centre-of-mass difference or an explicit `shift_vec`, using `affine_grid` + nearest `grid_sample`.
   - `get_rcvs` ray-casts from the source mask's surface to this mask's surface, chunked over source points (`chunk_size`). Samples are rounded to voxels, and samples in a ray's own end voxels never block it. Without that rule, every surface within about `(n_steps + 1) / 4` voxels of the target came out hidden (the bug fixed for 1.0).
2. **`Region`** (`region.py`): an ordered list of masks for one patient position.
   - It has a `target` (index or label) and an `anchor` (a mask's centre of mass, or explicit coordinates).
   - It computes per-OAR quantities relative to the target. Results exclude the target and are dicts keyed by label when `labels` was given, otherwise lists (`Region._keyed`).
3. **`Metrics`** (`metrics.py`): pairs Regions A and B. Every difference is `A - B`.
   - Structures are zipped by position (`strict=True`), so both sides must list them in the same order. Results are keyed by A's labels.

Supporting modules:

- `distance_transform_edt.py`: a PyTorch replacement for SciPy's EDT (tested against it). It uses a cummax/cummin pass on axis 0 and Meijster on the other axes. `_meijster` is wrapped in `torch.compile` and falls back to eager permanently on any failure.
- `utils.series_analysis`: the 8-stat vector `SERIES_ANALYSIS_LABELS`. BSD modes index into it by position (ASD `[3]`, HD95 `[-2]`, HD `[-1]`).
- `_deprecation.py`: the camelCase API of ≤ 0.5 (`getBSDDiff`, `useAnchor=`, `SHAPE`, …) is kept as warning aliases at the bottom of each class or module. Add an alias when renaming anything public; `tests/test_deprecation.py` checks each one.

## Conventions

- snake_case everywhere, except the `A`/`B` names in `Metrics`, which match the paper's notation.
- Coordinates and shift vectors are in voxels, ordered like the tensor dims (`(z, y, x)` / `(y, x)`). `affine_grid` wants x/y/z order; that flip lives only in `Mask.align_to`.
- `dev` is passed through every constructor (default `"cpu"`).
- Tests: geometry tests take the `ndim` fixture (`tests/conftest.py`) so they run in both 2-D and 3-D. `tests/helpers.py` builds synthetic balls, boxes and a four-organ region.
