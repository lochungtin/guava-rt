# guava-rt

**guava-rt** is a PyTorch-native library implementing a standardised **G**eometric **U**pright-supine **A**natomical **V**arition **A**nalysis framework for comparing binary segmentation masks. It was developed to quantify anatomical variation between upright and supine patient positions in radiotherapy (RT) research, though its metrics are general-purpose and applicable to any binary mask comparison task in 2-D or 3-D voxel space.

The library provides formal, mathematically reproducible implementations of a suite of geometric metrics — from simple volume and surface area differences through to bidirectional surface discrepancy, separation distance distributions, ray-cast visible surfaces, and discretised distance thresholding. All computation runs natively on PyTorch tensors with CPU and CUDA support.

> **Note:** guava-rt is a geometric analysis framework, not a dose engine. Its metrics serve as spatial heuristics that approximate relationships relevant to treatment planning but do not constitute physical dose calculations.

---

## Features

- Euclidean distance transform (EDT) implemented entirely in PyTorch — no SciPy dependency
- Boundary surface detection via morphological erosion
- Volume and surface area difference metrics
- ROI displacement vectors via rigid centre-of-mass alignment
- Bidirectional Surface Discrepancy (BSD), including ASD, HD, and HD95
- Separation distance distributions (volume, surface, and RCVS modes)
- Ray-cast Visible Surface (RCVS) computation
- Discretised distance thresholding / overlap percentage curves
- Optional `torch.compile` acceleration for the distance transform

## Installation

```bash
pip install guava-rt
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy

## Core Concepts

### `Mask`

A `Mask` wraps a single binary segmentation and lazily computes all derived geometric quantities on first access.

| Property / Method | Mathematical definition | Description |
|---|---|---|
| `volume` | `V = \|X\|` | Voxel count of the mask |
| `center_of_mass` | `C = (1/V) Σ v` | Arithmetic mean of coordinates per dimension |
| `surface()` | `S = X ∩ ¬Erode(X)` | Boundary voxels via morphological erosion |
| `surface_area()` | `\|S\|` | Count of boundary voxels |
| `dmap()` | `d(v) = min_{w∈X} ‖v−w‖₂` | Distance from every voxel to the nearest mask voxel |
| `sdmap()` | EDT of `¬S` | Distance map computed from the surface rather than the full mask |

### `Region`

A `Region` holds a collection of `Mask` objects representing multiple anatomical structures (organs at risk, OARs) within one study. One mask is designated the **target** (e.g. the tumour or reference organ receiving radiation) and one point is the **anchor** (used as the reference for displacement vector computation). In clinical practice the anchor is typically derived from a bony structure, which is stable across positions and suitable as a rigid alignment reference.

### `Metrics`

`Metrics` takes two `Region` objects (A and B — e.g. upright and supine) and computes pairwise geometric differences between their corresponding masks.

---

## Usage

### Basic setup

```python
import numpy as np
from mask import Mask
from region import Region
from metrics import Metrics

# Binary numpy arrays, shape (D, H, W) for 3-D or (H, W) for 2-D
labels = ["spine", "prostate", "bladder", "rectum"]

m = Metrics(
    A=[upright_spine, upright_prostate, upright_bladder, upright_rectum],
    B=[supine_spine,  supine_prostate,  supine_bladder,  supine_rectum],
    labels=labels,
    target="prostate",  # the structure receiving radiation
    anchor="spine",     # rigid reference for alignment (bony structure)
    dev="cuda",         # or "cpu"
)
```

---

## Metrics Reference

### Volume and Surface Area

```python
vol_diff = m.getVolDiff()
# {"bladder": (diff, rel_diff, vol_A, vol_B), "rectum": (...), ...}

sa_diff = m.getSADiff()
```

Each entry is a tuple of `(absolute_diff, relative_diff, value_A, value_B)`.

---

### ROI Displacement via Rigid Alignment

```python
disp = m.getROIDisplacementDiff()
# {"bladder": tensor([dz, dy, dx]), ...}
```

This metric computes the relative displacement vector of each ROI between positions A and B after rigidly aligning their coordinate spaces to a common reference anchor.

**How it works.** Given reference anchor COMs `C_uR` and `C_sR`, the base translation vector is:

```
US = C_uR − C_sR
```

Each upright mask is translated by `US` to align the coordinate spaces. The displacement vector for each ROI is then the difference between the translated upright COM and the supine COM:

```
displacement = C'_u − C_s  =  C_u − US − C_s
```

---

### Bidirectional Surface Discrepancy (BSD)

```python
# Full statistics (MIN, P05, P10, AVG, MDN, P90, P95, MAX)
bsd_all = m.getBSDDiff(mode="all")

# Single statistics
asd  = m.getBSDDiff(mode="asd")    # average surface distance
hd   = m.getBSDDiff(mode="hd")     # Hausdorff distance
hd95 = m.getBSDDiff(mode="hd95")   # 95th-percentile Hausdorff distance
```

BSD quantifies the geometric difference between two masks by measuring surface-to-surface distances, pixel by pixel. It is computed on COM-aligned masks (each pair is centre-of-mass aligned before the BSD is computed, isolating local shape differences from global positional offsets).

**How it works.** From the surface distance maps `d_S''_ui` and `d_S_si`, two subsets of distances are constructed by *distance map masking* — evaluating each surface's distance map at the coordinates of the opposing surface:

```
SD_si = { d_S''_ui(v) | v ∈ S_si }     # distance from each supine surface point to the upright surface
SD_ui = { d_S_si(v)   | v ∈ S''_ui }   # distance from each upright surface point to the supine surface
BSD_i = SD_ui ∪ SD_si                  # bidirectional union, sorted ascending
```

The ASD, HD, and HD95 are then derived from this set:

```
ASD   = (1 / |BSD|) × Σ d
HD    = max(BSD)
HD95  = BSD[⌈0.95 × |BSD|⌉]
```

---

### Separation Distance

Measures the distribution of distances from each non-target OAR to the target mask — quantifying how far each surrounding structure sits from the organ receiving radiation, across its full spatial extent rather than just at a single closest point.

```python
# mode controls which part of the OAR is used as the measurement mask
# "volume"  → all voxels of the OAR
# "surface" → only the boundary voxels of the OAR
# "rcvs"    → only the ray-cast visible surface of the OAR (see below)

sep = m.getSeparationDistanceDiff(mask="volume", mode="mean")
sep = m.getSeparationDistanceDiff(mask="surface", mode="all")   # full statistics
```

The `mode` parameter controls how the resulting distance values are summarised: `"min"`, `"max"`, `"mean"`, `"median"`, `"p5"`, `"p95"`, or `"all"` for the full 8-statistic vector `[MIN, P05, P10, AVG, MDN, P90, P95, MAX]`.

#### Ray-cast Visible Surface (RCVS)

When `mask="rcvs"`, the separation distance is computed only over the subset of the OAR surface that has direct line-of-sight to the target — approximating the region most likely to receive radiation.

**How it works.** A ray from surface point `v` (on the target surface `S_T`) to surface point `w` (on the OAR surface `S_i`) is defined as:

```
r_vw(a) = v + a · vw⃗,    a ∈ (0, 1)
```

The ray is sampled at `n` evenly spaced steps. It is unobstructed if none of the sampled intermediate points fall inside the target mask `X_T` or the OAR itself `X_i`. A surface point `w` belongs to the RCVS `S*_i` if at least one unobstructed ray from any point in `S_T` reaches it:

```
S*_i = { w ∈ S_i | |R'_w| ≥ 1 }
```

---

### Thresholded Overlap Percentages

Models the effect of expanding a uniform margin around the target and measuring what fraction of each OAR falls within that margin — a geometric surrogate for dose fall-off.

```python
# mode: "volume" | "surface" | "rcvs"
overlap = m.getPercentageOverlapDiff(mode="volume")
```

**How it works.** A dilated target mask at radius `r` is defined by thresholding the distance map of the target:

```
X^r_T = { v ∈ Ω | d_XT(v) ≤ r }
```

The percentage overlap at each radius is:

```
o_i(r) = |X^r_T ∩ X_i| / |X_i|
```

This is computed for every integer radius from the minimum to the maximum separation distance, producing a cumulative overlap curve per OAR. Any of the three mask modes (full volume, surface, RCVS surface) can be used to restrict which voxels of the OAR are included in the overlap calculation.

**Interpretation.** The overlap–radius curve reflects spatial proximity and geometric conformity between the target and each OAR. It enables evaluation of positional trade-offs: a position that increases lung volume but reduces high-margin overlap with a target may be geometrically favourable. The metric assumes isotropic expansion and ignores tissue heterogeneity, beam modulation, and beam-angle geometry; it should be interpreted as a geometric proximity heuristic only.

---

## Working directly with `Mask` and `Region`

```python
from mask import Mask

m = Mask(my_array, dev="cpu")   # accepts np.ndarray or torch.Tensor

dmap   = m.dmap()               # float32 tensor, distance to nearest foreground voxel
surf   = m.surface()            # bool tensor, boundary voxels
sa     = m.surface_area()       # int32 scalar
com    = m.center_of_mass       # float tensor, shape (ndim,)

# Align one mask to another by centre-of-mass shift (affine grid_sample, nearest)
aligned = m.alignTo(other_mask)

# Bidirectional surface distance between two masks
bsd_combined, st_d, ts_d = m.getBSD(other_mask)

# Ray-cast visible surface with respect to a target mask
rcvs_mask = m.getRCVS(target_mask, cutAwayDist=8, N=32, chunk_size=1)
```

```python
from region import Region

r = Region(
    mask_target, mask_oar1, mask_oar2,
    target=0,           # index or label string of the target structure
    anchor=0,           # index, label, or coordinate tensor for the anchor point
    labels=["target", "oar1", "oar2"],
    dev="cpu",
)

vecs  = r.getDisplacementVectors(useAnchor=True)
dists = r.getSeparationDistances(mode="surface", distances_only=True)
percs = r.getThresholdedOverlapPercentages(mode="rcvs")
```

---

## Distance Transform

`distance_transform_edt` is a drop-in PyTorch replacement for `scipy.ndimage.distance_transform_edt`. It supports 2-D and 3-D tensors and optional anisotropic voxel spacing.

```python
from distance_transform_edt import distance_transform_edt
import torch

mask = torch.zeros(64, 64, 64, dtype=torch.bool)
mask[20:44, 20:44, 20:44] = True

# Distance from every voxel to the nearest foreground voxel
dist = distance_transform_edt(~mask)

# Anisotropic voxel spacing (z, y, x) in mm
dist = distance_transform_edt(~mask, sampling=(3.0, 1.0, 1.0))
```

The implementation uses a cummax/cummin scan along the first axis and the Meijster algorithm along all remaining axes. `torch.compile` is attempted at import time and silently falls back to eager mode if compilation fails.

---

## Utility

```python
from utils import seriesAnalysis, prettyPrintTable, SERIES_ANALYSIS_LABELS
# SERIES_ANALYSIS_LABELS = ["MIN", "P05", "P10", "AVG", "MDN", "P90", "P95", "MAX"]

# Returns an 8-element tensor of summary statistics
stats = seriesAnalysis(some_tensor)

# Pretty-print a comparison table to stdout
prettyPrintTable(
    rowNames=["bladder", "rectum"],
    rows=[[diff1, val_A1, val_B1], [diff2, val_A2, val_B2]],
    labels=["Diff", "Upright", "Supine"],
    rounding=3,
)
```

---

## PyPI

[https://pypi.org/project/guava-rt/](https://pypi.org/project/guava-rt/)

---

## License

See `LICENSE` for details.