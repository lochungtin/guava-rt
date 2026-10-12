"""guava-rt: Geometric Upright-supine Anatomical Variation Analysis for binary masks.

Volume, surface, displacement, surface-distance, separation-distance and overlap metrics
between pairs of 2-D or 3-D segmentations, implemented in PyTorch for CPU or CUDA.
"""

from importlib.metadata import PackageNotFoundError, version

from guava_rt.distance_transform_edt import distance_transform_edt
from guava_rt.mask import Mask
from guava_rt.metrics import Metrics
from guava_rt.region import Region
from guava_rt.utils import (
    SERIES_ANALYSIS_LABELS,
    pretty_print_table,
    prettyPrintTable,
    series_analysis,
    seriesAnalysis,
)

try:
    __version__ = version("guava-rt")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "SERIES_ANALYSIS_LABELS",
    "Mask",
    "Metrics",
    "Region",
    "__version__",
    "distance_transform_edt",
    "pretty_print_table",
    "series_analysis",
]

# deprecated aliases, importable but not part of __all__
_ = (prettyPrintTable, seriesAnalysis)
