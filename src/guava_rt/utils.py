"""Summary statistics and table printing for metric results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from guava_rt._deprecation import alias, renamed_kwargs

SERIES_ANALYSIS_LABELS = ["MIN", "P05", "P10", "AVG", "MDN", "P90", "P95", "MAX"]


def series_analysis(arr: Tensor) -> Tensor:
    """The 8 statistics named in :data:`SERIES_ANALYSIS_LABELS`; zeros for an empty input."""
    arr = arr.flatten()
    if len(arr) == 0:
        return torch.zeros(8, device=arr.device)
    return torch.stack(
        [
            arr.min(),
            torch.quantile(arr, 0.05),
            torch.quantile(arr, 0.10),
            arr.mean(),
            torch.quantile(arr, 0.50),
            torch.quantile(arr, 0.90),
            torch.quantile(arr, 0.95),
            arr.max(),
        ]
    )


@renamed_kwargs(rowNames="row_names", minWidth="min_width")
def pretty_print_table(
    row_names: Sequence[str],
    rows: Sequence[Sequence[Any]],
    labels: Sequence[str],
    rounding: int = 3,
    min_width: int = 0,
) -> None:
    """Print ``rows`` as a boxed table; floats and scalar tensors are rounded to ``rounding``."""

    def fmt(value: Any) -> str:
        if isinstance(value, Tensor):
            value = float(value.item())
        if isinstance(value, float):
            value = round(value, rounding)
        return str(value)

    cells = [[fmt(v) for v in row] for row in rows]
    widths = [max(map(len, row_names))]
    for i, label in enumerate(labels):
        widths.append(max([len(label), min_width, *(len(row[i]) for row in cells)]))

    def line(items: Sequence[str]) -> str:
        return "| " + "| ".join(s.ljust(w + 1) for s, w in zip(items, widths, strict=True)) + "|"

    row_sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(row_sep)
    print(line(["", *labels]))
    print(row_sep)
    for name, row in zip(row_names, cells, strict=True):
        print(line([name, *row]))
        print(row_sep)


# Deprecated camelCase API (guava-rt <= 0.5)
seriesAnalysis = alias(series_analysis, "seriesAnalysis")
prettyPrintTable = alias(pretty_print_table, "prettyPrintTable")
