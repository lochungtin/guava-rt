from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SERIES_ANALYSIS_LABELS = ["MIN", "P05", "P10", "AVG", "MDN", "P90", "P95", "MAX"]


def seriesAnalysis(arr):
    arr = arr.flatten()
    if len(arr) == 0:
        return torch.zeros(8)
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


def prettyPrintTable(rowNames, rows, labels, rounding: int = 3, minWidth: int = 0):
    def pad(data, gaps):
        return [str(d).ljust(g + 1) for d, g in zip(data, gaps)]

    for r_idx in range(len(rows)):
        rows[r_idx] = [
            (
                round(
                    float(i.item()) if isinstance(i, torch.Tensor) else float(i),
                    rounding,
                )
                if isinstance(i, (float, torch.Tensor))
                else i
            )
            for i in rows[r_idx]
        ]

    gaps = [max(map(len, rowNames))]
    for i in range(len(labels)):
        col_max = len(labels[i])
        for row in rows:
            col_max = max(len(str(row[i])), col_max, minWidth)
        gaps.append(col_max)

    rows = [list(map(str, row)) for row in rows]
    row_sep = "+" + "+".join("-" * (g + 2) for g in gaps) + "+"

    print(row_sep)
    print("| " + "| ".join(pad([""] + labels, gaps)) + "|")
    print(row_sep)
    for name, row in zip(rowNames, rows):
        print("| " + "| ".join(pad([name] + row, gaps)) + "|")
        print(row_sep)
