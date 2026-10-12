import numpy as np
import torch

from guava_rt import SERIES_ANALYSIS_LABELS, pretty_print_table, series_analysis


def test_series_analysis_matches_numpy() -> None:
    x = torch.rand(1000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    out = series_analysis(x.view(10, 100))
    ref = [
        x.min(),
        *np.quantile(x.numpy(), [0.05, 0.10]),
        x.mean(),
        *np.quantile(x.numpy(), [0.5, 0.9, 0.95]),
        x.max(),
    ]
    assert len(SERIES_ANALYSIS_LABELS) == out.shape[0] == 8
    np.testing.assert_allclose(out.numpy(), np.array(ref, dtype=float))


def test_series_analysis_of_empty_input() -> None:
    assert torch.equal(series_analysis(torch.zeros(0)), torch.zeros(8))


def test_pretty_print_table(capsys) -> None:  # type: ignore[no-untyped-def]
    rows = [[torch.tensor(1.23456), 2, "x"], [0.5, 10, "longer"]]
    original = [list(r) for r in rows]
    pretty_print_table(["a", "bb"], rows, ["Diff", "N", "Label"], rounding=2)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "+----+------+----+--------+",
        "|    | Diff | N  | Label  |",
        "+----+------+----+--------+",
        "| a  | 1.23 | 2  | x      |",
        "+----+------+----+--------+",
        "| bb | 0.5  | 10 | longer |",
        "+----+------+----+--------+",
    ]
    assert rows == original  # input is not modified


def test_pretty_print_table_min_width(capsys) -> None:  # type: ignore[no-untyped-def]
    pretty_print_table(["r"], [[1]], ["c"], min_width=5)
    assert capsys.readouterr().out.splitlines()[1] == "|   | c     |"
