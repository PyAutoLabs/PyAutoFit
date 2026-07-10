import numpy as np

from autofit.non_linear.plot.samples_plotters import _corner_range_from


def test__corner_range__real_spread_columns_preserved():
    data = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    plot_range = _corner_range_from(data)

    assert plot_range == [(1.0, 3.0), (10.0, 30.0)]


def test__corner_range__degenerate_columns_widened_to_nonzero_span():
    # Columns 0, 1, 2 have no dynamic range (all samples equal) — the input
    # that made `corner` raise "no dynamic range". Column 3 has real spread.
    data = np.array(
        [
            [0.0, 5.0, -2.0, 1.0],
            [0.0, 5.0, -2.0, 2.0],
            [0.0, 5.0, -2.0, 3.0],
        ]
    )

    plot_range = _corner_range_from(data)

    # Every column ends up with a strictly positive width, so `corner` gets a
    # valid range for all of them and does not raise.
    for lower, upper in plot_range:
        assert upper > lower

    # Degenerate windows stay centred on the constant value.
    for (lower, upper), value in zip(plot_range[:3], [0.0, 5.0, -2.0]):
        assert lower < value < upper

    # The real-spread column is left untouched.
    assert plot_range[3] == (1.0, 3.0)
