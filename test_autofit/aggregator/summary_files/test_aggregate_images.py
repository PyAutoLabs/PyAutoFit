from enum import Enum

import pytest
from pathlib import Path

from PIL import Image

from autofit.aggregator import Aggregator
from autofit.aggregator.summary.aggregate_images import AggregateImages

class SubplotFit(Enum):
    """
    The subplots that can be extracted from the subplot_fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    Data = (0, 0)
    DataSourceScaled = (1, 0)
    SignalToNoiseMap = (2, 0)
    ModelData = (3, 0)
    LensLightModelData = (0, 1)
    LensLightSubtractedImage = (1, 1)
    SourceModelData = (2, 1)
    SourcePlaneZoomed = (3, 1)
    NormalizedResidualMap = (0, 2)
    NormalizedResidualMapOneSigma = (1, 2)
    ChiSquaredMap = (2, 2)
    SourcePlaneNoZoom = (3, 2)

@pytest.fixture
def aggregate(aggregator):
    return AggregateImages(aggregator)


def test(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    )
    assert result.size == (122, 120)
    assert result == aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    )


def test_different_plots(aggregate):
    assert aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    ) != aggregate.extract_image(
        [
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.Data,
        ],
    )


def test_longer(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.NormalizedResidualMap,
            SubplotFit.SourcePlaneNoZoom,
            SubplotFit.SourceModelData,
        ],
    )

    assert result.size == (183, 120)


def test_subplot_width(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.NormalizedResidualMap,
            SubplotFit.SourcePlaneNoZoom,
            SubplotFit.SourceModelData,
        ],
        subplot_width=2,
    )

    assert result.size == (122, 240)


def test_output_to_folder(aggregate, output_directory):
    aggregate.output_to_folder(
        output_directory,
        "id",
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelData,
        ],
    )
    assert list(Path(output_directory).glob("*.png"))


def test_list_of_names(aggregate, output_directory):
    aggregate.output_to_folder(
        output_directory,
        ["one", "two"],
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelData,
        ],
    )
    assert set([path.name for path in Path(output_directory).glob("*.png")]) == set([
        "two.png",
        "one.png",
    ])


def test_output_to_folder_name(
    aggregate,
    output_directory,
    aggregator,
):
    aggregate.output_to_folder(
        output_directory,
        "id",
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelData,
        ],
    )

    id_ = next(iter(aggregator)).id
    assert list(Path(output_directory).glob(f"{id_}.png"))


def test_custom_images(
    aggregate,
    aggregator,
):
    image = Image.new("RGB", (10, 10), "white")
    images = [image for _ in aggregator]

    result = aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelData,
            images,
        ]
    )

    assert result.size == (244, 120)


def test_custom_function(aggregate):
    def make_image(output):
        return Image.new("RGB", (10, 10), "white")

    result = aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelData,
            make_image,
        ]
    )

    assert result.size == (244, 120)


def test_panel_size_explicit_normalizes_all_panels(aggregate, aggregator):
    small = Image.new("RGB", (10, 10), "white")
    images = [small for _ in aggregator]

    result = aggregate.extract_image(
        [SubplotFit.Data, images],
        panel_size=(50, 50),
    )

    assert result.size == (100, 100)


def test_panel_size_defaults_to_max(aggregate, aggregator):
    small = Image.new("RGB", (10, 10), "white")
    images = [small for _ in aggregator]

    result = aggregate.extract_image(
        [SubplotFit.Data, images],
    )

    assert result.size == (122, 120)


def test_custom_subplot_fit(aggregate):
    class SubplotFit(Enum):
        """
        The subplots that can be extracted from the subplot_fit image.

        The values correspond to the position of the subplot in the 4x3 grid.
        """

        Data = (0, 0)

    result = aggregate.extract_image(
        [
            SubplotFit.Data,
        ]
    )
    assert result.size == (244, 362)


def test_bad_aggregator():
    directory = Path(__file__).parent / "aggregate_summaries"
    aggregator = Aggregator.from_directory(directory)

    with pytest.raises(ValueError):
        AggregateImages(aggregator)
