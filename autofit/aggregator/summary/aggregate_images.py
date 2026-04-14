import sys
from typing import Optional, List, Tuple, Union, Callable, Type
from pathlib import Path

from PIL import Image

from autofit.aggregator.search_output import SearchOutput
from autofit.aggregator.aggregator import Aggregator

import re
from enum import Enum


def subplot_filename(subplot: Enum) -> str:
    subplot_type = subplot.__class__
    return (
        re.sub(
            r"([A-Z])",
            r"_\1",
            subplot_type.__name__,
        )
        .lower()
        .lstrip("_")
    )


class SubplotFitImage:
    def __init__(
        self,
        image: Image.Image,
        suplot_type,
    ):
        """
        The subplot_fit image associated with one fit.

        Parameters
        ----------
        image
            The subplot_fit image.
        """
        self._image = image

        x_max = 0
        y_max = 0

        for subplot in suplot_type:
            x, y = subplot.value
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        self._single_image_width = self._image.width // (x_max + 1)
        self._single_image_height = self._image.height // (y_max + 1)

    def image_at_coordinates(
        self,
        x: int,
        y: int,
    ) -> Image.Image:
        """
        Extract the image at the specified coordinates.

        Parameters
        ----------
        x
        y
            The integer coordinates of the plot (see Subplot).

        Returns
        -------
        The extracted image.
        """
        return self._image.crop(
            (
                x * self._single_image_width,
                y * self._single_image_height,
                (x + 1) * self._single_image_width,
                (y + 1) * self._single_image_height,
            )
        )


class AggregateImages:
    def __init__(
        self,
        aggregator: Union[Aggregator, List[SearchOutput]],
    ):
        """
        Extracts images from the aggregator and combines them into one
        overall image.

        Parameters
        ----------
        aggregator
            The aggregator containing the fit results.
        """
        if len(aggregator) == 0:
            raise ValueError("The aggregator is empty.")

        self._aggregator = aggregator
        self._source_images = None

    def extract_image(
        self,
        subplots: List[Union[Enum, List[Image.Image], Callable]],
        subplot_width: Optional[int] = sys.maxsize,
        transpose: bool = False,
        panel_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Extract the images at the specified subplots and combine them into
        one overall image including those subplots for all fits in the
        aggregator.

        Parameters
        ----------
        subplots
            The subplots to output. These can be:
            - enum values
            - lists of the same length as the aggregator output
            - functions that take a SearchOutput as an argument
        subplot_width
            Defines the width of each subplot in number of images.
            If this is greater than the number of subplots then it defaults to
            the number of subplots.
            If this is less than the number of subplots then it causes the
            images to wrap.
        transpose
            If True the output image is transposed before being returned, else it
            is returned as is.
        panel_size
            If provided, all extracted panels are resized to this `(width, height)`
            before compositing. If `None` (the default), panels are resized to the
            maximum width and height across all panels, so that panels extracted
            from source images with different grid layouts end up the same size
            in the output.

        Returns
        -------
        The combined image.
        """
        matrix = []
        for i, result in enumerate(self._aggregator):
            matrix.extend(
                self._matrix_for_result(
                    i,
                    result,
                    subplots,
                    subplot_width=subplot_width,
                )
            )

        if transpose:

            matrix = [list(row) for row in zip(*matrix)]

        matrix = self._normalize_panel_sizes(matrix, panel_size=panel_size)

        return self._matrix_to_image(matrix)

    def output_to_folder(
        self,
        folder: Path,
        name: Union[str, List[str]],
        subplots: List[Union[List[Image.Image], Callable]],
        subplot_width: Optional[int] = sys.maxsize,
        panel_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Output one subplot image for each fit in the aggregator.

        Parameters
        ----------
        folder
            The target folder in which to store the subplots.
        subplots
            The subplots to output. These can be:
            - enum values
            - lists of the same length as the aggregator output
            - functions that take a SearchOutput as an argument
        subplot_width
            Defines the width of each subplot in number of images.
            If this is greater than the number of subplots then it defaults to
            the number of subplots.
            If this is less than the number of subplots then it causes the
            images to wrap.
        name
            The attribute of each fit to use as the name of the output file.
            OR a list of names, one for each fit.
        panel_size
            If provided, all extracted panels are resized to this `(width, height)`
            before compositing. If `None` (the default), panels are resized to the
            maximum width and height across all panels, so that panels extracted
            from source images with different grid layouts end up the same size
            in the output.
        """
        if len(subplots) == 0:
            raise ValueError("At least one subplot must be provided.")

        folder.mkdir(exist_ok=True, parents=True)

        for i, result in enumerate(self._aggregator):
            matrix = self._normalize_panel_sizes(
                self._matrix_for_result(
                    i,
                    result,
                    subplots,
                    subplot_width=subplot_width,
                ),
                panel_size=panel_size,
            )
            image = self._matrix_to_image(matrix)

            if isinstance(name, str):
                output_name = getattr(result, name)
            else:
                output_name = name[i]

            output_path = folder / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(folder / f"{output_name}.png")

    @staticmethod
    def _matrix_for_result(
        i: int,
        result: SearchOutput,
        subplots: List[Union[List[Image.Image], Callable]],
        subplot_width: int = sys.maxsize,
    ) -> List[List[Image.Image]]:
        """
        Create a matrix of images each in the position they will be in the
        final image.

        Parameters
        ----------
        result
            The fit result.
        subplots
            The subplots to output. These can be:
            - enum values
            - lists of the same length as the aggregator output
            - functions that take a SearchOutput as an argument
        subplot_width
            The number of subplots to include in each row of the matrix.

        Returns
        -------
        The matrix of images.
        """
        _images = {}

        def get_image(subplot_: Enum) -> SubplotFitImage:
            """
            Get the image for the subplot.

            This assumes that the subplot filename is the same as the subplot
            class name but using snake_case.

            Parameters
            ----------
            subplot_
                The type of subplot to get the image for.

            Returns
            -------
            The image for the subplot.
            """
            subplot_type = subplot_.__class__
            if subplot_type not in _images:
                _images[subplot_type] = SubplotFitImage(
                    result.image(
                        subplot_filename(subplot_),
                    ),
                    subplot_type
                )
            return _images[subplot_type]

        matrix = []
        row = []
        for subplot in subplots:
            if isinstance(subplot, Enum):
                row.append(
                    get_image(subplot).image_at_coordinates(
                        *subplot.value,
                    )
                )
            elif isinstance(subplot, list):
                if not isinstance(subplot[i], Image.Image):
                    raise TypeError(
                        "The subplots must be of type Subplot or a list of "
                        "images or a function that takes a SearchOutput as an "
                        "argument."
                    )
                row.append(subplot[i])
            else:
                try:
                    row.append(subplot(result))
                except TypeError:
                    raise TypeError(
                        "The subplots must be of type Subplot or a list of "
                        "images or a function that takes a SearchOutput as an "
                        "argument."
                    )

            if len(row) == subplot_width:
                matrix.append(row)
                row = []

        if len(row) > 0:
            matrix.append(row)

        return matrix

    @staticmethod
    def _normalize_panel_sizes(
        matrix: List[List[Image.Image]],
        panel_size: Optional[Tuple[int, int]] = None,
    ) -> List[List[Image.Image]]:
        """
        Resize every panel in the matrix to a common `(width, height)` so that
        panels extracted from source images with different grid layouts are the
        same size when composited.

        If `panel_size` is `None`, the target size is the maximum width and
        height across all panels in the matrix. LANCZOS resampling is used to
        preserve image quality.
        """
        if not matrix:
            return matrix

        if panel_size is None:
            max_width = max(image.width for row in matrix for image in row)
            max_height = max(image.height for row in matrix for image in row)
            target = (max_width, max_height)
        else:
            target = panel_size

        return [
            [
                image if image.size == target else image.resize(target, Image.LANCZOS)
                for image in row
            ]
            for row in matrix
        ]

    @staticmethod
    def _matrix_to_image(matrix: List[List[Image.Image]]) -> Image.Image:
        """
        Create an image including all the images in the matrix.

        Parameters
        ----------
        matrix
            The matrix of images to combine into one image

        Returns
        -------
        The combined image.
        """
        total_width = max(sum(image.width for image in row) for row in matrix)
        total_height = max(
            sum(image.height for image in column) for column in list(zip(*matrix))
        )

        image = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for row in matrix:
            x_offset = 0
            for subplot_image in row:
                image.paste(subplot_image, (x_offset, y_offset))
                x_offset += subplot_image.width
            y_offset += row[0].height

        return image
