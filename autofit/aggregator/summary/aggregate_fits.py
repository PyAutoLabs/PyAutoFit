import re
from enum import Enum
from typing import Dict, List, Union

from pathlib import Path

from autofit.aggregator.search_output import SearchOutput
from autofit.aggregator import Aggregator


def subplot_filename(subplot: Enum) -> str:
    subplot_type = subplot.__class__
    return (
        re.sub(
            r"([A-Z])",
            r"_\1",
            subplot_type.__name__.replace("FITS", ""),
        )
        .lower()
        .lstrip("_")
    )

class AggregateFITS:
    def __init__(self, aggregator: Union[Aggregator, List[SearchOutput]]):
        """
        A class for extracting fits files from the aggregator.

        Parameters
        ----------
        aggregator
            The aggregator containing the fits files.
        """
        if len(aggregator) == 0:
            raise ValueError("The aggregator is empty.")

        self.aggregator = aggregator

    @staticmethod
    def _hdus(
        result: SearchOutput,
        hdus: List[Enum],
        extname_prefix = None
    ) -> "List[fits.ImageHDU]":
        """
        Extract the HDUs from a given fits for a given search.

        Each source fits file is opened once (not once per requested HDU) and closed
        deterministically — the extracted data is copied out of the memmap, so no
        file handle stays open per result. Previously each HDU leaked one handle,
        which exhausted the default open-file limit when aggregating many hundreds
        of results.

        Parameters
        ----------
        result
            The search output.
        hdus
            The HDUs to extract.

        Returns
        -------
        The extracted HDUs.
        """
        from astropy.io import fits

        row = []
        sources = {}
        try:
            for hdu in hdus:
                source_name = subplot_filename(hdu)
                if source_name not in sources:
                    sources[source_name] = result.value(source_name)
                source = sources[source_name]
                source_hdu = source[source.index_of(hdu.value)]

                if extname_prefix is not None:
                    source_hdu.header["EXTNAME"] = f"{extname_prefix.upper()}_{source_hdu.header['EXTNAME']}"

                row.append(
                    fits.ImageHDU(
                        data=source_hdu.data.copy(),
                        header=source_hdu.header,
                    )
                )
        finally:
            for source in sources.values():
                source.close()
        return row

    def extract_fits(self, hdus: List[Enum], extname_prefix_list = None) -> "fits.HDUList":
        """
        Extract the HDUs from the fits files for every search in the aggregator.

        Return the result as a list of HDULists. The first HDU in each list is an empty PrimaryHDU.

        Parameters
        ----------
        hdus
            The HDUs to extract.

        Returns
        -------
        The extracted HDUs.
        """
        from astropy.io import fits

        output = [fits.PrimaryHDU()]
        for i, result in enumerate(self.aggregator):

            extname_prefix = extname_prefix_list[i] if extname_prefix_list is not None else None

            output.extend(self._hdus(result, hdus, extname_prefix))

        return fits.HDUList(output)

    def extract_csv(self, filename: str) -> List[Dict]:
        """
        Extract .csv files which store imaging results that are typically on irregular grids and thus don't suit
        a .fits file.

        Return the result as a list of Dicts corresponding to the fits in the aggregator.

        Parameters
        ----------
        filename
            The name of the .csv file to extract.

        Returns
        -------
        The extracted HDUs.
        """
        from astropy.table import Table

        output = []
        for result in self.aggregator:
            source = result.value(filename)
            try:
                output.append(Table.read(source, format="fits").copy())
            finally:
                source.close()

        return output

    def output_to_folder(
        self,
        folder: Path,
        name: Union[str, List[str]],
        hdus: List[Enum],
    ):
        """
        Output the fits files for every search in the aggregator to a folder.

        Only include HDUs specific in the hdus argument.

        Parameters
        ----------
        folder
            The folder to output the fits files to.
        hdus
            The HDUs to output.
        name
            The name of the fits file. This is the attribute of the search output that is used to name the file.
            OR a list of names for each HDU.
        """
        from astropy.io import fits

        folder.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(self.aggregator):
            if isinstance(name, str):
                output_name = getattr(result, name)
            else:
                output_name = name[i]

            hdu_list = fits.HDUList(
                [fits.PrimaryHDU()]
                + self._hdus(
                    result,
                    hdus,
                )
            )
            with open(folder / f"{output_name}.fits", "wb") as file:
                hdu_list.writeto(file)
