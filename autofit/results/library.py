"""
``ResultsLibrary``: what a set of fits is *expected* to have produced — datasets ×
searches, and the products each result must hold — so ``check`` can say what is
missing rather than only counting what is there.

A library is declared in one of three ways:

- by API, ``ResultsLibrary.grid(datasets, searches, roots=...)`` or a list of
  ``ExpectedResult``;
- in ``library.yaml`` (``ResultsLibrary.from_yaml``):

  .. code-block:: yaml

      name: dr1_v1                     # names <name>.missing.csv (default: file stem)
      roots: [output]                  # where the results are, relative to this file
      index_dir: .autofit_index        # optional, as for ``ResultsIndex``
      datasets_file: manifest.txt      # or ``datasets: [a, b, ...]``
      searches:
        source_lp:
          products: [image, latent, files/model.fits]
          location: "{dataset}/source_lp"   # optional, relative to the first root
        mass_total: [image, samples_summary]

- or from an hpc-campaign file (``from_campaign: campaign.yaml`` in ``library.yaml``,
  or ``ResultsLibrary.from_campaign``): every manifest line is a dataset, every stage
  with a ``success.zip_complete`` is a search found at that path, and its
  ``success.members`` are its required products. The campaign file stays the single
  declaration of what was run; a library never keeps a second manifest.

A ``dataset`` without a ``location`` is matched to results by the search ``name`` and
the dataset as the result's ``unique_tag``, ``object_id`` (``dataset_identity.json``),
``path_prefix`` or last ``path_prefix`` segment.

A product is either a name the index already records — ``samples``,
``samples_summary``, ``latent``, ``image``, ``dataset_identity``, ``search_internal``
— or a path inside the result (``files/model.fits``; ``image/`` for a folder), checked
against the zip's member list without reading any member.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from autofit.results.index import DEFAULT_LATEST_KEY, ResultsIndex

PathLike = Union[str, os.PathLike]

LIBRARY_FIELDS = {
    "name",
    "roots",
    "index_dir",
    "datasets",
    "datasets_file",
    "searches",
    "from_campaign",
    "stages",
    "products",
    "note",
}


class LibraryException(Exception):
    pass


@dataclass(frozen=True)
class ExpectedResult:
    dataset: str
    """The dataset (object) the result is a fit of."""
    search: str
    """The search's ``name`` (for a campaign, the stage)."""
    products: Tuple[str, ...] = ()
    """What the result must hold beyond being complete."""
    location: Optional[str] = None
    """Where the result is: its ``output_path``, or the folder holding its identifier
    folders (``<path_prefix>/<name>``). ``None`` matches by dataset and search name."""
    exact: bool = False
    """``location`` is the ``output_path`` itself and the result must be a zip there
    holding the products as members — how an hpc-campaign's ``success`` check reads
    it (``autofit.campaign.collector.check_success``)."""
    error: Optional[str] = None
    """Why where the result should be is not known (e.g. a manifest line missing a
    field); such a result is reported ``failed`` with this reason."""


def _datasets_from_file(path: Path) -> List[str]:
    datasets = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            datasets.append(line.split()[0])
    return datasets


def _products(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(product) for product in value)


def _products_and_location(value) -> Tuple[Tuple[str, ...], Optional[str]]:
    if value is None or isinstance(value, str):
        return _products(value), None
    if isinstance(value, (list, tuple)):
        return tuple(str(product) for product in value), None
    if isinstance(value, Mapping):
        unknown = set(value) - {"products", "location"}
        if unknown:
            raise LibraryException(
                f"A search takes 'products' and 'location', not {sorted(unknown)}."
            )
        return _products(value.get("products")), value.get("location")
    raise LibraryException(
        f"A search is a list of products or a mapping, not {value!r}."
    )


class ResultsLibrary:
    def __init__(
        self,
        expected: Iterable[ExpectedResult],
        roots: Union[PathLike, Sequence[PathLike]],
        index_dir: Optional[PathLike] = None,
        name: str = "library",
        latest_key: Tuple[str, ...] = DEFAULT_LATEST_KEY,
    ):
        """
        The results ``expected`` under ``roots``.

        Parameters
        ----------
        expected
            Every result the library should hold.
        roots
            Where the results are (as for ``ResultsIndex``).
        index_dir
            Where the index is kept (as for ``ResultsIndex``).
        name
            Names the gaps file, ``<name>.missing.csv``.
        """
        self.expected = list(expected)
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        self.roots = [Path(root) for root in roots]
        if len(self.roots) != 1 and index_dir is None:
            raise LibraryException("A library with several roots needs an index_dir.")
        self.index_dir = index_dir
        self.name = name
        self.latest_key = tuple(latest_key)

    @classmethod
    def grid(
        cls,
        datasets: Iterable[str],
        searches: Mapping[str, Union[Sequence[str], Mapping, None]],
        roots: Union[PathLike, Sequence[PathLike]],
        **kwargs,
    ) -> "ResultsLibrary":
        """
        Every dataset × every search.

        Parameters
        ----------
        searches
            Search name to its required products, or to ``{products: [...],
            location: "..."}`` where ``location`` is a template with ``{dataset}``,
            relative to the first root.
        """
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        first_root = Path(roots[0])
        expected = []
        for dataset in datasets:
            for search, value in searches.items():
                products, location = _products_and_location(value)
                if location is not None:
                    location = str(first_root / location.format(dataset=dataset))
                expected.append(
                    ExpectedResult(
                        dataset=str(dataset),
                        search=str(search),
                        products=products,
                        location=location,
                    )
                )
        return cls(expected, roots=roots, **kwargs)

    @classmethod
    def from_campaign(
        cls,
        campaign,
        stages: Optional[Sequence[str]] = None,
        products: Optional[Mapping[str, Sequence[str]]] = None,
        roots: Optional[Union[PathLike, Sequence[PathLike]]] = None,
        **kwargs,
    ) -> "ResultsLibrary":
        """
        The library an hpc-campaign declares: each manifest line × each stage.

        Parameters
        ----------
        campaign
            An ``autofit.campaign.Campaign`` or the path of its ``campaign.yaml``.
        stages
            The stages to expect (default: every stage with a ``success.zip_complete``).
        products
            Required products added per stage to its ``success.members``.
        roots
            Where to index (default: the campaign's ``status_root``).
        """
        from autofit.campaign.spec import Campaign

        if not isinstance(campaign, Campaign):
            campaign = Campaign.load(campaign)
        products = dict(products or {})

        if stages is None:
            stages = [
                name
                for name, stage in campaign.stages.items()
                if stage.success.zip_complete
            ]
        for name in stages:
            if name not in campaign.stages:
                raise LibraryException(f"Campaign {campaign.name} has no stage {name}.")
            if not campaign.stages[name].success.zip_complete:
                raise LibraryException(
                    f"Stage {name} declares no success.zip_complete, so where its "
                    f"results are is not known."
                )
        unknown = set(products) - set(stages)
        if unknown:
            raise LibraryException(f"products names unknown stages {sorted(unknown)}.")

        expected = []
        for index in campaign.indices():
            fields = campaign.fields(index)
            for name in stages:
                success = campaign.stages[name].success
                required = tuple(success.members) + _products(products.get(name))
                try:
                    location = campaign.path(success.zip_complete.format(**fields))
                except (KeyError, IndexError) as exception:
                    expected.append(
                        ExpectedResult(
                            dataset=fields["line"],
                            search=name,
                            products=required,
                            exact=True,
                            error=f"manifest_line:{index} has no field {exception}",
                        )
                    )
                    continue
                if location.suffix == ".zip":
                    location = location.with_suffix("")
                expected.append(
                    ExpectedResult(
                        dataset=fields["line"],
                        search=name,
                        products=required,
                        location=str(location),
                        exact=True,
                    )
                )

        kwargs.setdefault("name", str(campaign.name))
        return cls(
            expected,
            roots=roots if roots is not None else campaign.path(campaign.status_root),
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ResultsLibrary":
        """
        The library declared in ``library.yaml`` (see the module docstring). Relative
        paths are relative to the file.
        """
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exception:
            raise LibraryException(f"Library file {path} cannot be read: {exception}")
        if not isinstance(raw, dict):
            raise LibraryException(f"Library file {path} must be a mapping.")
        unknown = set(raw) - LIBRARY_FIELDS
        if unknown:
            raise LibraryException(f"Library file {path}: unknown keys {sorted(unknown)}.")

        def relative(value):
            value = Path(value)
            return value if value.is_absolute() else path.parent / value

        roots = raw.get("roots")
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        roots = [relative(root) for root in roots] if roots else None
        kwargs = {"name": raw.get("name") or path.stem}
        if raw.get("index_dir") is not None:
            kwargs["index_dir"] = relative(raw["index_dir"])

        if raw.get("from_campaign"):
            clashes = {"datasets", "datasets_file", "searches"} & set(raw)
            if clashes:
                raise LibraryException(
                    f"Library file {path}: from_campaign takes its datasets and "
                    f"searches from the campaign; drop {sorted(clashes)}."
                )
            return cls.from_campaign(
                relative(raw["from_campaign"]),
                stages=raw.get("stages"),
                products=raw.get("products"),
                roots=roots,
                **kwargs,
            )

        for key in ("stages", "products"):
            if key in raw:
                raise LibraryException(
                    f"Library file {path}: '{key}' only applies with from_campaign."
                )
        if roots is None:
            raise LibraryException(f"Library file {path} names no roots.")
        if ("datasets" in raw) == ("datasets_file" in raw):
            raise LibraryException(
                f"Library file {path} needs exactly one of datasets and datasets_file."
            )
        datasets = (
            [str(dataset) for dataset in raw["datasets"]]
            if "datasets" in raw
            else _datasets_from_file(relative(raw["datasets_file"]))
        )
        searches = raw.get("searches")
        if not isinstance(searches, dict) or not searches:
            raise LibraryException(f"Library file {path} declares no searches.")
        return cls.grid(datasets, searches, roots=roots, **kwargs)

    def index(self) -> ResultsIndex:
        return ResultsIndex(
            self.roots, index_dir=self.index_dir, latest_key=self.latest_key
        )

    def check(self, update: bool = True):
        """
        Compare the library with the results index.

        Parameters
        ----------
        update
            Bring the index up to date first (only new or changed results are opened).

        Returns
        -------
        A ``CompletenessReport``.
        """
        from autofit.results.completeness import check

        index = self.index()
        if update:
            index.update()
        return check(self, index.records())
