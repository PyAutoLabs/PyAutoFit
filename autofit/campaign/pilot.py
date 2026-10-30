"""
``campaign pilot``: a small, stratified subset of a campaign to measure before the rest.

The pilot is a list of campaign indices — a fraction of the manifest (default 2 %)
drawn evenly from bins of input size, so the large and small objects that set the
p95 wall time and the peak memory are both in it. Sizes come from a manifest field
(``size_field``) or from the size on disk of a templated path (``size_path``, filled
from the manifest fields like a success check); with neither, the draw is uniform.

``write_pilot`` writes ``pilot.json`` (the indices, as a Slurm array spec, with the
bins they came from). **Nothing is submitted**: the file ends with the command a human
runs once they approve it, ``submit <campaign.yaml> --indices <spec>``, after which
``status`` measures the pilot and ``plan --pilot`` reads it.
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

from autofit.campaign.units import array_spec
from autofit.tools.util import open_atomic

DEFAULT_FRACTION = 0.02
DEFAULT_BINS = 5


def object_sizes(campaign, size_field: Optional[str] = None, size_path: Optional[str] = None) -> Dict[int, float]:
    """
    The size of each manifest object: a numeric manifest field, or the bytes on disk of
    ``size_path`` formatted with the object's fields. Objects whose size cannot be read
    are left out.
    """
    sizes = {}
    for index in campaign.indices():
        fields = campaign.fields(index)
        try:
            if size_field:
                sizes[index] = float(fields[size_field])
            elif size_path:
                sizes[index] = float(campaign.path(size_path.format(**fields)).stat().st_size)
        except (KeyError, ValueError, OSError):
            continue
    return sizes


def pilot_indices(
    campaign,
    fraction: float = DEFAULT_FRACTION,
    size_field: Optional[str] = None,
    size_path: Optional[str] = None,
    bins: int = DEFAULT_BINS,
    seed: int = 0,
) -> Dict:
    """
    A stratified pilot of ``campaign``: ``fraction`` of its objects (at least one per
    size bin), drawn with ``seed`` so the same call gives the same pilot.
    """
    indices = campaign.indices()
    rng = random.Random(seed)
    sizes = object_sizes(campaign, size_field, size_path) if (size_field or size_path) else {}
    total = max(1, math.ceil(len(indices) * fraction))

    if not sizes:
        chosen = sorted(rng.sample(indices, min(total, len(indices))))
        return {
            "indices": chosen,
            "fraction": fraction,
            "seed": seed,
            "stratified_by": None,
            "bins": [{"objects": len(indices), "chosen": len(chosen)}],
        }

    ordered = sorted(sizes, key=lambda index: (sizes[index], index))
    bins = max(1, min(bins, len(ordered)))
    strata = [ordered[len(ordered) * b // bins : len(ordered) * (b + 1) // bins] for b in range(bins)]
    per_bin = max(1, math.ceil(total / bins))
    chosen = []
    summary = []
    for stratum in strata:
        picked = rng.sample(stratum, min(per_bin, len(stratum)))
        chosen += picked
        summary.append(
            {
                "size_range": [sizes[stratum[0]], sizes[stratum[-1]]],
                "objects": len(stratum),
                "chosen": len(picked),
            }
        )
    return {
        "indices": sorted(chosen),
        "fraction": fraction,
        "seed": seed,
        "stratified_by": size_field or size_path,
        "unsized_objects": len(indices) - len(sizes),
        "bins": summary,
    }


def write_pilot(directory, campaign, pilot: Dict, campaign_file: Optional[str] = None) -> Path:
    """
    Write ``pilot.json`` into ``directory`` with the array spec and the (unrun) command
    that would submit it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = array_spec(pilot["indices"])
    document = dict(
        pilot,
        campaign=campaign.name,
        tasks=len(pilot["indices"]),
        array_spec=spec,
        submit_command=(
            f"python -m autofit.campaign submit {campaign_file or campaign.source} --indices {spec}"
        ),
        note="Not submitted. Run the command once the pilot is approved.",
    )
    path = directory / "pilot.json"
    with open_atomic(path, "w") as f:
        json.dump(document, f, indent=1)
    return path
