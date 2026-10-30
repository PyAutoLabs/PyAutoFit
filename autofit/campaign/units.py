"""
Parsing and formatting of the time, memory and array-index values a scheduler speaks.

Slurm writes times as ``minutes``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` or
``D-HH:MM:SS`` (with an optional ``.fff`` fraction from ``sacct``), and memory as an
integer number of MB, optionally suffixed ``K``/``M``/``G``/``T`` and, in older
``ReqMem`` output, ``n`` (per node) or ``c`` (per CPU).
"""
import re
from typing import Iterable, List, Optional, Tuple, Union

_MEMORY = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)B?([nc]?)\s*$", re.IGNORECASE)
_MEMORY_SCALE = {"": 1.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0**2}


def parse_time(value: Union[str, int, float, None]) -> Optional[int]:
    """
    Seconds in a Slurm time value, or ``None`` for an empty, ``UNLIMITED``,
    ``Partition_Limit`` or ``INVALID`` value. A bare number is minutes (Slurm's
    convention for ``--time``).

    Raises
    ------
    ValueError
        If ``value`` is not a Slurm time.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value * 60)
    text = value.strip()
    if not text or text.upper() in ("UNLIMITED", "PARTITION_LIMIT", "INVALID", "NONE"):
        return None

    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
        parts = text.split(":")
        # D-HH, D-HH:MM, D-HH:MM:SS
        parts += ["0"] * (3 - len(parts))
        hours, minutes, seconds = parts
    else:
        parts = text.split(":")
        if len(parts) == 1:
            return int(float(parts[0]) * 60)
        if len(parts) == 2:
            hours, (minutes, seconds) = "0", parts
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            raise ValueError(f"Not a Slurm time: {value!r}")
    try:
        return int(
            days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        )
    except ValueError:
        raise ValueError(f"Not a Slurm time: {value!r}")


def format_time(seconds: Optional[float]) -> Optional[str]:
    """
    ``seconds`` as Slurm's ``[D-]HH:MM:SS``, rounded up to the whole second.
    """
    if seconds is None:
        return None
    total = int(-(-seconds // 1))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{clock}" if days else clock


def parse_memory_mb(value: Union[str, int, float, None]) -> Optional[float]:
    """
    MB in a memory value (``12288``, ``12288M``, ``12G``, ``29823.80M``, ``3072Mc``),
    or ``None`` for an empty value. A bare number is MB.

    Raises
    ------
    ValueError
        If ``value`` is not a memory value.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if not text:
        return None
    match = _MEMORY.match(text)
    if match is None:
        raise ValueError(f"Not a memory value: {value!r}")
    number, unit, _ = match.groups()
    return float(number) * _MEMORY_SCALE[unit.upper()]


def memory_per(value: Optional[str]) -> Optional[str]:
    """
    ``"node"``, ``"cpu"`` or ``None``: what an old-style ``ReqMem`` value is per.
    """
    if not value:
        return None
    match = _MEMORY.match(value.strip())
    if match is None or not match.group(3):
        return None
    return {"n": "node", "c": "cpu"}[match.group(3).lower()]


def array_spec(indices: Iterable[int], throttle: Optional[int] = None) -> str:
    """
    A Slurm ``--array`` value for ``indices``: sorted runs collapsed into ranges
    (``0-9,12,15-20``), with ``%throttle`` when given.
    """
    values = sorted(set(int(index) for index in indices))
    if not values:
        raise ValueError("An array needs at least one index.")
    runs: List[Tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    spec = ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)
    if throttle:
        spec += f"%{int(throttle)}"
    return spec


def parse_array_spec(spec: str) -> Tuple[List[int], Optional[int]]:
    """
    The indices and throttle of a Slurm ``--array`` value (``0-9:2,12%4``).
    """
    throttle = None
    if "%" in spec:
        spec, throttle_text = spec.split("%", 1)
        throttle = int(throttle_text)
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_text = part.split(":", 1)
            step = int(step_text)
        if "-" in part:
            a, b = part.split("-", 1)
            indices.extend(range(int(a), int(b) + 1, step))
        else:
            indices.append(int(part))
    return indices, throttle


def requests_gpu(gres: Optional[str]) -> bool:
    """
    Whether a ``--gres`` / ``--gpus`` / TRES request asks for a GPU (``gpu:1``,
    ``gres/gpu=1``, ``gpu:a100:2``). A partition name never tells you the device:
    only the request does.
    """
    if not gres:
        return False
    for item in re.split(r"[,\s]+", str(gres)):
        item = item.strip().lower()
        if not item:
            continue
        name = item.split("=", 1)[0].split(":", 1)[0]
        if name in ("gpu", "gres/gpu", "gpus") or name.startswith("gres/gpu"):
            count = item.split("=", 1)[1] if "=" in item else None
            if count is not None and count.strip() in ("0", ""):
                continue
            return True
    return False
