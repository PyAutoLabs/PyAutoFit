"""
Split a set of results into ``N`` disjoint shards, so one pass over thousands of
results (``Aggregator.enrich``, an export) can be shared between Slurm array tasks, a
laptop process pool or anything else, with no coordination between the workers.

A result's shard depends only on its ``output_path`` (without ``.zip``), never on the
order results are found in, so every worker that sees the tree at the same path puts a
result in the same shard and the shards are disjoint. Run the workers of one pass on
one mount of the tree.
"""
import hashlib
import os
from typing import Optional, Tuple, Union

ShardLike = Union[None, str, Tuple[int, int]]


def parse_shard(shard: ShardLike) -> Optional[Tuple[int, int]]:
    """
    ``(i, n)`` from ``"i/N"`` or ``(i, N)`` (``None`` for no sharding).

    Raises
    ------
    ValueError
        Unless ``0 <= i < N``.
    """
    if shard is None:
        return None
    if isinstance(shard, str):
        try:
            i, n = (int(part) for part in shard.split("/"))
        except ValueError:
            raise ValueError(f"A shard is 'i/N' (e.g. '0/4'), not {shard!r}.")
    else:
        i, n = (int(part) for part in shard)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"Shard {i}/{n} is not one of 0/{n} .. {n - 1}/{n}.")
    return i, n


def shard_of(output_path: Union[str, os.PathLike], n: int) -> int:
    """
    Which of ``n`` shards the result at ``output_path`` (with or without ``.zip``)
    belongs to.
    """
    path = str(output_path)
    if path.endswith(".zip"):
        path = path[: -len(".zip")]
    return int(hashlib.sha1(path.encode("utf-8")).hexdigest(), 16) % n


def in_shard(output_path: Union[str, os.PathLike], shard: ShardLike) -> bool:
    """
    Whether the result at ``output_path`` is in ``shard`` (always, for ``None``).
    """
    shard = parse_shard(shard)
    if shard is None:
        return True
    i, n = shard
    return shard_of(output_path, n) == i
