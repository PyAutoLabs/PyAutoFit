import pytest

from autofit.campaign import units


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("04:00:00", 14400),
        ("1-21:10:24", 162624),
        ("2-00:00:00", 172800),
        ("2-12", 216000),
        ("2-12:30", 217800),
        ("10:30", 630),
        ("90", 5400),
        ("UNLIMITED", None),
        ("", None),
        ("Partition_Limit", None),
    ],
)
def test_parse_time(text, seconds):
    assert units.parse_time(text) == seconds


def test_parse_time_rejects_garbage():
    with pytest.raises(ValueError):
        units.parse_time("soon")


def test_format_time_round_trip():
    assert units.format_time(14400) == "04:00:00"
    assert units.format_time(172800) == "2-00:00:00"
    assert units.format_time(59.2) == "00:01:00"
    assert units.parse_time(units.format_time(162624)) == 162624


@pytest.mark.parametrize(
    "text, mb",
    [
        ("98304M", 98304),
        ("29823.80M", 29823.8),
        ("12G", 12288),
        ("12288", 12288),
        ("3072Mc", 3072),
        ("1024K", 1),
        ("", None),
    ],
)
def test_parse_memory(text, mb):
    assert units.parse_memory_mb(text) == pytest.approx(mb) if mb else units.parse_memory_mb(text) is None


def test_memory_per():
    assert units.memory_per("3072Mc") == "cpu"
    assert units.memory_per("98304Mn") == "node"
    assert units.memory_per("98304M") is None


def test_array_spec_round_trip():
    spec = units.array_spec([0, 1, 2, 3, 7, 9, 10], throttle=400)
    assert spec == "0-3,7,9-10%400"
    assert units.parse_array_spec(spec) == ([0, 1, 2, 3, 7, 9, 10], 400)
    assert units.parse_array_spec("0-6:2") == ([0, 2, 4, 6], None)


@pytest.mark.parametrize(
    "gres, expected",
    [
        ("gpu:1", True),
        ("gpu:a100:2", True),
        ("billing=8,cpu=8,gres/gpu=1,mem=1G", True),
        ("gres/gpu:a100=1", True),
        ("billing=8,cpu=8,mem=98304M,node=1", False),
        ("gpu:0", True),
        ("gres/gpu=0", False),
        (None, False),
        ("", False),
    ],
)
def test_requests_gpu(gres, expected):
    assert units.requests_gpu(gres) is expected
