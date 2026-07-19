import pytest

import autofit as af
from autonerves.conf import with_config
from autofit import SamplesNest, SearchOutput
from autofit.non_linear.samples.efficient import EfficientSamples


@pytest.fixture(name="samples")
def make_samples():
    return SamplesNest(
        model=af.Model(af.ex.Gaussian),
        sample_list=[],
        samples_info=None,
    )


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def test_search_output(output_directory, samples):
    paths = af.DirectoryPaths(
        "name",
        path_prefix=output_directory,
    )
    paths.save_samples(samples=samples)

    search_output = SearchOutput(paths.output_path)

    assert isinstance(search_output.samples, SamplesNest)


def test_efficient_samples_type(samples):
    efficient = EfficientSamples(samples=samples)
    assert isinstance(efficient.samples, SamplesNest)
