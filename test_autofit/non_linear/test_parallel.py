import autofit as af
from autofit.non_linear.parallel.sneaky import SneakyProcess

from pathlib import Path

class MockSneakyProcess(SneakyProcess):
    def __init__(
            self,
            paths,
    ):

        super().__init__(
            name="test",
            paths=paths
        )

def test__test_mode_parallel_profile_outputs_prof_files():

    paths = af.DirectoryPaths(
        path_prefix=str(Path("non_linear") / "parallel"),
    )

    process = MockSneakyProcess(paths=paths)

    # TODO : I dont know how to make it so run doesn't end up in an infinite loop?

#    process.run()