import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__loads_from_config_file_if_not_input():
    search = af.Nautilus(
        n_live=500,
        n_batch=200,
        f_live=0.05,
        n_eff=1000,
        iterations_per_full_update=501,
        number_of_cores=4,
    )

    assert search.iterations_per_full_update == 501

    assert search.config_dict_search["n_live"] == 500
    assert search.config_dict_search["n_batch"] == 200
    assert search.config_dict_run["f_live"] == 0.05
    assert search.config_dict_run["n_eff"] == 1000
    assert search.number_of_cores == 4

    search = af.Nautilus()

    assert search.iterations_per_full_update == 1e99

    assert search.config_dict_search["n_live"] == 200
    assert search.config_dict_search["n_batch"] == 50
    assert search.config_dict_search["enlarge_per_dim"] == 1.2
    assert search.config_dict_search["split_threshold"] == 50
    assert search.config_dict_search["n_networks"] == 2
    assert search.config_dict_search["vectorized"] == False
    assert search.config_dict_run["f_live"] == 0.02
    assert search.config_dict_run["n_shell"] == 1
    assert search.config_dict_run["n_eff"] == 250
    assert search.config_dict_run["discard_exploration"] == False
    assert search.number_of_cores == 2


def test__identifier_fields():
    search = af.Nautilus()

    assert "n_live" in search.__identifier_fields__
    assert "n_eff" in search.__identifier_fields__
    assert "n_shell" in search.__identifier_fields__


def test__config_dict_test_mode():
    search = af.Nautilus()

    config_dict = {"n_like_max": float("inf"), "f_live": 0.01, "n_eff": 500}
    test_config = search.config_dict_test_mode_from(config_dict)

    assert test_config["n_like_max"] == 1
