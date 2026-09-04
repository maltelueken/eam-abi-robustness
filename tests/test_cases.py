"""Tests for the test-case generators that replaced the per-experiment script directories."""

import numpy as np
import pytest
from cases import Case, num_obs_grid, prior_param_grid, subject_files


def test_num_obs_grid_keys_match_the_stored_filenames():
    cases = num_obs_grid(np.arange(50, 200, 50))

    assert [case.key for case in cases] == ["sample_size_50", "sample_size_100", "sample_size_150"]
    assert cases[0].labels == {"sample_size": 50}
    assert cases[0].sim_kwargs["num_obs"] == 50
    assert cases[0].is_simulated


def test_prior_param_grid_sweeps_one_hyperparameter_at_a_fixed_sample_size():
    cases = prior_param_grid("threshold_diff_scale", [0.02, 0.1, 0.18], num_obs=500)

    assert [case.key for case in cases] == [
        "threshold_diff_scale_0.02",
        "threshold_diff_scale_0.1",
        "threshold_diff_scale_0.18",
    ]
    assert cases[0].labels == {"threshold_diff_scale": 0.02}
    # The swept hyperparameter is passed to `simulator.sample`, which forwards it to the
    # prior: this is what makes study 3's test prior differ from its training prior.
    assert cases[0].sim_kwargs == {"threshold_diff_scale": 0.02, "num_obs": 500}


def test_subject_files_are_sorted_and_filtered(tmp_path):
    # Deliberately created out of order: the convergence masks that align MCMC and NPE
    # samples rely on a stable case order, which os.listdir does not give.
    for name in ["FV1.txt", "FF1.txt", "FN1.txt", "notes.md"]:
        (tmp_path / name).write_text("")

    cases = subject_files(str(tmp_path))

    assert [case.key for case in cases] == ["FF1", "FN1", "FV1"]
    assert all(not case.is_simulated for case in cases)
    assert cases[0].source_path.endswith("FF1.txt")


def test_case_is_immutable():
    # Frozen so a script cannot accidentally rewrite a case's key mid-loop and desynchronise
    # the artifacts written for it. (Not hashable: labels and sim_kwargs are dicts.)
    case = Case(key="a")

    with pytest.raises(AttributeError):
        case.key = "b"
