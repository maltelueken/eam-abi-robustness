"""Tests for the test-case generators that replaced the per-experiment script directories."""

import numpy as np
import pytest
from cases import Case, meta_param_grid, num_obs_grid, subject_files


def test_num_obs_grid_keys_match_the_stored_filenames():
    cases = num_obs_grid(np.arange(50, 200, 50))

    assert [case.key for case in cases] == ["sample_size_50", "sample_size_100", "sample_size_150"]
    assert cases[0].labels == {"sample_size": 50}
    assert cases[0].sim_kwargs["num_obs"] == 50
    assert cases[0].is_simulated


def test_meta_param_grid_is_the_outer_product_in_row_major_order():
    cases = meta_param_grid("drift_slope_loc", [0.5, 1.0], "threshold_scale", [0.05, 0.1], num_obs=500)

    assert [case.key for case in cases] == [
        "drift_slope_loc_0.5_threshold_scale_0.05",
        "drift_slope_loc_0.5_threshold_scale_0.1",
        "drift_slope_loc_1.0_threshold_scale_0.05",
        "drift_slope_loc_1.0_threshold_scale_0.1",
    ]
    assert cases[0].labels == {"drift_slope_loc": 0.5, "threshold_scale": 0.05}
    assert cases[0].sim_kwargs["num_obs"] == 500


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
