"""Tests for the empirical-data preprocessing.

This is the pipeline's highest-risk untested function: two separate 3xIQR trimming stages,
NaN padding of subjects that ran for different numbers of trials, and a `.sort("pp")` that
the convergence masks rely on to keep MCMC and NPE samples aligned per subject.
"""

import numpy as np
import polars as pl
import pytest
from utils import read_data_from_txt


def write_dataset(path, subjects, num_test_trials=40, rng_seed=0):
    """Write a space-separated file in the format the empirical study uses."""
    rng = np.random.default_rng(rng_seed)
    rows = []

    for pp, (rt_centre, accuracy) in subjects.items():
        # A couple of practice trials and a trl_number == 0 trial, all of which are dropped.
        rows.append({"pp": pp, "block": "practice", "trl_number": 0, "acc": 1, "RT": 900})
        rows.append({"pp": pp, "block": "test", "trl_number": 0, "acc": 1, "RT": 950})

        for trial in range(1, num_test_trials + 1):
            rows.append({
                "pp": pp,
                "block": "test",
                "trl_number": trial,
                "acc": int(rng.random() < accuracy),
                "RT": float(rt_centre + rng.normal(0, 50)),
            })

    pl.DataFrame(rows).write_csv(path, separator=" ")
    return path


def test_it_returns_one_entry_per_subject_with_rt_in_seconds(tmp_path):
    path = write_dataset(tmp_path / "data.txt", {1: (800, 0.9), 2: (850, 0.85), 3: (820, 0.88)})

    data = read_data_from_txt(str(path))

    assert data["x"].shape == (3, 40, 2)
    assert list(data["num_obs"]) == [40, 40, 40]
    # RT is divided by 1000, so ~800 ms becomes ~0.8 s.
    assert 0.5 < data["x"][:, :, 0].mean() < 1.2
    assert set(np.unique(data["x"][:, :, 1])) <= {0.0, 1.0}


def test_practice_and_zeroth_trials_are_excluded(tmp_path):
    path = write_dataset(tmp_path / "data.txt", {1: (800, 0.9), 2: (800, 0.9)}, num_test_trials=10)

    data = read_data_from_txt(str(path), trim=False)

    # 10 usable test trials per subject; the practice row and trl_number == 0 row are gone.
    assert list(data["num_obs"]) == [10, 10]


def test_subjects_keep_their_own_number_of_trials(tmp_path):
    rows = []
    for pp, num_trials in [(1, 30), (2, 12), (3, 25)]:
        for trial in range(1, num_trials + 1):
            rows.append({"pp": pp, "block": "test", "trl_number": trial, "acc": 1, "RT": 800.0})
    pl.DataFrame(rows).write_csv(tmp_path / "d.txt", separator=" ")

    data = read_data_from_txt(str(tmp_path / "d.txt"), trim=False)

    # No truncation to the shortest subject: everyone keeps every trial they ran.
    assert list(data["num_obs"]) == [30, 12, 25]
    assert data["x"].shape[1] == 30


def test_the_padding_is_the_tail_of_each_subject(tmp_path):
    rows = []
    for pp, num_trials in [(1, 30), (2, 12)]:
        for trial in range(1, num_trials + 1):
            rows.append({"pp": pp, "block": "test", "trl_number": trial, "acc": 1, "RT": 800.0})
    pl.DataFrame(rows).write_csv(tmp_path / "d.txt", separator=" ")

    data = read_data_from_txt(str(tmp_path / "d.txt"), trim=False)

    # Everything up to a subject's own count is real; everything past it is NaN, so a padded
    # trial that reached a likelihood or a summary network could not pass for a real one.
    for index, num_obs in enumerate(data["num_obs"]):
        assert not np.isnan(data["x"][index, :num_obs]).any()
        assert np.isnan(data["x"][index, num_obs:]).all()


def test_subjects_come_back_in_sorted_order(tmp_path):
    # Written in descending order on purpose: the MCMC/NPE alignment depends on the sort.
    rows = []
    for pp in [30, 10, 20]:
        for trial in range(1, 15):
            rows.append({"pp": pp, "block": "test", "trl_number": trial, "acc": 1, "RT": 700.0 + pp})
    pl.DataFrame(rows).write_csv(tmp_path / "d.txt", separator=" ")

    data = read_data_from_txt(str(tmp_path / "d.txt"), trim=False)

    # Subject 10's RTs are lowest, 30's highest, so a correct sort gives ascending means.
    means = data["x"][:, :, 0].mean(axis=1)
    assert list(means) == sorted(means)


def test_trimming_removes_an_extreme_outlier_subject(tmp_path):
    normal = {pp: (800, 0.9) for pp in range(1, 9)}
    path = write_dataset(tmp_path / "data.txt", normal)

    untrimmed = read_data_from_txt(str(path), trim=False)

    # Add a subject whose mean RT is far outside the others' interquartile range.
    frame = pl.read_csv(path, separator=" ")
    outlier = frame.filter(pl.col("pp") == 1).with_columns(
        pp=pl.lit(99, dtype=frame.schema["pp"]),
        RT=pl.col("RT") * 10,
    )
    pl.concat([frame, outlier]).write_csv(path, separator=" ")

    trimmed = read_data_from_txt(str(path), trim=True)

    assert trimmed["x"].shape[0] == untrimmed["x"].shape[0]  # the outlier subject is dropped


@pytest.mark.parametrize("trim", [True, False])
def test_output_feeds_the_approximator_shape_contract(tmp_path, trim):
    path = write_dataset(tmp_path / "data.txt", {pp: (800, 0.9) for pp in range(1, 6)})

    data = read_data_from_txt(str(path), trim=trim)

    assert set(data) == {"x", "num_obs"}
    assert data["x"].ndim == 3
    assert data["x"].shape[2] == 2
    assert data["num_obs"].shape == (data["x"].shape[0],)
