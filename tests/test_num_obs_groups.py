"""Tests for the trial-count grouping that lets a case hold ragged datasets.

Study 4's subjects each keep their own number of trials, but the summary network and the NUTS
driver both take one rectangular batch. `pipeline.num_obs_groups` splits a case into blocks
that share a trial count and `restore_dataset_order` puts the per-block results back; between
them they are what makes a ragged case work, so both the slicing and the reordering are
guarded here. A simulated case has to come through untouched, since every other study is one.
"""

import numpy as np
import pytest
from pipeline import num_obs_groups, num_obs_per_dataset, restore_dataset_order


def ragged(num_obs):
    """A case's data with one dataset per entry of `num_obs`, NaN-padded to the longest."""
    num_obs = np.asarray(num_obs)
    x = np.full((num_obs.size, int(num_obs.max()), 2), np.nan)

    for index, count in enumerate(num_obs):
        # Channel 0 identifies the dataset, so a reordering that goes wrong is visible.
        x[index, :count, 0] = index
        x[index, :count, 1] = np.arange(count)

    return {"x": x, "num_obs": num_obs}


def test_a_simulated_case_is_one_unsliced_block():
    data = {"x": np.zeros((7, 25, 2)), "num_obs": np.array(25)}

    blocks = num_obs_groups(data)

    assert len(blocks) == 1
    index, block = blocks[0]
    assert list(index) == list(range(7))
    assert block is data


def test_blocks_are_sliced_to_their_own_trial_count():
    blocks = num_obs_groups(ragged([10, 4, 10, 7]))

    assert [int(block["num_obs"]) for _, block in blocks] == [4, 7, 10]
    assert [block["x"].shape for _, block in blocks] == [(1, 4, 2), (1, 7, 2), (2, 10, 2)]
    # The padding never leaves the grouping.
    assert not any(np.isnan(block["x"]).any() for _, block in blocks)


def test_every_dataset_lands_in_exactly_one_block():
    num_obs = [10, 4, 10, 7]
    blocks = num_obs_groups(ragged(num_obs))

    assert sorted(np.concatenate([index for index, _ in blocks])) == list(range(len(num_obs)))


def test_results_come_back_in_the_case_s_dataset_order():
    blocks = num_obs_groups(ragged([10, 4, 10, 7]))

    # Stand-in for a stage's per-block output: one row per dataset, labelled with the dataset
    # it came from, exactly as channel 0 of `x` records it.
    per_block = [block["x"][:, 0, 0] for _, block in blocks]

    assert list(restore_dataset_order(blocks, per_block)) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"x": np.zeros((3, 25, 2)), "num_obs": np.array(25)}, [25, 25, 25]),
        (ragged([10, 4, 7]), [10, 4, 7]),
    ],
)
def test_trial_counts_read_the_same_way_for_both_kinds_of_case(data, expected):
    assert list(num_obs_per_dataset(data)) == expected


def test_trial_counts_can_be_restricted_to_the_converged_datasets():
    counts = num_obs_per_dataset(ragged([10, 4, 7]), np.array([True, False, True]))

    assert list(counts) == [10, 7]
