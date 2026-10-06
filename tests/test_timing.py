"""Guards for the timing records the pipeline's expensive stages leave behind.

The numbers themselves cannot be asserted -- they are whatever the hardware did -- but their
schema and their durability can, and both are what makes a pile of these files usable: every
stage writes the same columns so the tables concatenate, and every row reaches the disk as it
is taken so a job killed at its walltime still accounts for what it finished.
"""

import csv
import pytest
from timing import FIELDS
from timing import TimingLog


def read(path):
    """Read a timing CSV back as a list of dicts."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_row_carries_the_run_labels_and_the_full_schema(tmp_path):
    log = TimingLog(tmp_path / "sub" / "timing.csv", {"experiment": "experiment_1", "model": "rdm_simple"})

    with log.timed("fit", case="sample_size_50", num_datasets=100, num_obs=50, num_chains=4):
        pass

    (row,) = read(log.path)

    assert list(row) == list(FIELDS)
    assert row["experiment"] == "experiment_1"
    assert row["task"] == "fit"
    assert row["case"] == "sample_size_50"
    assert row["num_chains"] == "4"
    # Columns the task says nothing about stay blank rather than absent, which is what lets
    # the tables of two stages be read as one.
    assert row["epochs"] == ""
    assert float(row["wall_seconds"]) >= 0
    assert float(row["cpu_seconds"]) >= 0


def test_rows_reach_the_disk_as_they_are_taken(tmp_path):
    """Rows land as they are taken.

    A cluster job killed at its walltime never gets to flush at the end, and its timings are
    exactly the ones worth having.
    """
    log = TimingLog(tmp_path / "timing.csv")

    with log.timed("fit", case="a"):
        assert not log.path.exists()

    assert [row["case"] for row in read(log.path)] == ["a"]

    with log.timed("fit", case="b"):
        pass

    # Appended under the one header, not restarted.
    assert [row["case"] for row in read(log.path)] == ["a", "b"]


def test_a_failed_block_is_still_timed(tmp_path):
    """The stage that crashed halfway still spent the time it spent getting there."""
    log = TimingLog(tmp_path / "timing.csv")

    with pytest.raises(RuntimeError), log.timed("fit", case="a"):
        raise RuntimeError

    assert [row["task"] for row in read(log.path)] == ["fit"]


def test_a_column_outside_the_schema_is_refused(tmp_path):
    """A typo would otherwise be written to a column nothing reads."""
    log = TimingLog(tmp_path / "timing.csv")

    with pytest.raises(ValueError, match="num_observations"), log.timed("fit", num_observations=50):
        pass


def test_an_unset_column_is_written_blank(tmp_path):
    """`predict_npe` passes `num_obs=None` for a case whose datasets do not share a trial count."""
    log = TimingLog(tmp_path / "timing.csv")

    with log.timed("sample", case="FF1", num_datasets=12, num_obs=None):
        pass

    (row,) = read(log.path)

    assert row["num_datasets"] == "12"
    assert row["num_obs"] == ""
