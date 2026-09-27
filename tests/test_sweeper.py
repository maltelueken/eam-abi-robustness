"""Tests for the architecture the sweep hands back to the experiments.

`scripts/select_architecture.py` writes `conf/architecture/<family>.yaml` straight from these
functions, and every network of that family is then trained at whatever they chose -- so the
rule that turns a three-objective Pareto front into one architecture is worth pinning. It is
also the one place in the pipeline where a silent mistake would not show up in any artifact:
a wrong pick still trains, still converges, and still produces every figure.
"""

import datetime

import numpy as np
import optuna
import pytest
from sweeper import constant_list
from sweeper import normalize_objectives
from sweeper import render_architecture
from sweeper import select_best_trial
from sweeper import sweep_timing

# The sweep's own directions: minimize RMSE, maximize posterior contraction, minimize
# calibration error.
DIRECTIONS = ["minimize", "maximize", "minimize"]

# `IntDistribution` in optuna 4, `IntUniformDistribution` in the 2.x that
# hydra-optuna-sweeper 1.2.0 pins. Nothing here depends on which is installed, and the
# cluster's database has been written by both.
INT = getattr(optuna.distributions, "IntDistribution", None) or optuna.distributions.IntUniformDistribution

SPACE = {"width": INT(1, 10)}


def make_study(values, name="test"):
    """Create an in-memory multi-objective study holding `values`, one row per trial."""
    study = optuna.create_study(study_name=name, directions=DIRECTIONS)

    for row in values:
        trial = study.ask(SPACE)

        if row is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, list(row))

    return study


def test_constant_list_repeats_its_value():
    assert constant_list(3, 8) == [8, 8, 8]


def test_normalization_puts_zero_at_the_best_end_of_every_objective():
    """The three objectives live on unrelated scales and two of three are minimized, so they
    have to be put on a common 0-is-best scale before any distance over them means anything."""
    values = np.array([[0.1, 0.9, 0.5], [0.5, 0.1, 0.1]])

    minimize, maximize = optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MAXIMIZE
    normalized = normalize_objectives(values, [minimize, maximize, minimize])

    # The minimized columns keep their order; the maximized one is flipped.
    assert normalized[0, 0] == 0.0
    assert normalized[1, 0] == 1.0
    assert normalized[0, 1] == 0.0
    assert normalized[1, 1] == 1.0
    assert normalized[1, 2] == 0.0


def test_a_constant_objective_does_not_divide_by_a_zero_range():
    """It separates no two trials, so it must not move the score -- or select one at random."""
    values = np.array([[0.1, 0.5, 0.2], [0.9, 0.5, 0.4]])

    normalized = normalize_objectives(values, [optuna.study.StudyDirection.MINIMIZE] * 3)

    assert np.all(np.isfinite(normalized))
    assert np.all(normalized[:, 1] == 0.0)


def test_the_selected_trial_is_the_one_nearest_the_ideal_point():
    """Trial 2 is worse than trial 0 on RMSE and worse than trial 1 on calibration, but it is
    the only one that is not far off on some objective -- which is what "closest to the ideal
    point" is for, and what a lexicographic rule would miss."""
    study = make_study([
        [0.00, 0.50, 1.00],  # best RMSE, worst calibration
        [1.00, 0.50, 0.00],  # worst RMSE, best calibration
        [0.40, 1.00, 0.40],  # nothing extreme, best contraction
    ])

    best, score = select_best_trial(study, np.ones(3))

    assert best.number == 2
    assert score == pytest.approx(np.sqrt(0.4**2 + 0.0 + 0.4**2))


def test_weights_can_privilege_one_objective():
    """The default is equal weights; a study that wants calibration to dominate says so here
    rather than by rewriting the rule."""
    study = make_study([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])

    assert select_best_trial(study, np.array([1.0, 1.0, 100.0]))[0].number == 1
    assert select_best_trial(study, np.array([100.0, 1.0, 1.0]))[0].number == 0


def test_a_dominated_trial_is_never_selected():
    """Trial 1 is beaten on every objective at once. It sits nearest the middle of the
    normalized cube, so a distance computed over all completed trials would pick it -- the
    restriction to the Pareto front is what stops that."""
    study = make_study([
        [0.00, 1.00, 0.00],  # dominates trial 1 outright
        [0.50, 0.50, 0.50],
        [1.00, 0.00, 1.00],
    ])

    best, _ = select_best_trial(study, np.ones(3))

    assert best.number == 0


def test_failed_trials_are_ignored():
    """A trial that crashed has no objective values, and an OOM or a diverging run is not rare
    at the edges of the search space."""
    study = make_study([[0.1, 0.9, 0.1], None, [0.9, 0.1, 0.9]])

    best, _ = select_best_trial(study, np.ones(3))

    assert best.number == 0


def test_a_study_with_nothing_completed_fails_loudly():
    """Silence here would write an architecture file from whatever the last sweep left behind."""
    study = make_study([None, None])

    with pytest.raises(RuntimeError, match="no completed trials"):
        select_best_trial(study, np.ones(3))


def test_the_rendered_file_is_a_global_package_naming_its_family():
    """`conf/architecture/<family>.yaml` is composed by every model of the family, so it has to
    carry the `_global_` package directive and the family key the sweeper's study name reads."""
    rendered = render_architecture("rdm", {"summary_dim": 24, "inference_mlp_depth": 5}, ["study: x"])

    assert rendered.startswith("# @package _global_")
    assert "# study: x" in rendered
    assert "architecture_family: rdm" in rendered
    assert "summary_dim: 24" in rendered
    assert "inference_mlp_depth: 5" in rendered


class _TimedTrial:
    """A trial with only the two timestamps `sweep_timing` reads."""

    def __init__(self, start_minutes, minutes):
        epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        self.datetime_start = epoch + datetime.timedelta(minutes=start_minutes)
        self.datetime_complete = None if minutes is None else self.datetime_start + datetime.timedelta(minutes=minutes)


class _TimedStudy:
    """Enough of a study for `sweep_timing`, which reads a name and a list of trials."""

    study_name = "train_npe_rdm"

    def __init__(self, trials):
        self.trials = trials


def test_sweep_timing_separates_the_job_from_the_trials():
    """The span is what the SLURM job occupied and the sum is what the trials themselves ran
    for; the gap between them is the sweeper's own overhead, which is why both are reported."""
    # Two 10-minute trials with a 5-minute gap between them: 25 minutes wall, 20 of trials.
    timing = sweep_timing(_TimedStudy([_TimedTrial(0, 10), _TimedTrial(15, 10)]))

    assert timing["study"] == "train_npe_rdm"
    assert timing["num_trials"] == 2
    assert timing["wall_seconds"] == 25 * 60
    assert timing["trial_seconds"] == 20 * 60
    assert timing["median_trial_seconds"] == 10 * 60


def test_a_crashed_trial_still_counts_towards_the_cost():
    """It spent the GPU time it spent -- but a trial still running has no end to measure to."""
    timing = sweep_timing(_TimedStudy([_TimedTrial(0, 10), _TimedTrial(10, 4), _TimedTrial(20, None)]))

    assert timing["num_trials"] == 3
    assert timing["num_timed_trials"] == 2
    assert timing["trial_seconds"] == 14 * 60


def test_a_study_with_no_finished_trial_has_no_timing():
    with pytest.raises(RuntimeError, match="start and an end"):
        sweep_timing(_TimedStudy([_TimedTrial(0, None)]))
