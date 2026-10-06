"""Utilities for hydra parameter sweeps."""

import numpy as np
import optuna


def constant_list(length, value):
    """Create a list repeating a single value."""
    return [value] * length


def normalize_objectives(
    values: np.ndarray,
    directions: list[optuna.study.StudyDirection],
) -> np.ndarray:
    """Map each objective of `values` onto [0, 1], 0 being best, over the trials given.

    The sweep optimizes RMSE, posterior contraction and calibration error together, and the three
    live on unrelated scales -- so they have to be put on a common one before any of them can be
    traded against another. A constant objective normalizes to 0 everywhere rather than dividing
    by a zero range: it separates no two trials, so it should not move the score either.

    Args:
        values: one row per trial, one column per objective.
        directions: the study's optimization direction per objective.

    Returns:
        An array shaped like `values`, with the maximized columns flipped so that 0 is best
        in every column.
    """
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    span = np.where(upper > lower, upper - lower, 1.0)

    normalized = (values - lower) / span
    maximized = np.array([d == optuna.study.StudyDirection.MAXIMIZE for d in directions])

    return np.where(maximized, 1.0 - normalized, normalized)


def select_best_trial(
    study: optuna.Study,
    weights: np.ndarray,
) -> tuple[optuna.trial.FrozenTrial, float]:
    """Return the Pareto-front trial closest to the ideal point, and its distance to it.

    A three-objective sweep has no single best trial, only a front of trials no other dominates.
    Picking one of them is a choice, and this is the scale-free one: normalize every objective
    over the completed trials (`normalize_objectives`), then take the front member at the
    smallest weighted Euclidean distance from the ideal point at the origin. Equal weights
    privilege no objective; `weights` is what a study that wants to would change.

    Raises:
        RuntimeError: if the study has no completed trials.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        msg = f"study '{study.study_name}' has no completed trials to select from"
        raise RuntimeError(msg)

    normalized = normalize_objectives(np.array([t.values for t in completed], dtype=float), study.directions)
    scores = np.sqrt(normalized**2 @ weights)

    # `best_trials` is the Pareto front; fall back to every completed trial for the degenerate
    # single-objective study, where the front is one point anyway.
    front = {t.number for t in study.best_trials} or {t.number for t in completed}
    winner = min((i for i, t in enumerate(completed) if t.number in front), key=lambda i: scores[i])

    return completed[winner], float(scores[winner])


def sweep_timing(study: optuna.Study) -> dict:
    """What the sweep cost, read back out of the study Optuna already timestamped.

    The sweep is the one stage that is not a run of a pipeline script, so it has nowhere to
    write a `timing.TimingLog`; it does not need one, because Optuna stamps every trial with a
    start and an end. Cost is reported two ways, and the gap between them is informative:
    `wall_seconds` spans the first trial's start to the last one's end -- what the SLURM job
    actually occupied -- while `trial_seconds` sums the trials, leaving out the sweeper's own
    overhead between them and counting concurrent trials twice if a study ever runs any.

    Failed trials are counted: a trial that died still spent the GPU time it spent, and the
    sweep had to pay for it. The per-trial durations themselves are in the study's trials
    dataframe, which `scripts/select_architecture.py` dumps next to this summary.

    Args:
        study: the study to summarize.

    Returns:
        A one-row summary, in column order.

    Raises:
        RuntimeError: if no trial of the study carries both timestamps.
    """
    timed = [t for t in study.trials if t.datetime_start is not None and t.datetime_complete is not None]

    if not timed:
        msg = f"study '{study.study_name}' has no trial with both a start and an end time"
        raise RuntimeError(msg)

    durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in timed]
    span = max(t.datetime_complete for t in timed) - min(t.datetime_start for t in timed)

    return {
        "study": study.study_name,
        "num_trials": len(study.trials),
        "num_timed_trials": len(timed),
        "wall_seconds": round(span.total_seconds(), 3),
        "trial_seconds": round(float(np.sum(durations)), 3),
        "median_trial_seconds": round(float(np.median(durations)), 3),
    }


def render_architecture(family: str, params: dict, provenance: list[str]) -> str:
    """Render the `conf/architecture/<family>.yaml` file that selects `params`.

    Written as text rather than dumped with OmegaConf so the file keeps the header explaining
    where it came from: it is checked in, and it is the only record of what the paper's networks
    were trained at once a sweep database has been archived.
    """
    header = "\n".join(f"# {line}" for line in provenance)
    body = "\n".join(f"{name}: {value!r}" for name, value in sorted(params.items()))

    return (
        "# @package _global_\n"
        "#\n"
        f"# The architecture every {family.upper()} model trains at.\n"
        "#\n"
        "# Written by scripts/select_architecture.py from the family's Optuna sweep; edit by hand\n"
        "# only to pin an architecture deliberately, since the next sweep of this family\n"
        "# overwrites the file. The keys are exactly the ones the sweeper searches over, and the\n"
        "# network nodes under conf/approximator/*_network/ interpolate them.\n"
        "#\n"
        f"{header}\n"
        f"architecture_family: {family}\n"
        "\n"
        f"{body}\n"
    )
