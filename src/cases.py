"""Test-case enumeration -- the one axis on which the three studies actually differ.

Each study evaluates the NPE over a different set of held-out test cases: study 1 sweeps
the number of observations, study 2 a grid of two meta-parameters, study 4 the empirical
subject files. Everything downstream (simulating, predicting, fitting MCMC, comparing) is
identical, so that difference is captured here as a list of `Case` objects rather than by
duplicating each pipeline script once per study.

The generators below are Hydra `_target_`s under `conf/test_case/`; scripts iterate with

    for case in instantiate(cfg["test_case"]):
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Case:
    """One held-out test case.

    Attributes:
        key: filename-safe stem identifying the case, e.g. ``"sample_size_50"``. Every
            artifact for this case is named after it, so readers and writers agree without
            retyping an f-string.
        labels: identifying columns written into the result CSVs, e.g. ``{"sample_size": 50}``.
        sim_kwargs: keyword arguments passed to ``simulator.sample()`` for this case.
        source_path: path to an empirical data file, when the case is not simulated.
    """

    key: str
    labels: dict[str, Any] = field(default_factory=dict)
    sim_kwargs: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def is_simulated(self):
        """Whether this case's data comes from the simulator rather than a file."""
        return self.source_path is None


def num_obs_grid(values):
    """Sweep the number of observations per dataset (study 1)."""
    return [
        Case(
            key=f"sample_size_{value}",
            labels={"sample_size": value},
            sim_kwargs={"num_obs": np.array(value)},
        )
        for value in np.asarray(values).tolist()
    ]


def meta_param_grid(name_1, values_1, name_2, values_2, num_obs):
    """Sweep a grid of two meta-parameters at a fixed number of observations (study 2)."""
    return [
        Case(
            key=f"{name_1}_{value_1}_{name_2}_{value_2}",
            labels={name_1: value_1, name_2: value_2},
            sim_kwargs={
                name_1: np.array(value_1),
                name_2: np.array(value_2),
                "num_obs": np.array(num_obs),
            },
        )
        for value_1 in np.asarray(values_1).tolist()
        for value_2 in np.asarray(values_2).tolist()
    ]


def subject_files(directory, pattern="*.txt"):
    """Enumerate empirical data files (study 4).

    Sorted, unlike the `os.listdir` this replaces: the convergence masks that align MCMC
    and NPE samples depend on a stable case order, and `os.listdir` does not provide one.
    The pattern also stops stray files in the directory from being picked up as cases.
    """
    return [
        Case(
            key=os.path.splitext(os.path.basename(path))[0],
            labels={"name": os.path.splitext(os.path.basename(path))[0]},
            source_path=path,
        )
        for path in sorted(glob(os.path.join(directory, pattern)))
    ]


def prior_param_grid(name, values, num_obs):
    """Sweep one prior hyperparameter at a fixed number of observations (study 3).

    The one-dimensional counterpart of `meta_param_grid`: study 3 shifts a single
    hyperparameter -- the scale of the prior on the speed-accuracy threshold difference --
    away from the value the approximator was trained on.
    """
    return [
        Case(
            key=f"{name}_{value}",
            labels={name: value},
            sim_kwargs={name: np.array(value), "num_obs": np.array(num_obs)},
        )
        for value in np.asarray(values).tolist()
    ]


def accuracy_bins(edges, num_obs):
    """Bin the empirical accuracy of the generated data (study 5).

    Unlike studies 2 and 3 this does not move the prior: `sim_kwargs` overrides the *rejection
    band* the approximator was trained on, so every case draws from the same wide prior and
    differs only in which datasets are kept.

    That distinction is what makes the untouched MCMC the right reference. Accuracy is a
    deterministic function of `x`, so the selection indicator cancels out of
    `p(theta | x, accuracy in band) = p(theta | x)` -- there is no prior tilt to correct for,
    the same ground-truth fits serve every band, and what the study measures is amortization
    coverage rather than prior misspecification.
    """
    edges = np.asarray(edges).tolist()

    return [
        Case(
            key=f"accuracy_{round(lower * 100)}_{round(upper * 100)}",
            labels={"accuracy_lower": lower, "accuracy_upper": upper},
            sim_kwargs={"acc_lower": lower, "acc_upper": upper, "num_obs": np.array(num_obs)},
        )
        for lower, upper in zip(edges[:-1], edges[1:])
    ]


def rt_windows(windows, num_obs):
    """Window the *response times* of the generated data (study 6).

    The study-5 idea with a different statistic: `sim_kwargs` overrides the rejection window
    the approximator was trained on, so every case draws from the same wide prior and differs
    only in which datasets are kept -- here, those whose response times all fall in
    `[lower, upper]`. The windows are nested rather than disjoint, so a wide case's draw
    contains datasets a narrow one would also have accepted; the figures plot the mismatch
    against each dataset's realized slowest response time to recover a continuous axis.

    Selection again cancels out of `p(theta | x, all rt in window) = p(theta | x)`, so the
    untouched MCMC remains the right reference and one set of fits serves every window. That
    holds because whole datasets are *discarded*: trimming the offending trials instead --
    what an empirical response-time filter does -- would censor `x`, and the MCMC likelihood
    would then need a per-trial truncation term to match.
    """
    return [
        Case(
            key=f"rt_{round(lower * 1000)}_{round(upper * 1000)}",
            labels={"rt_lower": lower, "rt_upper": upper},
            sim_kwargs={"rt_lower": lower, "rt_upper": upper, "num_obs": np.array(num_obs)},
        )
        for lower, upper in np.asarray(windows).tolist()
    ]
