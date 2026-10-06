"""Prior hyperparameters for the tests, read from the configs the pipeline itself uses.

The MCMC factories take every hyperparameter of a model's prior (see `priors`), so a test
builds that set from the same `conf/simulator/prior_simulator/*.yaml` the simulator is
instantiated from, overriding only what the test deliberately varies.
"""

from pathlib import Path

import yaml

CONF = Path(__file__).resolve().parent.parent / "conf" / "simulator" / "prior_simulator"


def prior(name, **overrides):
    """The hyperparameters `conf/simulator/prior_simulator/<name>.yaml` binds, with `overrides` applied."""
    sample_fn = yaml.safe_load((CONF / f"{name}.yaml").read_text())["sample_fn"]
    hyperparameters = {key: value for key, value in sample_fn.items() if not key.startswith("_") and key != "rng"}

    unknown = set(overrides) - set(hyperparameters)
    assert not unknown, f"{name} has no hyperparameter(s) {sorted(unknown)}"

    return hyperparameters | overrides
