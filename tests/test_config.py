"""Smoke tests over the Hydra config graph.

Roughly half of `src/` is reachable only through `_target_` strings, so ruff, mypy and IDE
"unused symbol" checks cannot see it, and a renamed function shows up as a runtime failure
in a cluster job rather than at review time. These tests instantiate every combination
instead.
"""

import itertools

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

CONFIG_PATH = "../conf"

EXPERIMENTS = ["experiment_1", "experiment_2", "experiment_4"]

MODELS = [
    "rdm_simple",
    "rdm_simple_lower",
    "rdm_simple_upper",
    "rdm_simple_meta",
    "rdm_simple_meta_lower",
    "rdm_simple_meta_upper",
    "rdm_simple_discrete_lower",
    "rdm_simple_discrete_upper",
    "rdm_simple_discrete_full",
    "lba_simple",
    "lba_simple_lower",
    "lba_simple_upper",
    "lba_simple_meta",
    "lba_simple_meta_lower",
    "lba_simple_meta_upper",
    "lba_simple_discrete_lower",
    "lba_simple_discrete_upper",
    "lba_simple_discrete_full",
]


def build(overrides):
    """Compose a config with the Hydra runtime available.

    `test_data_path` and the `subject_files` test-case group interpolate `${hydra:...}`, so a
    bare `compose()` cannot resolve them.
    """
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.mark.parametrize(("experiment", "model"), itertools.product(EXPERIMENTS, MODELS))
def test_every_experiment_and_model_composes(experiment, model):
    cfg = build([f"experiment={experiment}", f"model={model}"])

    assert cfg["mcmc_param_names"]
    assert cfg["test_data_path"]


@pytest.mark.parametrize("model", MODELS)
def test_simulator_and_mcmc_instantiate_and_agree_on_parameter_count(model):
    cfg = build(["experiment=experiment_2", f"model={model}"])

    instantiate(cfg["simulator"], _convert_="partial")

    make_logdensity_fn = instantiate(cfg["mcmc_model_fun"])
    assert callable(make_logdensity_fn)

    # The MCMC initial position must have one entry per parameter the log-density expects;
    # a mismatch here used to surface only once a GPU job had already started.
    init_position = instantiate(cfg["mcmc_sampling_fun"]["init_position"])
    assert len(init_position) == len(cfg["mcmc_param_names"]), model


@pytest.mark.parametrize("model", MODELS)
def test_transform_pair_round_trips_on_the_configured_initial_position(model):
    cfg = build(["experiment=experiment_2", f"model={model}"])

    to_unconstrained = instantiate(cfg["mcmc_to_unconstrained"])
    to_constrained = instantiate(cfg["mcmc_to_constrained"])

    init_position = np.asarray(instantiate(cfg["mcmc_sampling_fun"]["init_position"]), dtype=float)

    round_tripped = np.asarray(to_constrained(to_unconstrained(init_position)))

    assert np.allclose(round_tripped, init_position, atol=1e-6), model


@pytest.mark.parametrize(
    ("experiment", "expected_first_key"),
    [
        ("experiment_1", "sample_size_50"),
        ("experiment_2", "drift_slope_loc_0.5_threshold_scale_0.05"),
    ],
)
def test_test_case_groups_enumerate_the_expected_cases(experiment, expected_first_key):
    cfg = build([f"experiment={experiment}", "model=rdm_simple"])

    cases = instantiate(cfg["test_case"])

    assert cases
    assert cases[0].key == expected_first_key


def test_approximator_instantiates_for_the_default_config():
    cfg = build([])

    assert instantiate(cfg["approximator"], _convert_="partial") is not None
    assert instantiate(cfg["optimizer"], _convert_="partial") is not None
    assert instantiate(cfg["callbacks"], _convert_="partial")
