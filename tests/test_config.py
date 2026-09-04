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

EXPERIMENTS = [
    "experiment_1",
    "experiment_2",
    "experiment_3",
    "experiment_4",
    "experiment_5",
    "experiment_6",
]

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
    "rdm_sat",
    "rdm_sat_lower",
    "rdm_sat_upper",
    "rdm_sat_meta",
    "rdm_sat_meta_lower",
    "rdm_sat_meta_upper",
    "lba_sat",
    "lba_sat_lower",
    "lba_sat_upper",
    "lba_sat_meta",
    "lba_sat_meta_lower",
    "lba_sat_meta_upper",
    *[f"{family}_simple_meta_acc_{band}"
      for family in ("rdm", "lba")
      for band in ("low", "medium", "high", "full")],
    *[f"{family}_simple_meta_rt_{window}"
      for family in ("rdm", "lba")
      for window in ("tight", "medium", "wide", "full")],
]

# The models whose simulator and MCMC configs are only coherent under their own study's
# test-case group; the shared checks below compose them against it rather than experiment_2.
EXPERIMENT_BY_MODEL = {
    **{model: "experiment_3" for model in MODELS if "_sat" in model},
    **{model: "experiment_5" for model in MODELS if "_acc_" in model},
    **{model: "experiment_6" for model in MODELS if "_rt_" in model},
}


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
    cfg = build([f"experiment={EXPERIMENT_BY_MODEL.get(model, 'experiment_2')}", f"model={model}"])

    instantiate(cfg["simulator"], _convert_="partial")

    make_logdensity_fn = instantiate(cfg["mcmc_model_fun"])
    assert callable(make_logdensity_fn)

    # The MCMC initial position must have one entry per parameter the log-density expects;
    # a mismatch here used to surface only once a GPU job had already started.
    init_position = instantiate(cfg["mcmc_sampling_fun"]["init_position"])
    assert len(init_position) == len(cfg["mcmc_param_names"]), model


@pytest.mark.parametrize("model", MODELS)
def test_transform_pair_round_trips_on_the_configured_initial_position(model):
    cfg = build([f"experiment={EXPERIMENT_BY_MODEL.get(model, 'experiment_2')}", f"model={model}"])

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
        ("experiment_3", "threshold_diff_scale_0.02"),
        ("experiment_5", "accuracy_50_60"),
        ("experiment_6", "rt_150_1000"),
    ],
)
def test_test_case_groups_enumerate_the_expected_cases(experiment, expected_first_key):
    model = {
        "experiment_3": "rdm_sat",
        "experiment_5": "rdm_simple_meta_acc_full",
        "experiment_6": "rdm_simple_meta_rt_full",
    }.get(experiment, "rdm_simple")
    cfg = build([f"experiment={experiment}", f"model={model}"])

    cases = instantiate(cfg["test_case"])

    assert cases
    assert cases[0].key == expected_first_key


def test_approximator_instantiates_for_the_default_config():
    cfg = build([])

    assert instantiate(cfg["approximator"], _convert_="partial") is not None
    assert instantiate(cfg["optimizer"], _convert_="partial") is not None
    assert instantiate(cfg["callbacks"], _convert_="partial")


@pytest.mark.parametrize(
    "model",
    [f"{family}_sat_meta{suffix}" for family in ("rdm", "lba") for suffix in ("", "_lower", "_upper")],
)
def test_the_hierarchical_sat_models_randomize_the_hyperparameter_the_test_cases_sweep(model):
    """The name is spelled out in three files; a mismatch would be silent.

    `conf/simulator/meta_simulator/random_prior_meta_continuous_sat*.yaml` decides which
    hyperparameter is drawn per dataset, `conf/test_case/threshold_diff_grid.yaml` decides
    which one the test cases shift, and `conf/mcmc/rdm_sat_meta.yaml` names it in the stored
    parameter vector. If they disagree, `CustomMetaSimulator.update_existing` silently
    ignores the test case's value and every case is drawn from the training prior.
    """
    cfg = build(["experiment=experiment_3", f"model={model}"])

    randomized = list(cfg["simulator"]["meta_simulator"]["sample_fn"]["name"])
    swept = cfg["test_case"]["name"]

    assert randomized == [swept]
    assert cfg["mcmc_param_names"][0] == swept


def test_the_swept_hyperparameter_actually_shifts_the_prior_it_names():
    """Study 3's premise: the test prior must differ from the training prior.

    `CustomSimulator` forwards the test case's kwargs to the prior simulator, where they
    override the value bound in `conf/simulator/prior_simulator/rdm_sat.yaml`. Without that
    forwarding every case would be drawn from the training prior and merely *labelled* with a
    different hyperparameter -- a sweep that looks fine in every artifact but measures nothing.
    """
    cfg = build(["experiment=experiment_3", "model=rdm_sat"])
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    cases = instantiate(cfg["test_case"])
    means = [
        np.mean(simulator.sample(200, **(case.sim_kwargs | {"num_obs": np.array(10)}))["b_diff"])
        for case in [cases[0], cases[-1]]
    ]

    shape = cfg["simulator"]["prior_simulator"]["sample_fn"]["threshold_diff_shape"]
    expected = [shape * case.labels["threshold_diff_scale"] for case in [cases[0], cases[-1]]]

    assert np.allclose(means, expected, rtol=0.2)


@pytest.mark.parametrize("family", ["rdm", "lba"])
@pytest.mark.parametrize(
    ("experiment", "kind", "conditions"),
    [
        ("experiment_5", "acc", ("low", "medium", "high", "full")),
        ("experiment_6", "rt", ("tight", "medium", "wide", "full")),
    ],
)
def test_the_data_band_models_share_one_set_of_held_out_data(family, experiment, kind, conditions):
    """The four training conditions of studies 5 and 6 differ only in training, so their test
    data are identical.

    `test_data_path` normally carries the model name; these point it at the family instead, so
    one `generate_test_data` and one `fit_mcmc_gpu` run serves all four -- MCMC being by far the
    most expensive artifact in the pipeline. It is sound only because the band is a function of
    the data alone, which leaves the posterior the MCMC targets unchanged.
    """
    paths = {
        build([f"experiment={experiment}", f"model={family}_simple_meta_{kind}_{condition}"])["test_data_path"]
        for condition in conditions
    }

    assert len(paths) == 1
    assert paths.pop().endswith(f"{experiment}/{family}_{kind}")


def test_the_accuracy_bands_cover_the_test_bins_they_are_read_against():
    """The four training bands are spelled out in conf/simulator/*_meta_acc_*.yaml and the five
    test bins in conf/test_case/accuracy_bins.yaml. Nothing ties them together, deliberately --
    but three of the bands should coincide with a test bin, which is what makes each narrow
    model's inside-band and outside-band behaviour comparable."""
    edges = instantiate(build(["experiment=experiment_5", "model=rdm_simple_meta_acc_full"])["test_case"])
    bins = {(case.labels["accuracy_lower"], case.labels["accuracy_upper"]) for case in edges}

    bands = {
        band: (
            build(["experiment=experiment_5", f"model=rdm_simple_meta_acc_{band}"])["simulator"]["acc_lower"],
            build(["experiment=experiment_5", f"model=rdm_simple_meta_acc_{band}"])["simulator"]["acc_upper"],
        )
        for band in ("low", "medium", "high", "full")
    }

    assert bands["low"] in bins
    assert bands["medium"] in bins
    assert bands["high"] in bins
    assert bands["full"] == (min(l for l, _ in bins), max(u for _, u in bins))


def test_the_rt_windows_cover_the_test_windows_they_are_read_against():
    """Study 6's counterpart of the check above: the four training windows are spelled out in
    conf/simulator/*_meta_rt_*.yaml and the test windows in conf/test_case/rt_windows.yaml.
    Nothing ties them together, deliberately -- but every training window should also be a test
    case, which is what gives each model exactly one case it was trained on to read the other
    three against."""
    cases = instantiate(build(["experiment=experiment_6", "model=rdm_simple_meta_rt_full"])["test_case"])
    tested = {(case.labels["rt_lower"], case.labels["rt_upper"]) for case in cases}

    trained = {
        window: (
            build(["experiment=experiment_6", f"model=rdm_simple_meta_rt_{window}"])["simulator"]["rt_lower"],
            build(["experiment=experiment_6", f"model=rdm_simple_meta_rt_{window}"])["simulator"]["rt_upper"],
        )
        for window in ("tight", "medium", "wide", "full")
    }

    assert set(trained.values()) == tested

    # Nested, so the widest window contains every other one -- what lets the figures fall back
    # on each dataset's realized slowest response time as a continuous axis.
    assert trained["full"] == (min(lower for lower, _ in tested), max(upper for _, upper in tested))
