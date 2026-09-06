"""Smoke tests over the Hydra config graph.

Roughly half of `src/` is reachable only through `_target_` strings, so ruff, mypy and IDE
"unused symbol" checks cannot see it, and a renamed function shows up as a runtime failure
in a cluster job rather than at review time. These tests instantiate every combination
instead.
"""

import itertools
import os
import pathlib

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import get_param_names
from utils import get_checkpoint_path

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
        ("experiment_2", "drift_slope_loc_0.7"),
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


def test_the_empirical_study_reads_the_networks_study_2_trained():
    """Study 4 trains nothing -- it applies study 2's networks to the subject files.

    That reuse is the amortization claim the empirical study exists to test, and
    `hydra.job.chdir` puts every stage in its own `outputs/<experiment>/<model>/...` directory,
    so without `npe_path` study 4 would look for a checkpoint that no train_npe run ever wrote
    -- which is why `slurm/jobs.tsv` has no train_npe row for it.
    """
    empirical = get_checkpoint_path(build(["experiment=experiment_4", "model=lba_simple"]))

    assert empirical.endswith(
        os.path.join(
            "outputs", "experiment_2", "lba_simple", "flow_matching", "checkpoints", "model.keras",
        ),
    )
    assert os.path.isabs(empirical), empirical

    # Every other study reads what its own training run wrote, relative to that run's directory.
    simulated = get_checkpoint_path(build(["experiment=experiment_1", "model=rdm_simple"]))

    assert os.path.normpath(simulated) == os.path.join("checkpoints", "model.keras")


def test_approximator_instantiates_for_the_default_config():
    cfg = build([])

    assert instantiate(cfg["approximator"], _convert_="partial") is not None
    assert instantiate(cfg["optimizer"], _convert_="partial") is not None
    assert instantiate(cfg["callbacks"], _convert_="partial")


def test_the_experiments_train_an_ensemble_by_default():
    """Every stage must agree on the approximator without being told.

    `hydra.run.dir` interpolates `override_dirname`, so an `approximator=` override remembered
    at `train_npe` and forgotten at `predict_npe` would send the latter looking for a model in a
    directory that does not exist. Making the ensemble the default is what removes that
    opportunity -- and it also keeps runs at the paths they used before ensembles existed, which
    the run-dir assertion below pins.
    """
    cfg = build([])

    assert cfg["approximator"]["_target_"] == "ensemble.create_ensemble_approximator"
    assert cfg["hydra"]["run"]["dir"].endswith("flow_matching/")


@pytest.mark.parametrize("model", MODELS)
def test_every_model_composes_its_own_familys_architecture(model):
    """The architecture group is chosen by `model`, and `slurm/sweep.sh` names the sweep's Optuna
    study and database after the family by taking the model name's prefix in bash.

    Those two have to agree, or a family's sweep would write into `conf/architecture/<family>.yaml`
    while the models of that family read a different file -- an architecture selected for one
    forward model and silently never used, which no artifact would show.
    """
    cfg = build([f"experiment={EXPERIMENT_BY_MODEL.get(model, 'experiment_2')}", f"model={model}"])

    assert cfg["architecture_family"] == model.split("_")[0]


@pytest.mark.parametrize("family", ["rdm", "lba"])
def test_the_architecture_files_define_exactly_the_swept_keys(family):
    """`conf/architecture/<family>.yaml` is written by `scripts/select_architecture.py` from the
    best trial's parameters, so it holds one key per swept parameter and nothing else.

    A key the sweeper searches but the file omits leaves the network node interpolating a value
    that only exists during a sweep; a key the file holds but the sweeper does not search is one
    the sweep can never move, and the next selection run would silently drop it.
    """
    cfg = build([f"model={family}_simple", "sweeper=optuna", "approximator=continuous_approximator"])

    path = pathlib.Path(__file__).parent.parent / "conf" / "architecture" / f"{family}.yaml"
    written = set(OmegaConf.load(path)) - {"architecture_family"}

    assert written == set(cfg["hydra"]["sweeper"]["params"])


@pytest.mark.parametrize("family", ["rdm", "lba"])
def test_the_sweep_is_keyed_on_the_family_rather_than_the_model(family):
    """One sweep per forward model, not one per (experiment, model) pair.

    Every variant of a family differs only in a prior, a design or a data band and trains at the
    family's architecture, so `rdm_simple` and `rdm_sat` have to land in the same Optuna study --
    and the two families have to land in different ones, or the second sweep would resume the
    first and merge two search spaces into one Pareto front.
    """
    names = {
        build([f"model={model}", "sweeper=optuna", "approximator=continuous_approximator"])["hydra"]["sweeper"]
        for model in (f"{family}_simple", f"{family}_sat", f"{family}_simple_meta")
    }
    studies = {sweeper["study_name"] for sweeper in names}
    storages = {sweeper["storage"] for sweeper in names}

    assert len(studies) == 1
    assert len(storages) == 1
    assert family in studies.pop()
    assert storages.pop().endswith(f"train_npe_trials_{family}.db")

    other = "lba" if family == "rdm" else "rdm"
    other_sweeper = build([f"model={other}_simple", "sweeper=optuna", "approximator=continuous_approximator"])
    assert other_sweeper["hydra"]["sweeper"]["study_name"] not in studies


def test_the_architecture_sweep_can_still_select_a_single_network():
    """`slurm/sweep.sh` tunes one network rather than five: the members share an architecture,
    so searching it five at a time would cost five times as much to reach the same answer.

    The two must therefore stay interchangeable -- same adapter, so the same parameters, and
    both instantiable -- or what the sweep selects would not transfer to what the experiments
    train.
    """
    single = build(["sweeper=optuna", "approximator=continuous_approximator"])

    assert single["approximator"]["_target_"] == "bayesflow.approximators.ContinuousApproximator"
    assert instantiate(single["approximator"], _convert_="partial") is not None
    assert get_param_names(single) == get_param_names(build([]))

    # The sweep resumes an existing Optuna study by name, and every swept key has to exist on
    # the config the sweeper is actually applied to.
    assert single["hydra"]["sweeper"]["study_name"].startswith("train_npe-")
    for param in single["hydra"]["sweeper"]["params"]:
        assert OmegaConf.select(single, param) is not None, param


def test_the_training_diagnostics_are_scored_on_the_grid_the_networks_train_on():
    """`diag_num_obs` and the default design simulator's `values` are spelled out separately.

    The first is the axis `train_npe` averages its three Optuna objectives over, the second is
    the set of trial counts the networks actually see. Nothing but this ties them together, and
    they have drifted once already -- the objective used to be scored at a single
    `eval_num_obs` of 1000 -- with a silent consequence: the sweep selects an architecture at
    trial counts that no study reads the networks at.
    """
    cfg = build([])

    trained = list(cfg["simulator"]["design_simulator"]["sample_fn"]["values"])

    assert list(cfg["diag_num_obs"]) == trained


def test_the_default_training_grid_reaches_the_empirical_studys_trial_counts():
    """Study 4 trains nothing: it applies study 2's networks to subjects who run 93-100 trials.

    The adapter conditions on `sqrt(num_obs)`, so a grid starting at 250 leaves every empirical
    dataset far below the smallest set its network was ever given, and the amortization gap that
    study reports would be confounded with the sample-size extrapolation study 1 exists to
    measure. The grid therefore has to reach at least study 4's own `test_num_obs`.
    """
    trained = list(build([])["simulator"]["design_simulator"]["sample_fn"]["values"])
    empirical = build(["experiment=experiment_4", "model=rdm_simple"])["test_num_obs"]

    assert min(trained) <= empirical


@pytest.mark.parametrize("family", ["rdm", "lba"])
def test_study_1s_coverage_arms_differ_in_position_rather_than_density(family):
    """`discrete_lower` and `discrete_upper` are read against each other, so the only thing that
    should separate them is *where* they sit on the num_obs axis -- not how many trial counts
    they hold or how wide a span those cover. `discrete_full` is the superset both are read
    against, so it has to contain them."""
    arms = {
        arm: instantiate(
            build([f"model={family}_simple_discrete_{arm}"])["simulator"]["design_simulator"]["sample_fn"]["values"],
        )
        for arm in ("lower", "upper", "full")
    }

    assert len(arms["lower"]) == len(arms["upper"])
    assert np.ptp(arms["lower"]) == np.ptp(arms["upper"])
    assert max(arms["lower"]) < min(arms["upper"])

    assert set(arms["lower"]) <= set(arms["full"])
    assert set(arms["upper"]) <= set(arms["full"])
    assert set(build([])["simulator"]["design_simulator"]["sample_fn"]["values"]) <= set(arms["full"])


@pytest.mark.parametrize(
    "model",
    [f"{family}_simple_meta{suffix}" for family in ("rdm", "lba") for suffix in ("", "_lower", "_upper")],
)
def test_the_hierarchical_models_randomize_the_hyperparameter_the_test_cases_sweep(model):
    """Study 2's counterpart of the check below, and for the same reason.

    `conf/simulator/meta_simulator/random_prior_meta_continuous_multivariate*.yaml` spells the
    name out rather than interpolating `meta_param_name`, which lives in the test-case group --
    so nothing but this test stops the two from drifting apart, and a mismatch is silent:
    `CustomMetaSimulator.update_existing` would ignore the case's value and every point of the
    sweep would be drawn from the training prior.
    """
    cfg = build(["experiment=experiment_2", f"model={model}"])

    randomized = list(cfg["simulator"]["meta_simulator"]["sample_fn"]["name"])
    swept = cfg["test_case"]["name"]

    assert randomized == [swept]
    assert cfg["mcmc_param_names"][0] == swept


@pytest.mark.parametrize("family", ["rdm", "lba"])
def test_study_2_moves_the_drift_slope_prior_and_leaves_the_threshold_prior_alone(family):
    """Study 2 sweeps one hyperparameter, and only one.

    It used to cross `drift_slope_loc` with the threshold prior's scale. Sweeping a Gamma's scale
    moves its mean and its sd together (`sd = mean / sqrt(shape)`), so degradation along that axis
    could not be attributed to either -- and the study-4 posteriors put the empirical thresholds
    well inside the fixed prior anyway. This pins that the threshold prior is now the same for
    every case while the drift-slope prior actually moves, which is the premise of the study:
    without the kwarg forwarding in `CustomSimulator.sample` each case would be drawn from the
    training prior and merely *labelled* with a different hyperparameter.
    """
    cfg = build(["experiment=experiment_2", f"model={family}_simple"])
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    cases = instantiate(cfg["test_case"])
    threshold = "b" if family == "rdm" else "B"

    draws = [simulator.prior_simulator.sample((20_000,), **case.sim_kwargs) for case in (cases[0], cases[-1])]

    # The swept hyperparameter is a truncated-normal location, so the realized mean sits at or
    # above it; what matters is that the two ends are far apart and ordered.
    means = [float(np.mean(d["v_slope"])) for d in draws]
    assert means[0] < cases[0].labels["drift_slope_loc"] + 0.2
    assert means[1] > cases[-1].labels["drift_slope_loc"] - 0.2
    assert means[1] - means[0] > 2.0

    # The threshold prior is untouched by the sweep.
    thresholds = [float(np.mean(d[threshold])) for d in draws]
    assert np.isclose(thresholds[0], thresholds[1], rtol=0.05)
    shape = cfg["simulator"]["prior_simulator"]["sample_fn"]["threshold_shape"]
    scale = cfg["simulator"]["prior_simulator"]["sample_fn"]["threshold_scale"]
    assert np.isclose(thresholds[0], shape * scale, rtol=0.05)


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
