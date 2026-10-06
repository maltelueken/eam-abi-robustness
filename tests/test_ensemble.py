"""Guards for the properties that make an NPE ensemble measure inference variability.

`src/ensemble.py` explains why they matter; none of them is expressible in `conf/`, so they
are checked here. All of it is cheap: no member is ever trained, only built.
"""

import keras
import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from data import stack_posterior_members
from ensemble import create_ensemble_approximator, member_names, num_members, sample_members

CONFIG_PATH = "../conf"

ENSEMBLE_SIZE = 3


def build(overrides):
    """Compose the config with the Hydra runtime available (see `tests/test_config.py`)."""
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.fixture
def cfg():
    return build([
        "approximator=ensemble_approximator",
        f"approximator.ensemble_size={ENSEMBLE_SIZE}",
    ])


@pytest.fixture
def approximator(cfg):
    return instantiate(cfg["approximator"], _convert_="partial")


@pytest.fixture
def dataset(cfg, approximator):
    """The training stream `approximator.fit(simulator=...)` would build, two batches long."""
    return approximator.build_dataset(
        simulator=instantiate(cfg["simulator"], _convert_="partial"),
        adapter=approximator.adapter,
        num_batches=2,
        batch_size=4,
    )


def without_names(config):
    """Strip Keras' auto-generated layer names, which differ by construction order alone."""
    if isinstance(config, dict):
        return {key: without_names(value) for key, value in config.items() if key != "name"}

    if isinstance(config, list):
        return [without_names(value) for value in config]

    return config


def test_the_member_count_matches_the_configured_size(approximator):
    assert num_members(approximator) == ENSEMBLE_SIZE


def test_the_ensemble_config_instantiates_one_member_per_requested_size(approximator):
    assert approximator.members == tuple(member_names(ENSEMBLE_SIZE))


def test_members_are_configured_alike_down_to_the_subnet(approximator):
    """Equal architecture is the premise: the members must differ only by initialisation.

    Sharing an *instance* -- what `EnsembleWorkflow`'s size mode does with the summary network
    -- would collapse the spread to zero and, per BayesFlow's own warning, break serialisation
    of the checkpoint `scripts/predict_npe.py` reloads. Comparing the whole config, not just
    the top-level arguments, is what ties every member to the architecture the Optuna sweep
    selected, subnet widths included.
    """
    members = list(approximator.approximators.values())

    for attribute in ("inference_network", "summary_network"):
        networks = [getattr(member, attribute) for member in members]
        configs = [without_names(network.get_config()) for network in networks]

        assert len({id(network) for network in networks}) == ENSEMBLE_SIZE, attribute
        assert all(config == configs[0] for config in configs), attribute


def test_members_start_from_different_weights(approximator, dataset):
    """The other half of the premise: identical architecture, independent initialisation.

    Identical configs would be satisfied just as well by members that share their tensors, in
    which case every member would draw the same posterior and the spread would be zero.
    """
    approximator.build_from_data(keras.tree.map_structure(keras.ops.convert_to_tensor, dataset[0]))

    first, second = (approximator.approximators[name].trainable_variables for name in member_names(2))

    assert first
    assert [variable.shape for variable in first] == [variable.shape for variable in second]
    assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_every_member_reads_the_same_adapter(approximator):
    """`EnsembleApproximator` takes its adapter from whichever member comes first, so members
    disagreeing about it would silently apply one member's preprocessing to all of them."""
    adapters = {id(member.adapter) for member in approximator.approximators.values()}

    assert len(adapters) == 1


def test_the_members_train_on_identical_batches(dataset):
    """Perfect training data sharing (`data_reuse=1.0`), stated as what it means for the data.

    With anything less, members would see different draws from the training prior and their
    spread would mix simulation noise into the initialisation-and-optimisation variability the
    ensemble exists to isolate.
    """
    batch = dataset[0]

    for variable, per_member in batch.items():
        drawn = list(per_member.values())

        assert len(drawn) == ENSEMBLE_SIZE, variable
        assert all(np.array_equal(value, drawn[0]) for value in drawn[1:]), variable

    # Without this the check above would also pass on a dataset that returned one frozen batch
    # forever, which shares data perfectly and trains nothing.
    assert not np.array_equal(batch["summary_variables"]["0"], dataset[1]["summary_variables"]["0"])


def test_an_ensemble_needs_more_than_one_member():
    with pytest.raises(ValueError, match="at least two members"):
        create_ensemble_approximator(
            adapter=None,
            inference_network=None,
            summary_network=None,
            ensemble_size=1,
        )


class _SingleApproximator:
    """Stands in for a trained `ContinuousApproximator`, which cannot sample before it is fit."""

    def sample(self, *, conditions, num_samples):
        return {name: np.zeros((len(conditions["x"]), num_samples, 1)) for name in ("a", "b")}


def test_the_member_count_is_read_off_the_approximator():
    """`scripts/train_npe.py` records it next to the training time: an ensemble's fit buys this
    many networks for those seconds, and a single-network run has to report on the same scale."""
    assert num_members(_SingleApproximator()) == 1


def test_a_single_network_is_reported_as_a_one_member_ensemble():
    """`scripts/predict_npe.py` has no branch for the single-network case, so this is what keeps
    an ordinary run writing exactly the one-chain posterior it wrote before ensembles existed."""
    samples = sample_members(_SingleApproximator(), {"x": np.zeros((4, 20, 2))}, num_samples=7)

    assert list(samples) == ["0"]
    assert stack_posterior_members(samples, ["a", "b"]).shape == (1, 7, 4, 2)


def test_member_order_survives_the_trip_into_the_chain_axis():
    """`save_posterior` has no member axis, so member `i` is chain `i` by construction only --
    and `scripts/check_robustness.py` labels its rows with that index."""
    param_names = ["a", "b"]
    samples = {
        name: {param: np.full((4, 7, 1), float(index)) for param in param_names}
        for index, name in enumerate(member_names(ENSEMBLE_SIZE))
    }

    theta = stack_posterior_members(samples, param_names)

    assert theta.shape == (ENSEMBLE_SIZE, 7, 4, len(param_names))
    for index in range(ENSEMBLE_SIZE):
        assert np.all(theta[index] == index)
