"""Tests for the vmapped simulator path that replaced BayesFlow's per-dataset Python loop.

The point of the batched entry points is that they are the *same function* as the
single-dataset ones, only mapped: `bayesflow.utils.batched_call` cost ~7 ms per dataset in
dispatch overhead alone, which made study 5's rejection sampling (up to 17 draws per accepted
dataset) unaffordable. So the load-bearing test is per-dataset equivalence, with distinct
parameters per dataset -- with identical ones, a swapped or mis-broadcast argument would pass.
"""

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lba_jax import (
    lba_experiment_sat_jax,
    lba_experiment_sat_jax_batched,
    lba_experiment_simple_jax,
    lba_experiment_simple_jax_batched,
)
from rdm_jax import (
    SplittableKey,
    rdm_experiment_sat_jax,
    rdm_experiment_sat_jax_batched,
    rdm_experiment_simple_jax,
    rdm_experiment_simple_jax_batched,
    static_num_obs,
)

import jax

SIMULATORS = [
    ("rdm_simple", rdm_experiment_simple_jax, rdm_experiment_simple_jax_batched,
     ["v_intercept", "v_slope", "s_true", "s_false", "b", "t0"]),
    ("rdm_sat", rdm_experiment_sat_jax, rdm_experiment_sat_jax_batched,
     ["v_intercept", "v_slope", "s_true", "s_false", "b", "b_diff", "t0"]),
    ("lba_simple", lba_experiment_simple_jax, lba_experiment_simple_jax_batched,
     ["v_intercept", "v_slope", "s_true", "s_false", "A", "B", "t0"]),
    ("lba_sat", lba_experiment_sat_jax, lba_experiment_sat_jax_batched,
     ["v_intercept", "v_slope", "s_true", "s_false", "A", "B", "B_diff", "t0"]),
]


class FixedKeys:
    """A `SplittableKey` stand-in handing out a known key per dataset, so the two paths align."""

    def __init__(self, keys):
        self.keys = keys

    def next(self, num=None):
        return self.keys


@pytest.mark.parametrize(("name", "single_fn", "batched_fn", "order"), SIMULATORS, ids=[s[0] for s in SIMULATORS])
def test_batched_path_reproduces_the_single_dataset_path(name, single_fn, batched_fn, order):
    rng = np.random.default_rng(0)
    num_datasets, num_obs = 7, 40
    keys = jax.random.split(jax.random.PRNGKey(3), num_datasets)

    # Distinct per dataset, and shaped (n, 1) as the priors deliver them.
    values = {
        key: (1 if key == "s_false" else rng.uniform(0.4, 1.6, size=(num_datasets, 1)))
        for key in order
    }

    batched = batched_fn(num_datasets, num_obs=np.array(num_obs), rng=FixedKeys(keys), **values)["x"]
    expected = np.stack([
        np.asarray(
            single_fn(
                keys[index],
                *[1.0 if key == "s_false" else float(np.reshape(values[key], -1)[index]) for key in order],
                num_obs,
            )["x"],
        )
        for index in range(num_datasets)
    ])

    # The RDM agrees bit-for-bit; the LBA to a couple of ULP, because vmap reassociates its
    # arithmetic. Either way this is the same function, not merely the same distribution.
    assert np.allclose(batched, expected, rtol=1e-12, atol=0.0)


def test_static_num_obs_accepts_one_value_in_any_wrapper():
    assert static_num_obs(500) == static_num_obs(np.array(500)) == static_num_obs(np.int64(500)) == 500


def test_static_num_obs_rejects_a_per_dataset_trial_count():
    # num_obs sets the JAX sample shape, so it cannot vary within a batch. Failing here beats
    # failing inside a vmap trace with an unrelated-looking shape error.
    with pytest.raises(ValueError, match="single value shared by the whole batch"):
        static_num_obs(np.array([50, 60]))


def test_splittable_key_gives_independent_keys_and_advances():
    key = SplittableKey(0)

    first = np.asarray(key.next(4))
    second = np.asarray(key.next(4))

    assert first.shape[0] == 4
    assert len({tuple(row) for row in np.concatenate([first, second])}) == 8


def test_splittable_key_still_supports_a_single_key():
    key = SplittableKey(0)

    assert not np.array_equal(np.asarray(key.next()), np.asarray(key.next()))


def build(overrides):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.mark.parametrize(
    ("model", "channels"),
    [("rdm_simple", 2), ("rdm_simple_meta", 2), ("lba_simple", 2), ("lba_simple_meta", 2),
     ("rdm_sat", 3), ("lba_sat_meta", 3)],
)
def test_batched_simulators_keep_the_shapes_the_rest_of_the_pipeline_expects(model, channels):
    """The adapter and `data.save_dataset` are shape-sensitive, so this pins the contract.

    Per-parameter draws must stay `(batch, 1)` -- `CustomSimulator` lifts the priors' `(batch,)`
    to that -- and `num_obs` must stay 0-d, one value for the whole batch.
    """
    simulator = instantiate(build([f"model={model}"])["simulator"], _convert_="partial")

    data = simulator.sample(8, num_obs=np.array(20))

    assert np.shape(data["x"]) == (8, 20, channels)
    assert np.ndim(data["num_obs"]) == 0
    assert all(np.shape(value) == (8, 1) for key, value in data.items() if key not in ("num_obs", "x"))


def test_batched_priors_honour_a_per_draw_hyperparameter():
    """The genuinely new behaviour: hierarchical models pass one hyperparameter value per draw.

    Previously each draw was a separate call with a scalar, so a broadcasting mistake had
    nowhere to hide. Now `threshold_scale` arrives as a `(batch,)` array and must line up
    element-wise with the draws it produces.
    """
    simulator = instantiate(build(["model=rdm_simple"])["simulator"], _convert_="partial")

    size = 20_000
    scale = np.concatenate([np.full(size // 2, 0.05), np.full(size // 2, 0.25)])

    draws = np.reshape(simulator.prior_simulator.sample((size,), threshold_scale=scale)["b"], -1)

    # b ~ Gamma(shape=8, scale), so the halves must separate at 8 * 0.05 and 8 * 0.25.
    assert np.isclose(draws[: size // 2].mean(), 0.4, rtol=0.05)
    assert np.isclose(draws[size // 2 :].mean(), 2.0, rtol=0.05)
