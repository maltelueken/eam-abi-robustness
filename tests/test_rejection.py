"""Tests for study 5's accuracy-band rejection sampling.

Study 5 keeps the prior fixed and restricts the *data* instead, so these guard the two things
that would silently invalidate it: that the band is actually enforced, and that a test case's
band replaces rather than intersects the one the approximator was trained on.
"""

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from simulation import CustomSimulator, in_accuracy_band


def build(model, experiment="experiment_5"):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment}", f"model={model}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        return cfg


def accuracy_of(data):
    return np.asarray(data["x"])[:, :, 1].mean(axis=1)


@pytest.mark.parametrize(
    ("model", "band"),
    [("rdm_simple_meta_acc_low", (0.5, 0.6)),
     ("rdm_simple_meta_acc_medium", (0.7, 0.8)),
     ("rdm_simple_meta_acc_high", (0.9, 1.0)),
     ("lba_simple_meta_acc_medium", (0.7, 0.8))],
)
def test_every_returned_dataset_falls_in_the_configured_band(model, band):
    simulator = instantiate(build(model)["simulator"], _convert_="partial")

    data = simulator.sample(24, num_obs=np.array(200))

    assert np.shape(data["x"])[0] == 24, "the batch must be filled exactly, not merely reached"
    assert in_accuracy_band(accuracy_of(data), *band).all()


def test_a_test_cases_band_replaces_the_trained_band():
    # The premise of sharing one set of held-out data across the four training conditions: the
    # case decides the bin, whichever model happens to generate it.
    simulator = instantiate(build("rdm_simple_meta_acc_high")["simulator"], _convert_="partial")

    data = simulator.sample(16, acc_lower=0.5, acc_upper=0.6, num_obs=np.array(200))

    assert in_accuracy_band(accuracy_of(data), 0.5, 0.6).all()


def test_the_batch_shares_one_trial_count_across_rejection_rounds():
    # num_obs is drawn once per batch, so rounds that disagreed on it could not be concatenated.
    simulator = instantiate(build("rdm_simple_meta_acc_low")["simulator"], _convert_="partial")

    data = simulator.sample(48)

    assert np.ndim(data["num_obs"]) == 0
    assert np.shape(data["x"])[1] == int(data["num_obs"])


def test_the_band_bounds_do_not_leak_into_the_saved_dataset():
    # `CustomSimulator._sample_once` puts leftover kwargs into the design dict, which is written
    # to NetCDF; the bounds are consumed by `sample` instead.
    simulator = instantiate(build("rdm_simple_meta_acc_full")["simulator"], _convert_="partial")

    data = simulator.sample(8, acc_lower=0.7, acc_upper=0.8, num_obs=np.array(100))

    assert "acc_lower" not in data
    assert "acc_upper" not in data


def test_an_unreachable_band_reports_the_realized_rate():
    simulator = instantiate(build("rdm_simple_meta_acc_full")["simulator"], _convert_="partial")
    simulator.max_rounds = 2

    with pytest.raises(RuntimeError, match="acceptance rate"):
        simulator.sample(8, acc_lower=0.0, acc_upper=0.01, num_obs=np.array(100))


def test_an_unbanded_simulator_draws_exactly_once():
    # The no-regression guard for studies 1-4: no band means no loop.
    simulator = instantiate(build("rdm_simple", experiment="experiment_1")["simulator"], _convert_="partial")
    calls = []

    original = simulator._sample_once

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    simulator._sample_once = counted
    simulator.sample(8, num_obs=np.array(20))

    assert len(calls) == 1


def test_bands_are_half_open_apart_from_the_last():
    # Accuracy is discrete (k / num_obs), so exact hits on a bin edge are common; the five test
    # bins must stay disjoint, and a perfect dataset must still land in 90-100.
    assert in_accuracy_band(np.array([0.6]), 0.6, 0.7)
    assert not in_accuracy_band(np.array([0.6]), 0.5, 0.6)
    assert in_accuracy_band(np.array([1.0]), 0.9, 1.0)


def test_an_incomplete_band_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must be given together"):
        CustomSimulator(prior_simulator=None, design_simulator=None, experiment_simulator=None, acc_lower=0.5)

    with pytest.raises(ValueError, match="acc_lower < acc_upper"):
        CustomSimulator(
            prior_simulator=None, design_simulator=None, experiment_simulator=None,
            acc_lower=0.8, acc_upper=0.6,
        )
