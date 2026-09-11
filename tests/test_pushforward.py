"""The prior pushforward figure.

The figure itself cannot be asserted on, so these cover the two things about it that can go
silently wrong: which test cases reach which panel -- a case that shifts a prior hyperparameter
has to be outlined where a case that restricts the data must not be -- and that a figure builds
for every shape of model the studies compose, two-channel and three-channel `x` alike.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cases import accuracy_bins
from cases import num_obs_grid
from cases import prior_param_grid
from cases import rt_windows
from pushforward import CaseOverlay
from pushforward import case_bands
from pushforward import case_overlays
from pushforward import example_indices
from pushforward import prior_hyperparameters
from pushforward import prior_pushforward_figure
from pushforward import pushforward_num_obs

PARAM_NAMES = ["v_intercept", "v_slope", "s_true", "b", "t0"]


def visible_panels(figure):
    """The panels actually drawn -- a grid row is filled out with hidden axes."""
    return [axis for axis in figure.get_axes() if axis.get_visible()]


class FakePriorSimulator:
    """A prior whose drift slope moves with `drift_slope_loc`, as the real ones do."""

    def __init__(self):
        self.calls = []

    def sample(self, num_draws, drift_slope_loc=1.5, **_kwargs):
        """Draw `num_draws` parameter vectors, recording the hyperparameters it was given."""
        self.calls.append(drift_slope_loc)
        rng = np.random.default_rng(0)

        return {
            "v_intercept": rng.gamma(4.0, 0.25, num_draws),
            "v_slope": rng.normal(drift_slope_loc, 0.5, num_draws),
            "s_true": rng.gamma(12.0, 0.1, num_draws),
            "b": rng.gamma(8.0, 0.15, num_draws),
            "t0": rng.gamma(3.0, 0.1, num_draws),
        }


def make_draws(num_datasets=64, num_obs=50, num_channels=2, seed=0):
    """A simulator draw: a parameter per name, plus `x` of the shape the simulators return."""
    rng = np.random.default_rng(seed)

    draws = {name: rng.gamma(4.0, 0.3, (num_datasets, 1)) for name in PARAM_NAMES}
    data_x = np.empty((num_datasets, num_obs, num_channels))
    data_x[..., 0] = rng.gamma(3.0, 0.2, (num_datasets, num_obs))
    data_x[..., 1] = rng.binomial(1, rng.uniform(0.5, 1.0, (num_datasets, 1)), (num_datasets, num_obs))

    if num_channels > 2:
        data_x[..., 2] = np.tile([0, 1], num_obs // 2)

    draws["x"] = data_x

    return draws


def test_a_prior_shifting_case_is_told_apart_from_a_banded_one():
    """Studies 2 and 3 move the prior; studies 1, 5 and 6 move the design or the data."""
    shifted = prior_param_grid("drift_slope_loc", [0.7, 1.5], num_obs=100)
    binned = accuracy_bins([0.5, 0.6, 0.7], num_obs=100)

    assert prior_hyperparameters(shifted[0].sim_kwargs) == {"drift_slope_loc": 0.7}
    assert prior_hyperparameters(binned[0].sim_kwargs) == {}
    assert prior_hyperparameters(num_obs_grid([50])[0].sim_kwargs) == {}

    assert case_bands(binned[0].sim_kwargs) == {"acc": (0.5, 0.6)}
    assert case_bands(rt_windows([[0.15, 1.0]], num_obs=100)[0].sim_kwargs) == {"rt": (0.15, 1.0)}
    assert case_bands(shifted[0].sim_kwargs) == {}


def test_only_the_prior_shifting_cases_are_redrawn():
    """A band restricts the data, so there is no shifted prior to outline -- and none is drawn."""
    prior_simulator = FakePriorSimulator()

    overlays = case_overlays(prior_simulator, accuracy_bins([0.5, 0.6, 0.7], num_obs=100), 32)

    assert overlays == []
    assert prior_simulator.calls == []

    overlays = case_overlays(prior_simulator, prior_param_grid("drift_slope_loc", [0.7, 3.9], 100), 32)

    assert [round(float(call), 3) for call in prior_simulator.calls] == [0.7, 3.9]
    assert [overlay.label for overlay in overlays] == ["drift_slope_loc = 0.7", "drift_slope_loc = 3.9"]
    # The case's value is pushed through the prior, so the parameter it governs moves with it.
    assert overlays[0].draw["v_slope"].mean() < overlays[1].draw["v_slope"].mean()


def test_a_hyperparameter_the_model_samples_is_pinned_rather_than_redrawn():
    """A hierarchical model draws the hyperparameter its test cases fix.

    The prior cannot supply a distribution for it -- the case makes it a point mass -- so the
    figure needs a line, and `case_overlays` says so only when the parameter is one the model
    actually has.
    """
    cases = prior_param_grid("drift_slope_loc", [0.7], num_obs=100)

    hierarchical = case_overlays(FakePriorSimulator(), cases, 32, ["drift_slope_loc", *PARAM_NAMES])
    flat = case_overlays(FakePriorSimulator(), cases, 32, PARAM_NAMES)

    assert hierarchical[0].pins == {"drift_slope_loc": 0.7}
    assert flat[0].pins == {}
    # Either way the parameters the prior does draw are outlined.
    assert "v_slope" in hierarchical[0].draw


def test_the_examples_span_the_pushforward_and_do_not_move_between_runs():
    """Evenly spaced quantiles of accuracy, so a rerun of the same draw picks the same datasets."""
    draws = make_draws()

    chosen = example_indices(draws["x"], 6)

    assert len(chosen) == 6
    assert len(set(chosen.tolist())) == 6
    np.testing.assert_array_equal(chosen, example_indices(draws["x"], 6))

    rates = np.mean(draws["x"][..., 1], axis=1)
    assert np.all(np.diff(rates[chosen]) >= 0)
    assert rates[chosen][0] == rates.min()
    assert rates[chosen][-1] == rates.max()


def test_fewer_datasets_than_examples_asked_for_is_not_an_error():
    """A narrow band can leave a small draw; the figure shows what there is."""
    assert len(example_indices(make_draws(num_datasets=3)["x"], 6)) == 3


@pytest.mark.parametrize("num_channels", [2, 3])
def test_the_figure_builds_for_both_model_families(num_channels):
    """Two channels for the simple models, three for the speed/accuracy ones."""
    figure = prior_pushforward_figure(make_draws(num_channels=num_channels), PARAM_NAMES, num_examples=4)

    # One panel per parameter, three pushforward statistics, one per example.
    assert len(visible_panels(figure)) == len(PARAM_NAMES) + 3 + 4


def test_the_figure_builds_with_overlays_and_bands():
    """The study-2 and study-5 decorations, on one figure each panel type has to survive."""
    param_names = ["drift_slope_loc", *PARAM_NAMES]
    draws = make_draws()
    draws["drift_slope_loc"] = np.random.default_rng(1).uniform(0.7, 3.9, (draws["x"].shape[0], 1))

    overlay = CaseOverlay(
        label="drift_slope_loc = 3.9",
        draw=FakePriorSimulator().sample(64, drift_slope_loc=3.9),
        pins={"drift_slope_loc": 3.9},
    )

    figure = prior_pushforward_figure(
        draws,
        param_names,
        num_examples=2,
        overlays=[overlay],
        trained_bands={"acc": (0.5, 0.6)},
        test_bands=[{"acc": (0.6, 0.7)}, {"rt": (0.15, 20.0)}],
        title="experiment_5 / rdm_simple_meta_acc_low",
    )

    assert len(visible_panels(figure)) == len(param_names) + 3 + 2


def test_the_trial_count_falls_back_to_what_the_study_tests_at():
    """Only study 1 has no `test_num_obs`, because the trial count is what it sweeps."""
    assert pushforward_num_obs({"pushforward_num_obs": 250, "test_num_obs": 500, "eval_num_obs": 1000}) == 250
    assert pushforward_num_obs({"pushforward_num_obs": None, "test_num_obs": 500, "eval_num_obs": 1000}) == 500
    assert pushforward_num_obs({"pushforward_num_obs": None, "eval_num_obs": 1000}) == 1000
