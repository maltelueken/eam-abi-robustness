"""Utilities to use the racing diffusion model as a simulator in BayesFlow."""

from typing import Callable

import bayesflow as bf
import numpy as np

from bayesflow.utils.decorators import allow_batch_size
from bayesflow.types import Shape


class AccuracyBandSimulator(bf.simulators.Simulator):
    """Base class adding optional rejection sampling on the accuracy of the generated data.

    Study 5 holds the prior fixed and manipulates the *data* instead: a training condition, or a
    test case, keeps only datasets whose empirical accuracy falls in a band. Channel 1 of `x` is
    1 when the correct accumulator won, so a dataset's accuracy is `mean(x[:, :, 1])`.

    The loop lives here rather than in a wrapper class because the rest of the pipeline reaches
    into the simulator: `src/metrics.py` reaches for `.experiment_simulator`, and study 5's
    band has to be visible to `sample()` itself. A wrapper would hide both.

    With no band configured `sample` is a single call to `_sample_once`, so studies 1-4 are
    unaffected.
    """

    accuracy_channel = 1
    _round_quantum = 256

    def __init__(
        self,
        acc_lower: float | None = None,
        acc_upper: float | None = None,
        max_rounds: int = 12,
        max_round_elements: int = 4_000_000,
        safety: float = 1.25,
    ):
        if (acc_lower is None) != (acc_upper is None):
            msg = "acc_lower and acc_upper must be given together."
            raise ValueError(msg)

        if acc_lower is not None and not 0.0 <= acc_lower < acc_upper <= 1.0:
            msg = f"Need 0 <= acc_lower < acc_upper <= 1, got [{acc_lower}, {acc_upper}]."
            raise ValueError(msg)

        self.acc_lower = acc_lower
        self.acc_upper = acc_upper
        self.max_rounds = max_rounds
        self.max_round_elements = max_round_elements
        self.safety = safety

        # Acceptance statistics persist across calls. Training makes ~50,000 `sample()` calls
        # against the same band, so after the first one the rate is known to three digits and
        # every later call can size a single round correctly instead of feeling its way up
        # from 256 again. This is the difference between ~36 and ~20 draws per accepted dataset.
        self._num_drawn = 0
        self._num_kept = 0

    def _sample_once(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """Draw a batch without regard to accuracy. Implemented by the subclasses."""
        raise NotImplementedError

    @allow_batch_size
    def sample(self, batch_shape: Shape, acc_lower=None, acc_upper=None, **kwargs) -> dict[str, np.ndarray]:
        """Draw `batch_shape` datasets, restricted to an accuracy band if one is in force.

        `acc_lower`/`acc_upper` are keyword-only and are consumed here rather than passed on: a
        test case's band must *replace* the band the approximator was trained on, and anything
        left in `kwargs` would be written into the saved dataset by `CustomSimulator`'s
        `design_dict.update(**kwargs)`.
        """
        lower = self.acc_lower if acc_lower is None else float(acc_lower)
        upper = self.acc_upper if acc_upper is None else float(acc_upper)

        if lower is None:
            return self._sample_once(batch_shape, **kwargs)

        return self._rejection_sample(batch_shape, lower, upper, **kwargs)

    def _rejection_sample(self, batch_shape: Shape, lower: float, upper: float, **kwargs):
        size = int(np.prod(batch_shape))
        rounds, num_accepted = [], 0
        num_obs, round_size = None, self._round_size(size, None)

        for _ in range(self.max_rounds):
            draw = self._sample_once((round_size,), **kwargs)

            if num_obs is None:
                # The design simulator draws one num_obs per batch, so a later round would
                # otherwise disagree with this one and `x` could not be concatenated.
                num_obs = draw["num_obs"]
                kwargs = {**kwargs, "num_obs": num_obs}

            accuracy = np.asarray(draw["x"])[..., self.accuracy_channel].mean(axis=-1)
            keep = in_accuracy_band(accuracy, lower, upper)

            self._num_drawn += round_size
            self._num_kept += int(keep.sum())

            if keep.any():
                rounds.append(_take_datasets(draw, keep, round_size))
                num_accepted += int(keep.sum())

                if num_accepted >= size:
                    return _concatenate_datasets(rounds, size)

            round_size = self._round_size(size - num_accepted, num_obs)

        msg = (
            f"Only {num_accepted} of {size} datasets had accuracy in [{lower}, {upper}] after "
            f"{self.max_rounds} rounds (acceptance rate {self._acceptance_rate():.4%} over "
            f"{self._num_drawn} draws) at num_obs={int(num_obs)}. Widen the band, raise max_rounds, "
            f"or check that the prior can reach this accuracy at all."
        )
        raise RuntimeError(msg)

    def _acceptance_rate(self):
        """Jeffreys-smoothed acceptance rate, so it is never zero and starts out pessimistic."""
        return (self._num_kept + 0.5) / (self._num_drawn + 1.0)

    def _round_size(self, needed, num_obs):
        """How many datasets to draw for `needed` accepted ones, quantized to a multiple of 256.

        The quantization is not cosmetic: `jax.jit` caches on the vmap axis size, so a freely
        adaptive round size would trigger a fresh XLA compilation nearly every round and undo
        the speedup the batched simulators exist for. Multiples of 256 keep the set of compiled
        shapes to a couple of dozen per trial count, which amortizes over a training run, while
        overshooting the ideal size by only a few percent (powers of two overshoot by up to 2x,
        which at a 6% acceptance rate is thousands of wasted simulations per batch).
        """
        want = int(np.ceil(needed / self._acceptance_rate() * self.safety))
        want = int(np.ceil(max(want, 1) / self._round_quantum)) * self._round_quantum

        if num_obs is not None:
            want = min(want, self.max_round_elements // max(int(num_obs), 1))

        return max(self._round_quantum, want)


def in_accuracy_band(accuracy, lower, upper):
    """Half-open `[lower, upper)`, closed at the top so an accuracy of exactly 1.0 is reachable.

    Accuracy is discrete (`k / num_obs`), so landing exactly on a bin edge is common enough to
    matter; half-open bins keep study 5's five test bins disjoint.
    """
    below_upper = accuracy <= upper if upper >= 1.0 else accuracy < upper

    return (accuracy >= lower) & below_upper


def _is_per_dataset(value, round_size):
    return np.ndim(value) > 0 and np.shape(value)[0] == round_size


def _take_datasets(draw, keep, round_size):
    return {key: (value[keep] if _is_per_dataset(value, round_size) else value) for key, value in draw.items()}


def _concatenate_datasets(rounds, size):
    """Stack the accepted rounds, leaving batch-level scalars such as `num_obs` alone.

    BayesFlow's own `Simulator.rejection_sample` cannot be reused here: it indexes *every* key
    along axis 0, which corrupts the scalar `num_obs`, and it stops at "at least `batch_shape`"
    rather than truncating, which would hand the trainer an oversized batch.
    """
    return {
        key: value if np.ndim(value) == 0 else np.concatenate([r[key] for r in rounds], axis=0)[:size]
        for key, value in rounds[0].items()
    }


class CustomSimulator(AccuracyBandSimulator):
    """Custom simulator for the racing diffusion model."""
    def __init__(
        self,
        prior_simulator: Callable,
        design_simulator: Callable,
        experiment_simulator: Callable,
        **band_kwargs,
    ):
        super().__init__(**band_kwargs)
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    def _sample_once(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        # `kwargs` goes to the prior as well as the design, so a test case can shift a prior
        # hyperparameter (`threshold_diff_scale` in study 3, `drift_slope_loc` /
        # `threshold_scale` in study 2) away from the value the approximator was trained on --
        # which is the whole point of those studies. `LambdaSimulator.sample` filters kwargs
        # against `sample_fn`'s signature, so unrelated ones (`num_obs`) are dropped here and
        # the hyperparameters are dropped by the experiment simulator below; a hyperparameter
        # passed at call time overrides the one bound into the `_partial_` in
        # `conf/simulator/prior_simulator/*.yaml`.
        prior_dict = self.prior_simulator.sample(batch_shape, **kwargs)

        design_dict = self.design_simulator.sample(batch_shape)

        design_dict.update(**kwargs)

        sims_dict = self.experiment_simulator.sample(
            batch_shape, **prior_dict, **design_dict
        )

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
            for key, value in data.items()
        }

        return data


class CustomMetaSimulator(AccuracyBandSimulator):
    """Custom simulator for the racing diffusion model with a varying prior."""
    def __init__(
        self,
        meta_simulator: Callable,
        prior_simulator: Callable,
        design_simulator: Callable,
        experiment_simulator: Callable,
        **band_kwargs,
    ):
        super().__init__(**band_kwargs)
        self.meta_simulator = meta_simulator
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    @staticmethod
    def update_existing(d: dict, **kwargs):
        """Update dictionary only with kwargs that match existing keys."""
        d.update((k, v) for k, v in kwargs.items() if k in d)

    def _sample_once(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        meta_dict = self.meta_simulator.sample(batch_shape)

        self.update_existing(meta_dict, **kwargs)

        prior_dict = self.prior_simulator.sample(batch_shape, **meta_dict)

        design_dict = self.design_simulator.sample(batch_shape)

        self.update_existing(design_dict, **kwargs)

        sims_dict = self.experiment_simulator.sample(
            batch_shape, **prior_dict, **design_dict
        )

        data = meta_dict | prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
            for key, value in data.items()
        }

        return data


def create_data_adapter(
    inference_variables, inference_conditions=None, summary_variables=None
):
    """Create an adapter for the racing diffusion model simulator."""
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=(-2, -1), squeeze=-1)
        .broadcast(
            inference_conditions, to="num_obs"
        )  # Make sure that all inference conditions have same shape
        .as_set(summary_variables)
        .log(inference_variables)
        .sqrt("num_obs")
        .concatenate(inference_variables, into="inference_variables")
        .concatenate(summary_variables, into="summary_variables")
        .rename("num_obs", "inference_conditions")
    )

    adapter = adapter.keep(
        ["inference_variables", "inference_conditions", "summary_variables"]
    )

    return adapter
