"""Utilities to use the racing diffusion model as a simulator in BayesFlow."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import bayesflow as bf
import numpy as np

from bayesflow.utils.decorators import allow_batch_size
from bayesflow.types import Shape


# Channels of `x`, as produced by `rdm_jax`/`lba_jax`: response time, then whether the correct
# accumulator won. `pipeline.py` reads the same two.
RT_CHANNEL = 0
ACCURACY_CHANNEL = 1


def in_accuracy_band(accuracy, lower, upper):
    """Half-open `[lower, upper)`, closed at the top so an accuracy of exactly 1.0 is reachable.

    Accuracy is discrete (`k / num_obs`), so landing exactly on a bin edge is common enough to
    matter; half-open bins keep study 5's five test bins disjoint.
    """
    below_upper = accuracy <= upper if upper >= 1.0 else accuracy < upper

    return (accuracy >= lower) & below_upper


def in_rt_window(rt, lower, upper):
    """Whether *every* response time of a dataset falls in `[lower, upper]`.

    Study 6's selection -- `min(rt) >= lower and max(rt) <= upper` -- is the dataset-level
    analogue of the response-time filter routinely applied to empirical data, but applied by
    *discarding* whole datasets rather than by trimming trials. Trimming would censor `x`
    itself and the MCMC reference would then need a truncated, renormalized likelihood;
    discarding leaves the likelihood alone (see `cases.rt_windows`).

    Closed at both ends, unlike the accuracy bins: response time is continuous, so an exact tie
    with an edge has probability zero, and study 6's windows are nested rather than disjoint,
    so there is nothing to keep apart.
    """
    return (np.min(rt, axis=-1) >= lower) & (np.max(rt, axis=-1) <= upper)


@dataclass(frozen=True)
class _Band:
    """One rejection criterion: what to measure per dataset, and what counts as passing."""

    statistic: Callable
    predicate: Callable
    max_upper: float


# The bands `DataBandSimulator.sample` understands, keyed by the prefix of their bounds.
_BANDS = {
    "acc": _Band(
        statistic=lambda x: np.mean(x[..., ACCURACY_CHANNEL], axis=-1),
        predicate=in_accuracy_band,
        max_upper=1.0,
    ),
    "rt": _Band(
        statistic=lambda x: x[..., RT_CHANNEL],
        predicate=in_rt_window,
        max_upper=np.inf,
    ),
}


def bands_from_bounds(**bounds):
    """Collect `{name: (lower, upper)}` from `<name>_lower`/`<name>_upper` keyword pairs.

    A name whose pair is absent is simply not in force; a half-given pair is an error rather
    than a silently unbanded simulator.
    """
    bands = {}

    for name, band in _BANDS.items():
        lower, upper = bounds.get(f"{name}_lower"), bounds.get(f"{name}_upper")

        if (lower is None) != (upper is None):
            msg = f"{name}_lower and {name}_upper must be given together."
            raise ValueError(msg)

        if lower is None:
            continue

        if not 0.0 <= lower < upper <= band.max_upper:
            msg = f"Need 0 <= {name}_lower < {name}_upper <= {band.max_upper}, got [{lower}, {upper}]."
            raise ValueError(msg)

        bands[name] = (float(lower), float(upper))

    return bands


def _passes_bands(x, bands):
    """Which datasets of a draw pass every band in force."""
    x = np.asarray(x)
    keep = np.ones(np.shape(x)[0], dtype=bool)

    for name, (lower, upper) in bands.items():
        band = _BANDS[name]
        keep &= band.predicate(band.statistic(x), lower, upper)

    return keep


def _describe(bands):
    return ", ".join(f"{name} in [{lower}, {upper}]" for name, (lower, upper) in bands.items())


class DataBandSimulator(bf.simulators.Simulator):
    """Base class adding optional rejection sampling on a statistic of the generated data.

    Studies 5 and 6 hold the prior fixed and manipulate the *data* instead: a training
    condition, or a test case, keeps only the datasets that pass a band. Two exist, each named
    by the prefix of its bounds:

    * `acc_lower`/`acc_upper` (study 5) -- the dataset's mean accuracy, channel 1 of `x`, must
      fall in the band.
    * `rt_lower`/`rt_upper` (study 6) -- *every* response time, channel 0 of `x`, must fall in
      the window, i.e. `min(rt) >= rt_lower and max(rt) <= rt_upper`.

    Both are functions of `x` alone, which is what makes the untouched MCMC the right
    reference for either study: the selection indicator cancels out of `p(theta | x, band)`
    (see `cases.accuracy_bins` and `cases.rt_windows`).

    The loop lives here rather than in a wrapper class because the rest of the pipeline reaches
    into the simulator: `src/metrics.py` reaches for `.experiment_simulator`, and the band has
    to be visible to `sample()` itself. A wrapper would hide both.

    With no band configured `sample` is a single call to `_sample_once`, so studies 1-4 are
    unaffected.
    """

    _round_quantum = 256

    def __init__(  # noqa: PLR0913
        self,
        acc_lower: float | None = None,
        acc_upper: float | None = None,
        rt_lower: float | None = None,
        rt_upper: float | None = None,
        max_rounds: int = 12,
        max_round_elements: int = 4_000_000,
        safety: float = 1.25,
    ):
        self.acc_lower, self.acc_upper = acc_lower, acc_upper
        self.rt_lower, self.rt_upper = rt_lower, rt_upper

        self.bands = bands_from_bounds(
            acc_lower=acc_lower, acc_upper=acc_upper, rt_lower=rt_lower, rt_upper=rt_upper,
        )

        self.max_rounds = max_rounds
        self.max_round_elements = max_round_elements
        self.safety = safety

        # Acceptance statistics persist across calls, keyed by the band in force. Training makes
        # ~50,000 `sample()` calls against the same band, so after the first one the rate is
        # known to three digits and every later call can size a single round correctly instead
        # of feeling its way up from 256 again -- the difference between ~36 and ~20 draws per
        # accepted dataset. Keying by band matters once one simulator sees several:
        # `generate_test_data.py` visits every test case in turn, and study 6's windows differ
        # in acceptance rate by a factor of 25, so one blended estimate would mis-size the first
        # round of every case. The trial count is deliberately not part of the key -- it moves
        # the rate by about 2x, which the within-call adaptation absorbs.
        self._stats = defaultdict(lambda: [0, 0])

    def _sample_once(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """Draw a batch without regard to any band. Implemented by the subclasses."""
        raise NotImplementedError

    @allow_batch_size
    def sample(
        self,
        batch_shape: Shape,
        acc_lower=None,
        acc_upper=None,
        rt_lower=None,
        rt_upper=None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Draw `batch_shape` datasets, restricted to whatever bands are in force.

        The bounds are keyword-only and are consumed here rather than passed on: a test case's
        band must *replace* the band the approximator was trained on -- all of them, not only
        the one the case names, or one case would mean different things to two models and a
        family's shared held-out data would be a fiction -- and anything left in `kwargs` would
        be written into the saved dataset by `CustomSimulator`'s `design_dict.update(**kwargs)`.
        """
        requested = bands_from_bounds(
            acc_lower=acc_lower, acc_upper=acc_upper, rt_lower=rt_lower, rt_upper=rt_upper,
        )

        bands = requested or self.bands

        if not bands:
            return self._sample_once(batch_shape, **kwargs)

        return self._rejection_sample(batch_shape, bands, **kwargs)

    def _rejection_sample(self, batch_shape: Shape, bands: dict, **kwargs):
        size = int(np.prod(batch_shape))
        stats = self._stats[tuple(sorted(bands.items()))]
        rounds, num_accepted = [], 0
        num_obs, round_size = None, self._round_size(size, None, stats)

        for _ in range(self.max_rounds):
            draw = self._sample_once((round_size,), **kwargs)

            if num_obs is None:
                # The design simulator draws one num_obs per batch, so a later round would
                # otherwise disagree with this one and `x` could not be concatenated.
                num_obs = draw["num_obs"]
                kwargs = {**kwargs, "num_obs": num_obs}

            keep = _passes_bands(draw["x"], bands)

            stats[0] += round_size
            stats[1] += int(keep.sum())

            if keep.any():
                rounds.append(_take_datasets(draw, keep, round_size))
                num_accepted += int(keep.sum())

                if num_accepted >= size:
                    return _concatenate_datasets(rounds, size)

            round_size = self._round_size(size - num_accepted, num_obs, stats)

        msg = (
            f"Only {num_accepted} of {size} datasets passed {_describe(bands)} after "
            f"{self.max_rounds} rounds (acceptance rate {_acceptance_rate(stats):.4%} over "
            f"{stats[0]} draws) at num_obs={int(num_obs)}. Widen the band, raise max_rounds, "
            f"or check that the prior can reach this region at all."
        )
        raise RuntimeError(msg)

    def _round_size(self, needed, num_obs, stats):
        """How many datasets to draw for `needed` accepted ones, quantized to a multiple of 256.

        The quantization is not cosmetic: `jax.jit` caches on the vmap axis size, so a freely
        adaptive round size would trigger a fresh XLA compilation nearly every round and undo
        the speedup the batched simulators exist for. Multiples of 256 keep the set of compiled
        shapes to a couple of dozen per trial count, which amortizes over a training run, while
        overshooting the ideal size by only a few percent (powers of two overshoot by up to 2x,
        which at a 6% acceptance rate is thousands of wasted simulations per batch).
        """
        want = int(np.ceil(needed / _acceptance_rate(stats) * self.safety))
        want = int(np.ceil(max(want, 1) / self._round_quantum)) * self._round_quantum

        if num_obs is not None:
            want = min(want, self.max_round_elements // max(int(num_obs), 1))

        return max(self._round_quantum, want)


def _acceptance_rate(stats):
    """Jeffreys-smoothed acceptance rate, so it is never zero and starts out pessimistic."""
    num_drawn, num_kept = stats

    return (num_kept + 0.5) / (num_drawn + 1.0)


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


class CustomSimulator(DataBandSimulator):
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
        # hyperparameter (`drift_slope_loc` in study 2, `threshold_diff_scale` in study 3)
        # away from the value the approximator was trained on --
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


class CustomMetaSimulator(DataBandSimulator):
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
