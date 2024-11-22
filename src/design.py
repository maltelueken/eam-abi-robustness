import numpy as np


def random_num_obs_discrete(batch_shape, values, rng) -> int:
    return {"num_obs": np.tile(rng.choice(values), batch_shape)}


def random_num_obs_range(start, end, step, rng) -> int:
    return rng.choice(np.arange(start, end + step, step))


def random_num_obs_continuous(min_obs, max_obs, rng) -> int:
    """Draws a random number of observations for all simulations in a batch."""

    return rng.integers(low=min_obs, high=max_obs + 1)


def fixed_num_obs_range(start, end, step) -> np.ndarray:
    return np.arange(start, end + step, step)


def random_num_accumulators(min_n, max_n, probs, rng):
    return rng.choice(max_n, p=probs)+min_n


def random_prior_meta_continuous(batch_shape, min_value, max_value, name, rng):
    return {name: rng.uniform(min_value, max_value, size=batch_shape)}


def random_prior_meta_continuous_multivariate(batch_shape, min_value, max_value, name, rng):
    x = rng.uniform(low=min_value, high=max_value, size=(*batch_shape, 2)).T
    return {key: val for key, val in zip(name, x)}
