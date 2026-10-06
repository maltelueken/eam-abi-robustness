"""Design utilities for generating random numbers of observations and priors.

Like the priors, these draw with `jax.random` from an `rdm_jax.SplittableKey` on a stream of
their own, and return numpy -- see the `priors` module docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np


def random_num_obs_discrete(batch_shape, values, rng) -> dict:
    """Randomly select one number of observations, shared by the whole batch, from `values`."""
    num_obs = jax.random.choice(rng.next(), jnp.asarray(values))
    return {"num_obs": np.int64(num_obs)}


def random_prior_meta_continuous_multivariate(
    batch_shape, min_value, max_value, name, rng
):
    """Generate random continuous meta-priors for one or more variables.

    `name` decides how many hyperparameters are randomized: two for the hierarchical models
    of study 2, one for study 3's speed-accuracy model, which shifts only the prior on the
    threshold difference.
    """
    dtype = jnp.result_type(float)
    x = jax.random.uniform(
        rng.next(),
        (*batch_shape, len(name)),
        dtype=dtype,
        minval=jnp.asarray(min_value, dtype=dtype),
        maxval=jnp.asarray(max_value, dtype=dtype),
    )
    return {key: np.asarray(val) for key, val in zip(name, x.T)}
