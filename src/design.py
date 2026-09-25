"""Design utilities for generating random numbers of observations and priors."""


def random_num_obs_discrete(batch_shape, values, rng) -> int:
    """Randomly select a number of observations from given values."""
    num_obs = rng.choice(values)
    return {"num_obs": num_obs}


def random_prior_meta_continuous_multivariate(
    batch_shape, min_value, max_value, name, rng
):
    """Generate random continuous meta-priors for one or more variables.

    `name` decides how many hyperparameters are randomized: two for the hierarchical models
    of study 2, one for study 3's speed-accuracy model, which shifts only the prior on the
    threshold difference.
    """
    x = rng.uniform(low=min_value, high=max_value, size=(*batch_shape, len(name))).T
    return {key: val for key, val in zip(name, x)}
