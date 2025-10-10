import numpy as np


def random_num_obs_discrete(batch_shape, values, rng) -> int:
    num_obs = rng.choice(values)
    return {"num_obs": num_obs}


def random_prior_meta_continuous_multivariate(batch_shape, min_value, max_value, name, rng):
    x = rng.uniform(low=min_value, high=max_value, size=(*batch_shape, 2)).T
    return {key: val for key, val in zip(name, x)}
