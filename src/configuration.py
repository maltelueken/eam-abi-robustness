import numpy as np


def rdm_configurator_simple(forward_dict, prior_means, prior_stds, transform_fun):
    """Configure the output of the GenerativeModel for a BayesFlow setup."""

    # Prepare placeholder dict
    out_dict = {}

    # Get data generating parameters
    params = transform_fun(forward_dict["prior_draws"].astype(np.float32))

    # Standardize parameters
    out_dict["parameters"] = (params - prior_means) / prior_stds

    # Concatenate rt, resp, context
    out_dict["summary_conditions"] = forward_dict["sim_data"].astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = forward_dict["sim_non_batchable_context"] * np.ones(
        (forward_dict["sim_data"].shape[0], 1)
    )
    out_dict["direct_conditions"] = np.sqrt(vec_num_obs).astype(np.float32)

    return out_dict
