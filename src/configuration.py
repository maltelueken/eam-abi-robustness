import numpy as np


def rdm_configurator_simple(forward_dict):
    """Configure the output of the GenerativeModel for a BayesFlow setup."""

    # Prepare placeholder dict
    out_dict = {}

    # Get data generating parameters
    out_dict["parameters"] = forward_dict["prior_draws"].astype(np.float32)

    # Standardize parameters
    # out_dict["parameters"] = (params - prior_means) / prior_stds

    # Extract simulated response times
    data = forward_dict["sim_data"]

    # Convert list of condition indicators to a 2D array and add a
    # trailing dimension of 1, so shape becomes (batch_size, num_obs, 1)
    # We need this in order to easily concatenate the context with the data
    # context = np.array(forward_dict["sim_batchable_context"])[..., None]

    # One-hot encoding of integer choices
    # categorical_resp = to_categorical(data[:, :, 1], num_classes=3)

    # Concatenate rt, resp, context
    out_dict["summary_conditions"] = forward_dict["sim_data"].astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = forward_dict["sim_non_batchable_context"] * np.ones(
        (data.shape[0], 1)
    )
    out_dict["direct_conditions"] = np.c_[np.sqrt(vec_num_obs),].astype(np.float32)

    return out_dict
