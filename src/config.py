"""Small accessors for config values that are otherwise reached through nested keys."""


def get_param_names(cfg):
    """Return the inference variables the approximator is trained on.

    These double as the parameters the MCMC posterior is aligned to: the hierarchical
    models sample additional hyperparameters that the NPE never sees, so this list -- not
    the stored MCMC parameter list -- defines what the two posteriors are compared on.
    """
    return list(cfg["approximator"]["adapter"]["inference_variables"])


def get_mcmc_param_names(cfg):
    """Return the full parameter vector the MCMC log-density is defined over."""
    return list(cfg["mcmc_param_names"])
