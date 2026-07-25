import functools
import logging
import os
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from data import load_hdf5
from data import save_hdf5
from rdm_jax import make_meta_to_unconstrained
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    p1 = cfg["slurm_p1"]
    p2 = cfg["slurm_p2"]

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "mcmc_samples")])

    test_data_path = os.path.join(
        cfg["test_data_path"], "test_data", f"test_data_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5",
    )
    logger.info("Loading test data from %s", os.path.abspath(test_data_path))
    data_x = load_hdf5(test_data_path)["x"]

    logger.info("Fitting MCMC for %s datasets with %s trials each", data_x.shape[0], data_x.shape[1])

    model_fun = instantiate(cfg["mcmc_model_fun"])
    fit_fun = instantiate(cfg["mcmc_sampling_fun"])

    if not cfg["is_meta"]:
        drift_slope_loc = cfg["simulator"]["prior_simulator"]["sample_fn"]["drift_slope_loc"]
        threshold_scale = cfg["simulator"]["prior_simulator"]["sample_fn"]["threshold_scale"]
        make_logdensity_fn = functools.partial(
            model_fun, drift_slope_loc=drift_slope_loc, threshold_scale=threshold_scale,
        )
        positions, infos = fit_fun(data=data_x, make_logdensity_fn=make_logdensity_fn)
    else:
        p1_lower = cfg["simulator"]["meta_simulator"]["sample_fn"]["min_value"][0]
        p1_upper = cfg["simulator"]["meta_simulator"]["sample_fn"]["max_value"][0]
        p2_lower = cfg["simulator"]["meta_simulator"]["sample_fn"]["min_value"][1]
        p2_upper = cfg["simulator"]["meta_simulator"]["sample_fn"]["max_value"][1]
        make_logdensity_fn = functools.partial(
            model_fun,
            drift_slope_loc_lower=p1_lower,
            drift_slope_loc_upper=p1_upper,
            threshold_scale_lower=p2_lower,
            threshold_scale_upper=p2_upper,
        )
        to_unconstrained = make_meta_to_unconstrained(p1_lower, p1_upper, p2_lower, p2_upper)
        positions, infos = fit_fun(data=data_x, make_logdensity_fn=make_logdensity_fn, to_unconstrained=to_unconstrained)

    positions.block_until_ready()

    logger.info("Divergence rate: %s", float(np.mean(infos.is_divergent)))

    # (num_datasets, num_sampling, num_chains, num_params) -> (num_datasets, num_chains,
    # num_sampling, num_params), matching what check_robustness.py/check_metrics.py/
    # check_summary_stats.py expect (the same layout the old fit_mcmc_slurm.py +
    # collect_mcmc.py combination produced).
    samples = np.transpose(np.asarray(positions), (0, 2, 1, 3))

    save_path = os.path.join(
        cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5",
    )
    logger.info("Saving MCMC samples to %s", os.path.abspath(save_path))
    save_hdf5(save_path, {"samples": samples})


if __name__ == "__main__":
    fit_mcmc()
