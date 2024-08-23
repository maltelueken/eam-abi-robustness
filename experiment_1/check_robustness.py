import logging
import os

import bayesflow as bf
import blackjax
import hydra
import numpy as np
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs, create_robustness_2d_plot

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def check_robustness(cfg: DictConfig):
    sample_sizes = instantiate(cfg["test_num_obs"])

    param_names = cfg["trainer"]["generative_model"]["prior"]["param_names"]

    create_missing_dirs(["robustness"])

    for t in sample_sizes:
        logger.info("Loading test data for sample size %s", t)
        data = load_hdf5(os.path.join("test_data", f"test_data_sample_size_{t}.hdf5"))

        mcmc_samples = load_hdf5(os.path.join("mcmc_samples", f"fit_mcmc_sample_size_{t}.hdf5"))

        posterior_mcmc = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)
        posterior_mcmc = np.exp(np.reshape(posterior_mcmc[:,:,-500:,:], (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged,:,:]
        prior_samples = data["prior_draws"][is_converged,:]
        npe_samples = load_hdf5(os.path.join("npe_samples", f"posterior_samples_sample_size_{t}.hdf5"))
        posterior_npe = np.exp(npe_samples["samples"])[is_converged,:]

        idx = 3

        fig = create_robustness_2d_plot(posterior_npe[idx,:,:], posterior_mcmc[idx,:,:], prior_samples, prior_samples[idx,:], param_names)
        fig.savefig(os.path.join("robustness", f"robustness_2d_sample_size_{t}.png"))

        logger.info("%s datasets did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))


if __name__ == "__main__":
    check_robustness()