import logging
import os

import bayesflow as bf
import blackjax
import hydra
import numpy as np
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_mcmc(cfg: DictConfig):
    sample_sizes = instantiate(cfg["test_num_obs"])

    param_names = cfg["trainer"]["generative_model"]["prior"]["param_names"]

    create_missing_dirs(["mcmc_recovery"])

    for t in sample_sizes:
        logger.info("Loading test data for sample size %s", t)

        data = load_hdf5(os.path.join("test_data", f"test_data_sample_size_{t}.hdf5"))

        mcmc_samples = load_hdf5(os.path.join("mcmc_samples", f"fit_mcmc_sample_size_{t}.hdf5"))

        posterior_samples = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_samples, chain_axis=1, sample_axis=2) < cfg["psrf_threshold"], axis=1)
        posterior_samples = np.exp(np.reshape(posterior_samples, (posterior_samples.shape[0], -1, posterior_samples.shape[3])))[is_converged,:,:]
        # posterior_samples[:, :, -1] += 0.1
        prior_samples = data["prior_draws"][is_converged,:]

        logger.info("%s datasets did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))

        fig = bf.diagnostics.plot_sbc_histograms(
            posterior_samples,
            prior_samples,
            param_names=param_names,
            num_bins=10
        )
        fig.savefig(os.path.join("mcmc_recovery", f"mcmc_histograms_sample_size_{t}.png"))

        fig = bf.diagnostics.plot_sbc_ecdf(
            posterior_samples,
            prior_samples,
            param_names=param_names,
            stacked=False,
            difference=True
        )
        fig.savefig(os.path.join("mcmc_recovery", f"mcmc_ecdf_sample_size_{t}.png"))

        fig = bf.diagnostics.plot_recovery(
            posterior_samples,
            prior_samples,
            param_names=param_names,
            point_agg=np.mean,
            uncertainty_agg=np.std
        )
        fig.savefig(os.path.join("mcmc_recovery", f"mcmc_recovery_sample_size_{t}.png"))

        fig = bf.diagnostics.plot_z_score_contraction(
            posterior_samples,
            prior_samples,
            param_names=param_names
        )
        fig.savefig(os.path.join("mcmc_recovery", f"mcmc_z_score_contraction_sample_size_{t}.png"))

        fig = bf.diagnostics.plot_posterior_2d(
            posterior_samples[1,-2000:,:],
            prior_draws=prior_samples,
            param_names=param_names
        )
        fig.savefig(os.path.join("mcmc_recovery", f"mcmc_posterior_2d_sample_size_{t}.png"))


if __name__ == "__main__":
    check_mcmc()