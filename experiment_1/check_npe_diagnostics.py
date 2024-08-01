
import logging
import os
import pickle
from functools import partial

import bayesflow as bf
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def check_npe_diagnostics(cfg: DictConfig):
    trainer = instantiate(cfg["trainer"])

    data = trainer.generative_model(
        cfg["diag_batch_size"]
    )

    # prior_samples = data["prior_draws"]

    prior_means = instantiate(cfg["prior_stats"]["prior_means"])
    prior_stds = instantiate(cfg["prior_stats"]["prior_stds"])

    forward_dict = trainer.configurator(data)
    prior_samples = forward_dict["parameters"]

    posterior_samples = trainer.amortizer.sample(
        forward_dict, n_samples=cfg["diag_num_posterior_samples"]
    )

    prior_samples = np.exp(prior_samples * prior_stds + prior_means)
    posterior_samples = np.exp(posterior_samples * prior_stds + prior_means)

    create_missing_dirs(["histograms", "ecdf", "recovery", "contraction", "posterior_2d"])

    fig = bf.diagnostics.plot_sbc_histograms(posterior_samples, prior_samples, num_bins=10, param_names=trainer.generative_model.param_names)
    fig.savefig(os.path.join("histograms", f"sbc_histograms_sample_size"))

    fig = bf.diagnostics.plot_sbc_ecdf(posterior_samples, prior_samples, stacked=False, difference=True, param_names=trainer.generative_model.param_names)
    fig.savefig(os.path.join("ecdf", f"sbc_ecdf_sample_size"))

    fig = bf.diagnostics.plot_recovery(
        posterior_samples, prior_samples, param_names=trainer.generative_model.param_names, point_agg=np.mean, uncertainty_agg=np.std
    )
    fig.savefig(os.path.join("recovery", f"recovery_sample_size"))

    fig = bf.diagnostics.plot_z_score_contraction(posterior_samples, prior_samples, param_names=trainer.generative_model.param_names)
    fig.savefig(os.path.join("contraction", f"contraction_sample_size"))

    fig = bf.diagnostics.plot_posterior_2d(posterior_draws=posterior_samples[0, :, :], prior_draws=prior_samples, param_names=trainer.generative_model.param_names)
    fig.savefig(os.path.join("posterior_2d", f"posterior_2d_sample_size"))


if __name__ == "__main__":
    check_npe_diagnostics()