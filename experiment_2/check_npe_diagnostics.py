
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import keras
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from bayesflow_plots import *
from utils import convert_posterior_samples, convert_prior_samples, create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_npe_diagnostics(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])
    approximator = instantiate(cfg["approximator"])
    data_adapter = approximator.adapter

    forward_dict = simulator.sample(
        batch_shape=(cfg["diag_batch_size"],),
        num_obs=np.tile([500], (cfg["diag_batch_size"],)),
        drift_slope_loc=np.tile([0.5], (cfg["diag_batch_size"],)),
        threshold_scale_loc=np.tile([0.5], (cfg["diag_batch_size"],))
    )

    if not approximator.built:
        dataset = approximator.build_dataset(
            simulator=simulator,
            adapter=data_adapter,
            num_batches=cfg["iterations_per_epoch"],
            batch_size=cfg["batch_size"],
        )
        dataset = keras.tree.map_structure(lambda x: keras.ops.convert_to_tensor(x, dtype="float32"), dataset[0])
        approximator.build_from_data(dataset)

    approximator.load_weights(cfg["callbacks"][1]["filepath"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    drift_slope_loc = instantiate(cfg["diag_drift_slope_loc"])
    threshold_scale = instantiate(cfg["diag_threshold_scale"])

    for p1 in drift_slope_loc:
        for p2 in threshold_scale:
            logger.info("Creating diag data for param %s", p1)

            forward_dict = simulator.sample(
                batch_shape=(cfg["diag_batch_size"],),
                drift_slope_loc=np.tile([p1], (cfg["diag_batch_size"],)),
                threshold_scale_loc=np.tile([p2], (cfg["diag_batch_size"],)),
                num_obs=np.tile([cfg["diag_num_obs"]], (cfg["diag_batch_size"],))
            )

            prior_samples = convert_prior_samples(forward_dict, param_names)

            sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}

            posterior_samples_sbc = convert_posterior_samples(approximator.sample(
                conditions=sample_dict, batch_size=cfg["diag_batch_size"], num_samples=cfg["diag_sbc_num_posterior_samples"]
            ), param_names)

            posterior_samples_sens = convert_posterior_samples(approximator.sample(
                conditions=sample_dict, batch_size=cfg["diag_batch_size"], num_samples=cfg["diag_sens_num_posterior_samples"]
            ), param_names)

            create_missing_dirs(["histograms", "ecdf", "recovery", "contraction", "posterior_2d"])

            fig = plot_sbc_histograms(posterior_samples_sbc, prior_samples, num_bins=10, param_names=param_names)
            fig.savefig(os.path.join("histograms", f"sbc_histograms_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))

            fig = plot_sbc_ecdf(posterior_samples_sbc, prior_samples, stacked=False, difference=True, param_names=param_names)
            fig.savefig(os.path.join("ecdf", f"sbc_ecdf_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))

            fig = plot_recovery(
                posterior_samples_sens, prior_samples, param_names=param_names, point_agg=np.mean, uncertainty_agg=np.std
            )
            fig.savefig(os.path.join("recovery", f"recovery_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))

            fig = plot_z_score_contraction(posterior_samples_sens, prior_samples, param_names=param_names)
            fig.savefig(os.path.join("contraction", f"contraction_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))

            fig = plot_posterior_2d(posterior_draws=posterior_samples_sens[0, :, :], prior_draws=prior_samples, param_names=param_names)
            fig.savefig(os.path.join("posterior_2d", f"posterior_2d_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))


if __name__ == "__main__":
    check_npe_diagnostics()
