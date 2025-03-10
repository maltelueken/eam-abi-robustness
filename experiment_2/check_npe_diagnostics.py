
import logging
import os

from functools import partial

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow.diagnostics.plots as bf_plots
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs, load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_npe_diagnostics(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    plot_fns = {
        "recovery": bf_plots.recovery,
        "calibration_ecdf": partial(bf_plots.calibration_ecdf, difference=True),
        "z_score_contraction": bf_plots.z_score_contraction,
    }

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    meta_param_1 = instantiate(cfg["meta_param_1"])
    meta_param_2 = instantiate(cfg["meta_param_2"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    for p1 in meta_param_1:
        for p2 in meta_param_2:
            logger.info("Creating diag data for params %s - %s", p1, p2)

            forward_dict = simulator.sample(
                batch_shape=(cfg["diag_batch_size"],),
                **{
                    meta_param_name_1: np.array(p1),
                    meta_param_name_2: np.array(p2)
                },
                num_obs=np.array(cfg["diag_num_obs"])
            )

            sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}

            posterior_samples = approximator.sample(
                num_samples=cfg["diag_num_posterior_samples"], 
                conditions=sample_dict
            )

            create_missing_dirs(list(plot_fns.keys()))

            for key, fun in plot_fns.items():
                fig = fun(
                    estimates=posterior_samples,
                    targets=forward_dict,
                    variable_names=param_names
                )
                fig.savefig(os.path.join(key, f"{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.png"))


if __name__ == "__main__":
    check_npe_diagnostics()
