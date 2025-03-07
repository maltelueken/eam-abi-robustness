
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

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    sample_sizes = instantiate(cfg["diag_num_obs"])

    plot_fns = {
        "recovery": bf_plots.recovery,
        "calibration_ecdf": partial(bf_plots.calibration_ecdf, difference=True),
        "z_score_contraction": bf_plots.z_score_contraction,
    }

    for t in sample_sizes:
        logger.info("Creating diag data for sample size %s", t)

        forward_dict = simulator.sample(
            cfg["diag_batch_size"], num_obs=np.array(t)
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
            fig.savefig(os.path.join(key, f"sample_size_{t}"))


if __name__ == "__main__":
    check_npe_diagnostics()
