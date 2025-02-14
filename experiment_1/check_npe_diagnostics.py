
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs, load_workflow

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_npe_diagnostics(cfg: DictConfig):
    workflow = load_workflow(cfg)

    param_names = cfg["workflow"]["adapter"]["inference_variables"]

    sample_sizes = instantiate(cfg["diag_num_obs"])

    for t in sample_sizes:
        logger.info("Creating diag data for sample size %s", t)

        forward_dict = workflow.simulate(
            batch_shape=(cfg["diag_batch_size"],), num_obs=np.tile([t], (cfg["diag_batch_size"],))
        )

        # sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}

        # posterior_samples = workflow.sample(
        #     num_samples=cfg["diag_sbc_num_posterior_samples"], 
        #     conditions=sample_dict
        # )

        figures = workflow.plot_diagnostics(
            test_data=forward_dict,
            num_samples=cfg["diag_sbc_num_posterior_samples"],
            variable_names=param_names
        )

        create_missing_dirs(list(figures.keys()))

        for name, fig in figures.items():
            fig.savefig(os.path.join(name, f"sample_size_{t}"))


if __name__ == "__main__":
    check_npe_diagnostics()
