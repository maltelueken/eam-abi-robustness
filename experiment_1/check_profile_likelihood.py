import logging
import os

import bayesflow as bf
import blackjax
import hydra
import numpy as np
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs, create_profile_likelihood_plot

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def check_profile_likelihood(cfg: DictConfig):
    sample_sizes = instantiate(cfg["test_num_obs"])

    param_names = cfg["trainer"]["generative_model"]["prior"]["param_names"]

    model_fun = instantiate(cfg["mcmc_model_fun"])

    create_missing_dirs(["profile_likelihood"])

    for t in sample_sizes:
        logger.info("Loading test data for sample size %s", t)

        data = load_hdf5(os.path.join("test_data", f"test_data_sample_size_{t}.hdf5"))

        sim_data = data["sim_data"]

        logger.info("Vectorizing model function over %s datasets", sim_data.shape[0])
        model_fun_vec = np.vectorize(model_fun, signature="(m,n)->()")

        model_funs_with_data = model_fun_vec(sim_data)

        for i, fun in enumerate(model_funs_with_data):
            fig = create_profile_likelihood_plot(fun, np.log(data["prior_draws"][i, :]), param_names, p_range=1.0)
            fig.savefig(os.path.join("profile_likelihood", f"profile_dataset_{i}.png"))


if __name__ == "__main__":
    check_profile_likelihood()