import concurrent.futures
import logging
import os
import cloudpickle as pickle
from functools import partial

import arviz as az
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def fit_mcmc(cfg: DictConfig):
    t = cfg["t"]
    i = cfg["i"]
    logger.info("Loading test dataset %s for sample size %s", i, t)

    data = np.load(os.path.join("test_data", f"test_data_sample_size_{500}.npz"))

    sim_data = data["sim_data"][i, :, :]

    model_fun = instantiate(cfg["model"]["model_fun"])

    trace = model_fun(
        sim_data,
        draws=cfg["test_num_posterior_samples"],
        chains=cfg["mcmc_chains"],
        progressbar=True,
        cores=cfg["mcmc_cores"],
        idata_kwargs={"log_likelihood": True}
    )

    with open(f"fit_mcmc_sample_size_{t}.pkl", "wb") as file:
        pickle.dump(trace, file)


if __name__ == "__main__":
    fit_mcmc()
