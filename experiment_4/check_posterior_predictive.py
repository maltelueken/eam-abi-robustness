import os
import logging

import blackjax
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data import load_hdf5
from metrics import calc_posterior_predictive
from utils import convert_posterior_samples, create_missing_dirs, load_approximator, read_data_from_txt


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_posterior_predictive(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    create_missing_dirs(["posterior_predictive"])

    dfs = []

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        basename = os.path.splitext(filename)[0]

        mcmc_data_path = os.path.join("..", "mcmc_samples", f"fit_mcmc_{basename}.hdf5")
        logger.info("Loading MCMC samples from %s", os.path.abspath(mcmc_data_path))
        mcmc_samples = load_hdf5(mcmc_data_path)

        posterior_mcmc = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)

        logger.info("%s MCMC models did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))
        posterior_mcmc = np.exp(np.reshape(posterior_mcmc, (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged]
        posterior_mcmc = posterior_mcmc[:, ::(posterior_mcmc.shape[1]//cfg["test_num_posterior_predictive_samples"]),:]

        data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

        data_x = data["x"][is_converged]

        df_mcmc = calc_posterior_predictive(data_x, posterior_mcmc, data["num_obs"], simulator, cfg["test_num_posterior_predictive_samples"], param_names)
        df_mcmc["name"] = basename

        npe_samples = approximator.sample(
            conditions=data,
            num_samples=cfg["test_num_posterior_predictive_samples"]
        )

        posterior_npe = convert_posterior_samples(npe_samples, param_names)[is_converged]

        df_npe = calc_posterior_predictive(data_x, posterior_npe, data["num_obs"], simulator, cfg["test_num_posterior_predictive_samples"], param_names)
        df_npe["name"] = basename

        dfs.append(pd.merge(df_npe, df_mcmc, on=["id", "sample", "acc_true", "quantile", "rt_true"], suffixes=["_npe", "_mcmc"]))

    pd.concat(dfs).to_csv(os.path.join("posterior_predictive", "ppd.csv"))


if __name__ == "__main__":
    check_posterior_predictive()
