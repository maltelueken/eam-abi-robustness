import os
import logging

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

# os.environ["JAX_PLATFORMS"] = "cpu"

import blackjax
import hydra
import numpy as np
import optuna
import polars as pl
from omegaconf import DictConfig

from data import load_hdf5
from utils import create_missing_dirs, load_approximator, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_posterior_predictive(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    create_missing_dirs(["posterior_predictive"])

    basenames = []
    npe_acc_rmsd = []
    npe_rt_rmsd = []
    mcmc_acc_rmsd = []
    mcmc_rt_rmsd = []

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        basename = os.path.splitext(filename)[0]

        mcmc_data_path = os.path.join("mcmc_samples", f"fit_mcmc_{basename}.hdf5")
        logger.info("Loading MCMC samples from %s", os.path.abspath(mcmc_data_path))
        mcmc_samples = load_hdf5(mcmc_data_path)

        posterior_mcmc = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)

        logger.info("%s MCMC models did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))
        posterior_mcmc = np.exp(np.reshape(posterior_mcmc, (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged]
        posterior_mcmc = posterior_mcmc[:, ::(posterior_mcmc.shape[1]//cfg["test_num_posterior_predictive_samples"]),:]

        data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

        data_x = data["x"][is_converged]

        mcmc_acc = []
        mcmc_rt = []

        for i in range(data_x.shape[0]):
            acc_i = []
            rt_i = []

            for j in range(cfg["test_num_posterior_predictive_samples"]):
                sim = simulator.experiment_simulator.sample_fn(
                    v_intercept = posterior_mcmc[i,j,0],
                    v_slope= posterior_mcmc[i,j,1],
                    s_true = posterior_mcmc[i,j,2],
                    b = posterior_mcmc[i,j,3],
                    t0 = posterior_mcmc[i,j,4],
                    num_obs=data["num_obs"]
                )
                acc_i.append(sim["x"][:,1].mean())
                rt_i.append(np.quantile(sim["x"][:,0], q=np.arange(1, 10)/10))

            mcmc_acc.append(np.sqrt(np.mean((np.array(acc_i) - data_x[i,:,1].mean())**2)))
            mcmc_rt.append(np.sqrt(np.mean((np.array(rt_i) - np.quantile(data_x[i,:,0], q=np.arange(1, 10)/10))**2)))

        posterior_npe = approximator.sample(
            conditions=data,
            num_samples=cfg["test_num_posterior_predictive_samples"]
        )

        npe_acc = []
        npe_rt = []

        for i in range(data_x.shape[0]):
            acc_i = []
            rt_i = []

            for j in range(cfg["test_num_posterior_predictive_samples"]):
                sim = simulator.experiment_simulator.sample_fn(
                    v_intercept = posterior_npe["v_intercept"][i,j,0],
                    v_slope= posterior_npe["v_slope"][i,j,0],
                    s_true = posterior_npe["s_true"][i,j,0],
                    b = posterior_npe["b"][i,j,0],
                    t0 = posterior_npe["t0"][i,j,0],
                    num_obs=data["num_obs"]
                )
                acc_i.append(sim["x"][:,1].mean())
                rt_i.append(np.quantile(sim["x"][:,0], q=np.arange(1, 10)/10))

            npe_acc.append(np.sqrt(np.mean((np.array(acc_i) - data_x[i,:,1].mean())**2)))
            npe_rt.append(np.sqrt(np.mean((np.array(rt_i) - np.quantile(data_x[i,:,0], q=np.arange(1, 10)/10))**2)))

        basenames.append(basename)
        npe_acc_rmsd.append(npe_acc)
        npe_rt_rmsd.append(npe_rt)
        mcmc_acc_rmsd.append(mcmc_acc)
        mcmc_rt_rmsd.append(mcmc_rt)


    pl.DataFrame({
        "name": basenames,
        "npe_acc_rmsd": npe_acc_rmsd,
        "npe_rt_rmsd": npe_rt_rmsd,
        "mcmc_acc_rmsd": mcmc_acc_rmsd,
        "mcmc_rt_rmsd": mcmc_rt_rmsd
    }).explode(["npe_acc_rmsd", "npe_rt_rmsd", "mcmc_acc_rmsd", "mcmc_rt_rmsd"]).write_csv(os.path.join("posterior_predictive", "ppd.csv"))


if __name__ == "__main__":
    check_posterior_predictive()
