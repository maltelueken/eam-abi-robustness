import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    4
)

os.environ["JAX_PLATFORMS"] = "cpu"

import optuna
import hydra
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    filename = cfg["slurm_filename"]
    idx = cfg["slurm_idx"]

    basename = os.path.splitext(filename)[0]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "mcmc_samples", basename)])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    logger.info("Loading test data for file %s", filename)

    data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

    sim_data = data["x"]

    study = optuna.create_study(
        study_name=basename,
        storage="sqlite:///../../check_summary_mmd.db",
        load_if_exists=True
    )

    p1 = study.best_params[meta_param_name_1]
    p2 = study.best_params[meta_param_name_2]

    logger.info("Running MCMC with parameters %s", (p1, p2))

    model_fun = instantiate(cfg["mcmc_model_fun"])

    # Need to pass sampler_fun here because it is not a function or class
    sampling_fun = instantiate(cfg["mcmc_sampling_fun"], sampler_fun=get_object(cfg["mcmc_sampler"]))

    model = model_fun(sim_data[idx, :, :], p1, p2)

    trace = sampling_fun(model, min_rt=sim_data[idx, :, 0].min())

    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", basename, f"samples_{idx}.hdf5"), {str(idx): trace[0].position})


if __name__ == "__main__":
    fit_mcmc()
