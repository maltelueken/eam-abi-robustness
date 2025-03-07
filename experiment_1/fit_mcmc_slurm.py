import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    4
)

os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    t = cfg["slurm_num_obs"]
    idx = cfg["slurm_idx"]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "mcmc_samples", str(t))])

    logger.info("Loading test data for sample size %s", t)

    data = load_hdf5(os.path.join(cfg["test_data_path"], "test_data", f"test_data_sample_size_{t}.hdf5"))
    
    sim_data = data["x"]

    model_fun = instantiate(cfg["mcmc_model_fun"])

    # Need to pass sampler_fun here because it is not a function or class
    sampling_fun = instantiate(cfg["mcmc_sampling_fun"], sampler_fun=get_object(cfg["mcmc_sampler"]))

    model = model_fun(sim_data[idx, :, :])

    trace = sampling_fun(model, min_rt=sim_data[idx, :, 0].min())

    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", str(t), f"samples_{idx}.hdf5"), {str(idx): trace[0].position})


if __name__ == "__main__":
    fit_mcmc()
