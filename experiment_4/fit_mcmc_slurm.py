import logging
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(4)

os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import save_hdf5
from utils import create_missing_dirs, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    filename = cfg["slurm_filename"]
    idx = cfg["slurm_idx"]

    basename = os.path.splitext(filename)[0]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "mcmc_samples", basename)])

    logger.info("Loading test data for file %s", filename)

    data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

    sim_data = data["x"]

    model_fun = instantiate(cfg["mcmc_model_fun"])

    # Need to pass sampler_fun here because it is not a function or class
    sampling_fun = instantiate(cfg["mcmc_sampling_fun"], sampler_fun=get_object(cfg["mcmc_sampler"]))

    if not cfg["is_meta"]:
        _p1 = cfg["simulator"]["prior_simulator"]["sample_fn"]["drift_slope_loc"]
        _p2 = cfg["simulator"]["prior_simulator"]["sample_fn"]["threshold_scale"]
        model = model_fun(sim_data[idx, :, :], _p1, _p2)
    else:
        _p1_lower = cfg["simulator"]["meta_simulator"]["sample_fn"]["min_value"][0]
        _p1_upper = cfg["simulator"]["meta_simulator"]["sample_fn"]["max_value"][0]
        _p2_lower = cfg["simulator"]["meta_simulator"]["sample_fn"]["min_value"][1]
        _p2_upper = cfg["simulator"]["meta_simulator"]["sample_fn"]["max_value"][0]
        model = model_fun(sim_data[idx, :, :], _p1_lower, _p1_upper, _p2_lower, _p2_upper)

    trace = sampling_fun(model, min_rt=sim_data[idx, :, 0].min())

    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", basename, f"samples_{idx}.hdf5"), {str(idx): trace[0].position})


if __name__ == "__main__":
    fit_mcmc()
