import concurrent.futures
import logging
import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import numpy as np
import tqdm
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="test_npe")
def fit_mcmc(cfg: DictConfig):
    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs(["mcmc_samples"])

    for t in sample_sizes:
        logger.info("Loading test data for sample size %s", t)

        data = load_hdf5(os.path.join("test_data", f"test_data_sample_size_{t}.hdf5"))
    
        sim_data = data["sim_data"]

        model_fun = instantiate(cfg["mcmc_model_fun"])

        # Need to pass sampler_fun here because it is not a function or class
        sampling_fun = instantiate(cfg["mcmc_sampling_fun"], sampler_fun=get_object(cfg["mcmc_sampler"]))

        trace_data = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=sim_data.shape[0]) as executor:
            future_to_idx = {
                executor.submit(
                    sampling_fun,
                    model_fun(sim_data[i, :, :])
                ): i
                for i in range(sim_data.shape[0])
            }

            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    trace_data[idx] = future.result()[0].position
                except Exception as exc:
                    logger.exception("Dataset %s generated an exception: %s", idx, exc)
                # else:
                #     logger.info("Dataset %s successfully fitted", idx)

        # Sort results so that order matches that of test data
        trace_data = dict(sorted(trace_data.items()))

        print(trace_data.keys())

        save_hdf5(os.path.join("mcmc_samples", f"fit_mcmc_sample_size_{t}.hdf5"), {"samples": np.array(list(trace_data.values()))})


if __name__ == "__main__":
    fit_mcmc()
