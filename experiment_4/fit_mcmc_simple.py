import concurrent.futures
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    32
)
os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import numpy as np
import tqdm
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from data import save_hdf5
from utils import create_missing_dirs, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):

    create_missing_dirs([os.path.join(cfg["test_data_path"], "mcmc_samples")])

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

        sim_data = data["x"]

        model_fun = instantiate(cfg["mcmc_model_fun"])

        # Need to pass sampler_fun here because it is not a function or class
        sampling_fun = instantiate(cfg["mcmc_sampling_fun"], sampler_fun=get_object(cfg["mcmc_sampler"]))

        trace_data = {}

        logger.info("Vectorizing model function over %s datasets", sim_data.shape[0])
        model_fun_vec = np.vectorize(model_fun, signature="(m,n)->()")

        model_funs_with_data = model_fun_vec(sim_data)

        min_rt = sim_data[:, :, 0].min(axis=1)

        num_workers = int(32/cfg["mcmc_sampling_fun"]["num_chains"])

        logger.info("Estimating posteriors with MCMC across %s workers", num_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(
                    sampling_fun,
                    fun,
                    min_rt=min_rt[i]
                ): i
                for i, fun in enumerate(model_funs_with_data)
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

        # Save to hdf5 file
        save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{os.path.splitext(filename)[0]}.hdf5"), {"samples": np.array(list(trace_data.values()))})


if __name__ == "__main__":
    fit_mcmc()
