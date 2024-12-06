import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import keras
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from bayesflow_plots import *
from data import save_hdf5
from utils import convert_posterior_samples, create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])
    # approximator = instantiate(cfg["approximator"])
    # data_adapter = approximator.adapter

    forward_dict = simulator.sample(
        batch_shape=(cfg["diag_batch_size"],), num_obs=np.tile([500], (cfg["diag_batch_size"],))
    )

    # if not approximator.built:
    #     dataset = approximator.build_dataset(
    #         simulator=simulator,
    #         adapter=data_adapter,
    #         num_batches=cfg["iterations_per_epoch"],
    #         batch_size=cfg["batch_size"],
    #     )
    #     dataset = keras.tree.map_structure(lambda x: keras.ops.convert_to_tensor(x, dtype="float32"), dataset[0])
    #     approximator.build_from_data(dataset)

    # approximator.load_weights(cfg["callbacks"][1]["filepath"])

    # param_names = cfg["approximator"]["adapter"]["inference_variables"]

    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])
    # create_missing_dirs(["npe_samples"])

    for t in sample_sizes:
        logger.info("Creating test data set for sample size %s", t)

        forward_dict = simulator.sample(
            batch_shape=(cfg["diag_batch_size"],), num_obs=np.tile([t], (cfg["diag_batch_size"],))
        )

        # sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}

        # posterior_samples = approximator.sample(
        #     conditions=sample_dict, batch_size=cfg["test_batch_size"], num_samples=cfg["test_num_posterior_samples"]
        # )
        test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_sample_size_{t}.hdf5")
        logger.info("Saving test data to %s", os.path.abspath(test_data_path))
        save_hdf5(test_data_path, forward_dict)
        # save_hdf5(os.path.join("npe_samples", f"posterior_samples_sample_size_{t}.hdf5"), {"samples": posterior_samples})


if __name__ == "__main__":
    generate_test_data()
