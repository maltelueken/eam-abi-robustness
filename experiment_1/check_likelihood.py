import logging
import os

import hydra
import keras
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from bayesflow_plots import *
from utils import convert_posterior_samples, convert_prior_samples, create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_likelihood(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])
    approximator = instantiate(cfg["approximator"])
    data_adapter = approximator.adapter

    forward_dict = simulator.sample(
        batch_shape=(cfg["diag_batch_size"],), num_obs=np.tile([500], (cfg["diag_batch_size"],))
    )

    if not approximator.built:
        dataset = approximator.build_dataset(
            simulator=simulator,
            adapter=data_adapter,
            num_batches=cfg["iterations_per_epoch"],
            batch_size=cfg["batch_size"],
        )
        dataset = keras.tree.map_structure(lambda x: keras.ops.convert_to_tensor(x, dtype="float32"), dataset[0])
        approximator.build_from_data(dataset)

    approximator.load_weights(cfg["callbacks"][1]["filepath"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]
