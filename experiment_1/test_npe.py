import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"
import logging

import bayesflow as bf
import bayesflow.diagnostics.metrics as bf_metrics
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    
    data = simulator.sample(
        10,
    )

    logger.info("Data")

    for key in data:
        print(key, ": ", data[key].shape)

    forward_dict = approximator.adapter(data)

    approximator = bf.approximators.ContinuousApproximator(
        adapter=bf.adapters.Adapter(),
        inference_network=bf.networks.FlowMatching(),
        summary_network=bf.networks.SetTransformer()
    )

    # approximator.build({key: val.shape for key, val in forward_dict.items()})

    # print(forward_dict)

    for key in forward_dict:
        print(key, ": ", forward_dict[key].shape)

    print(forward_dict["inference_conditions"])


if __name__ == "__main__":
    train_npe()
