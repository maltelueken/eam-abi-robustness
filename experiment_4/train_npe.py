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
from utils import read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    optimizer = instantiate(cfg["optimizer"], _convert_="partial")

    approximator.compile(optimizer)

    callbacks = instantiate(cfg["callbacks"], _convert_="partial") # [keras.callbacks.TensorBoard()]

    history = approximator.fit(
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        simulator=simulator
    )

    _ = bf.diagnostics.plots.loss(history)

    plt.savefig("loss_history.png")

    diag_data = simulator.sample(
        cfg["diag_batch_size"], num_obs=np.array(cfg["eval_num_obs"])
    )

    real_data = read_data_from_txt(cfg["real_data_filename"])

    diag_summary_x = approximator.summary_network(diag_data["x"])
    real_summary_x = approximator.summary_network(real_data["x"])

    return bf.metrics.functional.maximum_mean_discrepancy(diag_summary_x, real_summary_x)


if __name__ == "__main__":
    train_npe()
