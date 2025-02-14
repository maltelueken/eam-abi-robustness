import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"
import logging

import bayesflow as bf
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    # cfg = OmegaConf.to_object(cfg)
    workflow = instantiate(cfg["workflow"], _convert_="partial")
    # simulator = instantiate(cfg["simulator"])
    # optimizer = instantiate(cfg["optimizer"])

    # approximator.compile(optimizer=optimizer)

    callbacks = instantiate(cfg["callbacks"], _convert_="partial") # [keras.callbacks.TensorBoard()]

    history = workflow.fit_online(
        epochs=cfg["epochs"],
        num_batches_per_epoch=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
    )

    _ = bf.diagnostics.plots.loss(history)

    plt.savefig("loss_history.png")

    diag_sample = workflow.simulate(
        batch_shape=(cfg["diag_batch_size"],), num_obs=np.tile([1000], (cfg["diag_batch_size"],))
    )

    metrics = workflow.compute_diagnostics(
        test_data=diag_sample,
        num_samples=1000
    ).to_numpy().mean(axis=1)

    return metrics[0], metrics[1], metrics[2]


if __name__ == "__main__":
    train_npe()
