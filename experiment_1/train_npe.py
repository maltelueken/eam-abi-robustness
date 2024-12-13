import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"
import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    # cfg = OmegaConf.to_object(cfg)
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    simulator = instantiate(cfg["simulator"])
    optimizer = instantiate(cfg["optimizer"])

    approximator.compile(optimizer=optimizer)

    callbacks = instantiate(cfg["callbacks"], _convert_="partial") # [keras.callbacks.TensorBoard()]

    history = approximator.fit(
        simulator=simulator,
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
    )

    return(np.mean(history.history["loss"][-5:]))


if __name__ == "__main__":
    train_npe()
