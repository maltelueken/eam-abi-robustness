import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def train_npe(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    approximator = instantiate(cfg["approximator"])
    approximator.data_adapter.transforms = [t for t in approximator.data_adapter.transforms]
    simulator = instantiate(cfg["simulator"])
    optimizer = instantiate(cfg["optimizer"])

    approximator.compile(optimizer=optimizer)

    callbacks = [callback for callback in instantiate(cfg["callbacks"])] # [keras.callbacks.TensorBoard()]

    approximator.fit(
        simulator=simulator,
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
    )


if __name__ == "__main__":
    train_npe()
