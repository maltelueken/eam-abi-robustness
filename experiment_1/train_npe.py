import logging

import bayesflow as bf
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def train_npe(cfg: DictConfig):
    trainer = instantiate(cfg["trainer"])

    # logger.info("Starting training with: %s", cfg["trainer"]["dynamic"])
    history = trainer.train_online(epochs=2, iterations_per_epoch=50, batch_size=64)
    # logger.info("Finished training")

    history.to_csv("history.csv")

    fig = bf.diagnostics.plot_losses(history)

    fig.savefig("training_loss.png")


if __name__ == "__main__":
    train_npe()