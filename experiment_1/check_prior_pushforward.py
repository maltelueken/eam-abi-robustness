import logging

import bayesflow as bf
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_pushforward_plot

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def prior_pushforward(cfg: DictConfig):
    trainer = instantiate(cfg["trainer"])
    trainer.generative_model.simulator.context_gen = None

    fig = trainer.generative_model.prior.plot_prior2d(n_samples=500)
    fig.savefig("prior2d.png")

    example_sim = trainer.generative_model(batch_size=10, **{"sim_args": {"num_obs": 500}})

    fig = create_pushforward_plot(example_sim)
    fig.savefig("prior_pushforward.png")


if __name__ == "__main__":
    prior_pushforward()
