"""Train a neural posterior estimator and report recovery diagnostics."""

import logging

import bayesflow as bf
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_param_names
from metrics import diagnostic_summary

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    """Fit the approximator, then score it on a freshly simulated diagnostic batch."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    optimizer = instantiate(cfg["optimizer"], _convert_="partial")

    approximator.compile(optimizer)

    history = approximator.fit(
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=instantiate(cfg["callbacks"], _convert_="partial"),
        simulator=simulator,
        # Keras defaults to cpu_count() loader threads, which all share this simulator's one
        # SplittableKey and one numpy Generator -- so the training stream depended on thread
        # interleaving and `seed` did not fully determine a run. Now that a batch takes ~4 ms
        # to simulate rather than ~470, single-threaded loading costs nothing and buys
        # reproducibility.
        workers=1,
    )

    _ = bf.diagnostics.plots.loss(history)
    plt.savefig("loss_history.png")

    # Scored at every trial count in `diag_num_obs` and averaged, rather than at one. The
    # architecture the sweep selects here has to serve all six studies, and they read the
    # networks at 500 (studies 2, 3, 5 and 6), at the empirical subjects' 93-100 (study 4) and
    # across 50-1200 (study 1); a single num_obs -- it was 1000, the top of the training grid --
    # scored it at the one count none of them use, while set size is exactly what the summary
    # network's pooling generalizes over.
    scores = []

    for num_obs in cfg["diag_num_obs"]:
        diag_sample = simulator.sample(cfg["diag_batch_size"], num_obs=np.array(num_obs))

        posterior_samples = approximator.sample(
            num_samples=cfg["diag_num_posterior_samples"],
            conditions=diag_sample,
        )

        summary = diagnostic_summary(posterior_samples, diag_sample, get_param_names(cfg))
        logger.info("Diagnostics at num_obs=%s: %s", num_obs, summary)
        scores.append(summary)

    # Returned as a tuple of floats so the Optuna sweeper can optimize against these three
    # objectives; plain floats because the sweeper stores what it is handed.
    return tuple(float(value) for value in np.mean(scores, axis=0))


if __name__ == "__main__":
    train_npe()
