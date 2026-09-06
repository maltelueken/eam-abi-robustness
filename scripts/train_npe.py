"""Train a neural posterior estimator and report recovery diagnostics."""

import logging

import bayesflow as bf
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from artifacts import Artifacts
from config import get_param_names
from ensemble import num_members
from metrics import diagnostic_summary
from timing import TimingLog
from timing import run_labels

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    """Fit the approximator, then score it on a freshly simulated diagnostic batch."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    optimizer = instantiate(cfg["optimizer"], _convert_="partial")

    approximator.compile(optimizer)

    # The one-off cost the rest of the pipeline amortizes: recorded here, in the run directory,
    # so it is measured on the same hardware and the same configuration as the network it
    # produced. `num_datasets` is what those seconds bought -- the simulated datasets the fit
    # consumed -- and `num_members` how many networks were trained out of them at once, since
    # an ensemble trains its members in one Keras model at `ensemble_size` times the cost per
    # step. `num_obs` stays blank: the design simulator randomizes the trial count per dataset.
    timings = TimingLog(Artifacts(cfg).csv("timing", "train_npe"), run_labels())

    with timings.timed(
        "train",
        num_datasets=cfg["epochs"] * cfg["iterations_per_epoch"] * cfg["batch_size"],
        num_members=num_members(approximator),
        epochs=cfg["epochs"],
    ):
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

        # Timed alongside the fit because it is the same posterior sampling `predict_npe` does,
        # at four trial counts and a known batch size -- the amortized per-dataset cost, read
        # off the network that was just trained. The first of these calls also pays for the
        # sampling function's XLA compilation.
        with timings.timed(
            "diagnostics",
            num_datasets=cfg["diag_batch_size"],
            num_obs=num_obs,
            num_draws=cfg["diag_num_posterior_samples"],
            num_members=num_members(approximator),
        ):
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
