"""Prior pushforward figure: the marginal priors, and the datasets they generate.

One figure per experiment/model, written into the run directory as `prior_pushforward.png`.
What it shows -- and why each block is in it -- is in `src/pushforward.py`; this script is the
Hydra wiring: draw from the model's own simulator, redraw the priors this study's test cases
shift, and hand both to the figure.

The simulator is the *training* one, bands and all, so for studies 5 and 6 the draw is already
restricted to the band that model was trained under and the figure shades it. The test cases
enter as outlines (studies 2 and 3, which move a prior hyperparameter) and as marks on the
accuracy and response-time panels (studies 5 and 6, which move a band) -- so the same figure
says what the approximator saw and how far its test cases depart from it.

No checkpoint is read, so this can be run before `train_npe` as a check that a configuration
generates plausible data, and kept afterwards as the record of what was trained on.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_mcmc_param_names
from pipeline import setup
from pushforward import MAX_ROUNDS, case_bands, case_overlays, prior_pushforward_figure, pushforward_num_obs
from timing import run_labels

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prior_pushforward(cfg: DictConfig):
    """Write `prior_pushforward.png` for this experiment/model."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    # Studies 5 and 6 draw through a rejection band, and the narrowest of them accept a few
    # percent of what they simulate. See `pushforward.MAX_ROUNDS` for why the figure is allowed
    # more rounds to collect its draw than a training batch is.
    simulator.max_rounds = max(simulator.max_rounds, MAX_ROUNDS)

    artifacts, cases, _ = setup(cfg)

    # `mcmc_param_names`, not the adapter's inference variables: for a hierarchical model the
    # hyperparameter it is amortized over is drawn here too, and it is the one the test cases of
    # studies 2 and 3 sweep, so leaving it out would hide the axis the study is about.
    param_names = get_mcmc_param_names(cfg)

    num_obs = pushforward_num_obs(cfg)
    logger.info("Drawing %s datasets of %s trials", cfg["pushforward_num_draws"], num_obs)

    draws = simulator.sample(cfg["pushforward_num_draws"], num_obs=np.array(num_obs))

    # Prior draws only, one per case that shifts a hyperparameter -- cheap even for the banded
    # studies, whose *data* would be expensive to simulate once per case.
    overlays = case_overlays(simulator.prior_simulator, cases, cfg["pushforward_num_draws"], param_names)

    labels = run_labels()
    figure = prior_pushforward_figure(
        draws,
        param_names,
        num_examples=cfg["pushforward_num_examples"],
        overlays=overlays,
        trained_bands=simulator.bands,
        test_bands=[case_bands(case.sim_kwargs) for case in cases],
        title=f"{labels['experiment']} / {labels['model']}: prior pushforward",
    )

    path = artifacts.figure("prior_pushforward.png")
    artifacts.ensure(path)
    logger.info("Writing %s", path)
    figure.savefig(path, dpi=200)


if __name__ == "__main__":
    prior_pushforward()
