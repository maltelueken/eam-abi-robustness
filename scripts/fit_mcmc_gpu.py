"""Fit ground-truth MCMC posteriors on the GPU, one job per test case.

`mcmc.fit_mcmc_gpu_batch` vmaps BlackJAX NUTS over chains *and* over every dataset in the
case at once. The log-density, its parameter vector and the unconstraining transform all
come from the `mcmc` config group, so this script has no model-specific branching.

`vmap` needs one shape for the whole batch, so a case whose datasets differ in trial count --
the empirical study's, where every subject keeps their own trials -- is fitted one trial count
at a time (`pipeline.num_obs_groups`) and the draws are put back in dataset order. Each group
is a fresh XLA compilation; the empirical files have a handful of distinct counts, with most
subjects in one group, so that is cheap next to fitting the datasets one at a time. A
simulated case is a single group and is the one call this script always made.

Each block is timed into `timing/fit_mcmc.csv` -- the cost amortization is measured against,
so it is recorded on the same data, hardware and chain configuration the paper's fits used.
`positions.block_until_ready()` is inside the timed block on purpose: JAX dispatches
asynchronously, and timing around a call that returns before the device has run would report
the dispatch. The first block of a run also pays for its XLA compilation, once per distinct
trial count.
"""

import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from config import get_mcmc_param_names
from mcmc import save_mcmc_posterior
from pipeline import load_case_data, num_obs_groups, restore_dataset_order, select_cases, setup
from timing import TimingLog, run_labels

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    """Fit every dataset of the selected case(s) and save the posterior samples."""
    artifacts, cases, _ = setup(cfg)

    timings = TimingLog(artifacts.csv("timing", "fit_mcmc"), run_labels())

    make_logdensity_fn = instantiate(cfg["mcmc_model_fun"])
    to_unconstrained = instantiate(cfg["mcmc_to_unconstrained"])
    fit_fun = instantiate(cfg["mcmc_sampling_fun"])

    for case in select_cases(cases, cfg["case"]):
        blocks = num_obs_groups(load_case_data(case, artifacts))

        per_block = []

        for _, block in blocks:
            data_x = block["x"]

            logger.info(
                "Fitting MCMC for case %s: %s datasets with %s trials each",
                case.key,
                data_x.shape[0],
                data_x.shape[1],
            )

            with timings.timed(
                "fit",
                case=case.key,
                num_datasets=data_x.shape[0],
                num_obs=data_x.shape[1],
                num_draws=cfg["mcmc_sampling_fun"]["num_steps_sampling"],
                num_warmup=cfg["mcmc_sampling_fun"]["num_steps_warmup"],
                num_chains=cfg["mcmc_sampling_fun"]["num_chains"],
            ):
                positions, infos = fit_fun(
                    data=data_x,
                    make_logdensity_fn=make_logdensity_fn,
                    to_unconstrained=to_unconstrained,
                )
                positions.block_until_ready()

            logger.info("Divergence rate: %s", float(np.mean(infos.is_divergent)))

            per_block.append(np.asarray(positions))

        positions = restore_dataset_order(blocks, per_block)

        path = artifacts.mcmc_samples(case)
        artifacts.ensure(path)
        logger.info("Saving MCMC samples to %s", path)
        save_mcmc_posterior(path, positions, get_mcmc_param_names(cfg))


if __name__ == "__main__":
    fit_mcmc()
