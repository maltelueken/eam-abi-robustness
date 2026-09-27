"""The `fit_mcmc` pipeline stage, shared by `scripts/fit_mcmc_cpu.py` and `scripts/fit_mcmc_gpu.py`.

The two scripts differ only in which JAX devices exist when this runs: the CPU script splits
the host into one device per core before JAX starts, the GPU script leaves JAX on its default
backend. `mcmc.fit_mcmc_cpu_batch` is device-agnostic -- it `pmap`s over whatever
`jax.local_devices()` holds and `vmap`s the (chain, dataset) fits each device gets -- so on one
GPU every fit of a block runs in a single `vmap`, and on several GPUs they are dealt out
across them. Everything else, including the RNG key each fit receives and the artifact the
draws are written to, is the same for both, which is what lets the downstream `check_*` stages
read either without knowing which one ran.

`vmap` needs one shape for the whole batch, so a case whose datasets differ in trial count --
the empirical study's, where every subject keeps their own trials -- is fitted one trial count
at a time (`pipeline.num_obs_groups`) and the draws are put back in dataset order. Each group
is a fresh XLA compilation; a simulated case is a single group.

Each block is timed into `timing/fit_mcmc.csv` under `task`, which is what tells a CPU row
from a GPU row once the tables are concatenated. `positions.block_until_ready()` is inside the
timed block on purpose: JAX dispatches asynchronously, and timing around a call that returns
before the device has run would report the dispatch. The first block of a run also pays for
its XLA compilation, once per distinct trial count.
"""

import logging
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from config import get_mcmc_param_names
from mcmc import save_mcmc_posterior
from pipeline import load_case_data
from pipeline import num_obs_groups
from pipeline import restore_dataset_order
from pipeline import select_cases
from pipeline import setup
from timing import TimingLog
from timing import run_labels

logger = logging.getLogger(__name__)


def run_fit_mcmc(cfg: DictConfig, task: str):
    """Fit every dataset of the selected case(s) and save the posterior samples.

    Args:
        cfg: the composed Hydra config.
        task: the `task` label the blocks are timed under (`"fit"` on the CPU, `"fit_gpu"` on
            the GPU), so the hardware a fit ran on survives in the timing table.
    """
    artifacts, cases, _ = setup(cfg)

    timings = TimingLog(artifacts.csv("timing", "fit_mcmc"), run_labels())

    make_logdensity_fn = instantiate(cfg["mcmc_model_fun"])
    prior_sample_fn = instantiate(cfg["mcmc_prior_sample_fun"])
    spec = instantiate(cfg["mcmc_spec"])
    transform = instantiate(cfg["mcmc_transform"])
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
                task,
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
                    prior_sample_fn=prior_sample_fn,
                    spec=spec,
                    transform=transform,
                )
                positions.block_until_ready()

            logger.info("Divergence rate: %s", float(np.mean(infos.is_divergent)))

            per_block.append(np.asarray(positions))

        positions = restore_dataset_order(blocks, per_block)

        path = artifacts.mcmc_samples(case)
        artifacts.ensure(path)
        logger.info("Saving MCMC samples to %s", path)
        save_mcmc_posterior(path, positions, get_mcmc_param_names(cfg))
