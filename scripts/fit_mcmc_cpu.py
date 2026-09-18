"""Fit ground-truth MCMC posteriors on the CPU, one job per test case.

`mcmc.fit_mcmc_cpu_batch` runs BlackJAX NUTS over every CPU core the job has: each chain of
each dataset is an independent fit, and those fits are dealt out over the cores -- `pmap` over
devices, `vmap` over the fits each one holds. Every chain adapts its own tuning from its own
draw out of the prior. The log-density, the prior to start from, the
parameterization and the transform all come from the `mcmc` config group, so this script has
no model-specific branching. The stage itself lives in `src/mcmc_stage.py`, shared with
`scripts/fit_mcmc_gpu.py`.

The device split has to happen before JAX initialises its backend, and JAX does that on its
first array operation rather than at import -- so `configure_cpu_devices` runs on the first
line, ahead of every import that pulls JAX in. That is the whole reason `src/cpu_devices.py`
exists as a module importing nothing but `os`.
"""

from cpu_devices import configure_cpu_devices

# Before anything imports JAX. See the module docstring.
configure_cpu_devices()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from mcmc_stage import run_fit_mcmc  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    """Fit every dataset of the selected case(s) on the CPU and save the posterior samples."""
    run_fit_mcmc(cfg, task="fit")


if __name__ == "__main__":
    fit_mcmc()
