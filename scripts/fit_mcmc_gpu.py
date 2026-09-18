"""Fit ground-truth MCMC posteriors on the GPU, one job per test case.

The same stage as `scripts/fit_mcmc_cpu.py` (`src/mcmc_stage.py`), with JAX left on its
default backend instead of split into one CPU device per core. `mcmc.fit_mcmc_cpu_batch`
`pmap`s over whatever devices are visible, so on one GPU all `num_chains * num_datasets`
(chain, dataset) fits of a block run in a single `vmap`, and on several GPUs they are dealt out
across them. Every fit gets the same RNG key it gets on the CPU and writes to the same
artifact, so the draws agree with a CPU fit up to floating-point rounding (which NUTS
amplifies) and the `check_*` stages read either without a flag -- which also means a GPU run
overwrites a CPU run's `mcmc_samples/` in the same run directory, and vice versa.

The trade against the CPU stage: a GPU leapfrog step is far cheaper, but a `vmap`ped NUTS
advances its fits in lockstep, so every step of all 400 waits for the deepest trajectory among
them, where the CPU stage holds one or two fits per core. Which one wins depends on the case;
the blocks are timed under `task=fit_gpu` in `timing/fit_mcmc.csv` so the two can be compared
on the same case.

Nothing here pins the platform, so a job without a visible GPU would silently fall back to one
CPU device and run every fit on one core. That is refused rather than warned about.
"""

import hydra
import jax
from omegaconf import DictConfig
from mcmc_stage import run_fit_mcmc


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    """Fit every dataset of the selected case(s) on the GPU and save the posterior samples."""
    if jax.default_backend() != "gpu":
        msg = (
            f"JAX sees no GPU (backend: {jax.default_backend()!r}); use scripts/fit_mcmc_cpu.py "
            "to fit on the CPU."
        )
        raise RuntimeError(msg)

    run_fit_mcmc(cfg, task="fit_gpu")


if __name__ == "__main__":
    fit_mcmc()
