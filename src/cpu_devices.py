"""Give JAX one CPU device per MCMC chain, before anything imports JAX.

`scripts/fit_mcmc_cpu.py` runs each chain on its own device (see `mcmc.fit_mcmc_cpu_batch`),
and JAX exposes the host CPU as a *single* device unless told otherwise. Splitting it is an
XLA flag, and XLA reads its flags when the backend is initialised -- which happens on the
first array operation, not at `import jax`. So this has to run before any of that, which is
why it lives in a module of its own that imports nothing but `os`: `scripts/fit_mcmc_cpu.py`
calls it on its first line, ahead of the project imports that pull JAX in.

It is deliberately *not* called from `src/mcmc.py`. That module is imported by `pipeline`,
hence by every stage including `train_npe`, and pinning the platform to CPU there would take
the GPU away from training.
"""

import os

#: Chains per fit, and therefore CPU devices. Matches `mcmc_sampling_fun.num_chains` in
#: `conf/experiment/experiment_1.yaml`; override both together.
DEFAULT_NUM_DEVICES = 4

#: Environment variable a caller can use to raise or lower that, e.g. under a SLURM
#: allocation with a different `--cpus-per-task`. `slurm/submit_all.sh` exports it.
ENV_VAR = "MCMC_NUM_CPU_DEVICES"


def requested_num_devices():
    """How many CPU devices to ask for: `$MCMC_NUM_CPU_DEVICES`, or the default."""
    return int(os.environ.get(ENV_VAR, DEFAULT_NUM_DEVICES))


def configure_cpu_devices(num_devices=None):
    """Pin JAX to the CPU and split it into `num_devices` addressable devices.

    Both settings are `setdefault`-like: an `XLA_FLAGS` that already names the device count,
    or an explicit `JAX_PLATFORMS`, is left alone, so a caller can still override from the
    environment.

    Returns the device count that will be in force.
    """
    num_devices = requested_num_devices() if num_devices is None else int(num_devices)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={num_devices}".strip()

    return num_devices
