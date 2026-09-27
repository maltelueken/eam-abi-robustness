"""`mcmc.fit_mcmc_cpu_batch` spread over several CPU devices.

The device count is fixed before JAX initialises its backend (see `src/cpu_devices.py`), and
pytest's process has long since done that on one device, so each layout runs in a subprocess.
What is pinned: every (chain, dataset) fit keeps its own key whatever the device count -- a
layout that padded, dropped or misplaced a unit would put a different chain's draws in its
slot, far outside rounding -- and the output layout is the one `save_mcmc_posterior` expects.
"""

import subprocess
import sys
import textwrap
import numpy as np

SCRIPT = textwrap.dedent(
    """
    import sys
    from cpu_devices import configure_cpu_devices
    configure_cpu_devices(int(sys.argv[1]))

    import functools
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mcmc import BlockTransform, fit_mcmc_cpu_batch
    from rdm_jax import (
        make_rdm_simple_logdensity, make_rdm_simple_prior_sample, rdm_experiment_simple_jax, rdm_spec,
    )

    assert len(jax.devices()) == int(sys.argv[1])
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    data = jnp.stack([rdm_experiment_simple_jax(k, 1.0, 1.5, 0.3, 1.0, 1.2, 0.3, 50)["x"] for k in keys])
    positions, infos = fit_mcmc_cpu_batch(
        jax.random.key(1), data,
        functools.partial(make_rdm_simple_logdensity, drift_slope_loc=1.5, threshold_scale=1.0),
        prior_sample_fn=make_rdm_simple_prior_sample(drift_slope_loc=1.5, threshold_scale=1.0),
        spec=rdm_spec(), transform=BlockTransform(),
        num_chains=2, num_steps_warmup=20, num_steps_sampling=10,
    )
    assert infos.is_divergent.shape == positions.shape[:3]
    np.save(sys.argv[2], np.asarray(positions))
    """,
)


def fit_on(num_devices, tmp_path):
    out = tmp_path / f"positions_{num_devices}.npy"
    subprocess.run([sys.executable, "-c", SCRIPT, str(num_devices), str(out)], check=True)
    return np.load(out)


def test_fit_mcmc_cpu_batch_is_independent_of_device_count(tmp_path):
    # 1 device: all six units in one vmap. 4 devices: two per device, two padded slots.
    one, four = fit_on(1, tmp_path), fit_on(4, tmp_path)

    assert one.shape == four.shape == (3, 10, 2, 5)
    np.testing.assert_allclose(one, four, atol=1e-3)
