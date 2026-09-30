"""Make BayesFlow's adaptive ODE integrator work with JAX's x64 mode on.

`rdm_jax` turns on `jax_enable_x64` at import, and the pipeline scripts import it, since the
simulator and the MCMC reference need float64. The networks stay float32 (`keras.config.floatx()`).

BayesFlow 2.0.14's `integrate_adaptive`, which flow matching uses with
`steps: adaptive` (e.g. `method: tsit5`), keeps the current time in a `while_loop` and
starts it at the Python float that `FlowMatching` passes in (`0.0` or `1.0`). Under x64 that
float becomes a float64 scalar. The step `time + h` comes out float32, because `h` is floatx.
The accept/reject `keras.ops.cond` then returns float32 from one branch and float64 from the
other, and JAX refuses to trace it. Fixed-step integration (`method: euler`) has no such
`cond`, which is why it never hit this.

The fix casts both endpoints to floatx before the loop starts, so every time value in the loop
is float32, whatever the x64 setting. Without x64 the cast changes nothing.
"""

import functools
import os
import sys

# Same reason as in `utils`: keras picks its backend at import time.
os.environ.setdefault("KERAS_BACKEND", "jax")

import bayesflow.utils.integrate  # noqa: F401 -- loads the module patched below
import keras

# `bayesflow.utils.integrate` is also the name of the function that module re-exports, so the
# attribute path would resolve to the function. The module comes from `sys.modules` instead.
_integrate_module = sys.modules["bayesflow.utils.integrate"]
_integrate_adaptive = _integrate_module.integrate_adaptive


@functools.wraps(_integrate_adaptive)
def _integrate_adaptive_floatx(fn, state, start_time, stop_time, *args, **kwargs):
    dtype = keras.config.floatx()
    return _integrate_adaptive(
        fn,
        state,
        keras.ops.convert_to_tensor(start_time, dtype=dtype),
        keras.ops.convert_to_tensor(stop_time, dtype=dtype),
        *args,
        **kwargs,
    )


# `integrate` looks the name up in its module's globals on each call, so replacing the
# attribute is enough.
_integrate_module.integrate_adaptive = _integrate_adaptive_floatx
