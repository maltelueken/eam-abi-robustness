"""Measure whether one architecture trains and samples on this job's GPU, and at what speed.

Every architecture the Optuna sweep can select is trained twice on two different GPUs: as a single
network on a 20 GB MIG slice during the sweep (`slurm/sweep.sh`), and later as a five-member
ensemble on a 40 GB A100 (`slurm/submit.sh`). Neither job finds out in advance whether a point of
the search space fits. One that does not fit aborts the whole `--multirun`. One that fits only
because XLA traded speed for memory trains more slowly, and nothing reports it. This script
measures both for one architecture. `slurm/gpu_budget.sh` runs it over the corners of the search
space, one phase per process:

* `train` repeats one sweep trial (or one `train_npe` job): `approximator.fit` on the real
  simulator at each trial count of the training grid, followed by `train_npe`'s diagnostic
  sampling in the same process. Keeping them in one process is deliberate: a trial once died
  sampling its diagnostics, because training had grown XLA's pool far enough that the CUDA driver
  had no room left to instantiate a graph.
* `plan` compiles the train step at the largest trial count twice: once under the memory limit
  XLA:GPU normally plans against, and once with that limit lifted
  (`xla_gpu_memory_limit_slop_factor`). It then times both on the same batch. If the plans and
  the step times agree, memory pressure cost nothing. If the limited plan is smaller or slower,
  the compiler worked around memory, and that is the slowdown.
* `predict` repeats `predict_npe`'s sampling (members kept apart, `test_batch_size` datasets at
  once) at the largest trial count any test case has.

`peak_bytes_in_use` cannot be reset within a process, so phases do not share one, and in `train`
the trial counts run in ascending order. Each row's peak therefore belongs to the largest count
so far. The networks in `plan` and `predict` are freshly initialised: memory does not depend on
the weights, but the adaptive integrator's step count does, so sampling times from `predict` are
indicative only.

XLA's own warnings go to stderr, not to this process. `slurm/gpu_budget.sh` greps each run's log
for them.
"""

import argparse
import csv
import logging
import math
import os
import pathlib
import sys
import time

# Same reason as in `utils`: keras picks its backend at import time.
os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import keras
import numpy as np
from hydra import compose
from hydra import initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import ChoiceSweep
from hydra.core.override_parser.types import IntervalSweep
from hydra.core.override_parser.types import RangeSweep
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ensemble import num_members
from ensemble import sample_members

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "conf"

GIB = 2**30

# 100x the device's memory: far beyond anything that can be allocated, so the compiler plans
# as if memory were unlimited. The flag is a percentage, with a default of 95.
UNLIMITED_SLOP_FACTOR = 10_000

FIELDS = (
    "label",
    "approximator",
    "members",
    "experiment",
    "model",
    "device",
    "phase",
    "variant",
    "num_obs",
    "num_datasets",
    "num_samples",
    "steps",
    # Wall-clock up to the end of the warm-up steps (train) or of the first call (sampling),
    # which includes XLA compilation, and the steady-state cost per step or per call after it.
    "first_seconds",
    "steady_seconds",
    # Bytes XLA's allocator has handed out at its peak, and how much it has taken from the
    # driver to do so (it does not give memory back). What `capacity_gib - pool_gib` leaves is
    # all the CUDA driver has for everything outside the pool, CUDA graphs included.
    "peak_gib",
    "pool_gib",
    "limit_gib",
    "capacity_gib",
    "plan_temp_gib",
    "status",
    "message",
    "architecture",
)


class Results:
    """Append one row per measurement to a CSV shared by every run of a job.

    Each row is written as soon as it is taken, so a run that dies at the next step still
    leaves behind what it measured.
    """

    def __init__(self, path, base):
        self.path = pathlib.Path(path)
        self.base = base

    def write(self, **row):
        """Fill in the memory columns from the device's allocator, then append `row`."""
        stats = jax.local_devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit", math.nan) / GIB

        row = self.base | {
            "peak_gib": stats.get("peak_bytes_in_use", math.nan) / GIB,
            "pool_gib": stats.get("peak_pool_bytes", stats.get("peak_bytes_reserved", math.nan)) / GIB,
            "limit_gib": limit,
            "capacity_gib": limit / memory_fraction(),
            "status": "ok",
        } | row

        row = {key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()

        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)

        logger.info("%s", " | ".join(f"{key}={row.get(key, '')}" for key in FIELDS[6:21]))


def memory_fraction():
    """The fraction of device memory XLA's pool is capped at, as the CUDA client reads it."""
    value = os.environ.get("XLA_CLIENT_MEM_FRACTION") or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    return float(value) if value else 0.75


class FixedNumObs:
    """The training simulator, with the trial count pinned to one value.

    The design simulator draws `num_obs` per batch, and XLA compiles one train step per trial
    count. A run's peak memory is therefore set by the largest count, and only reached once that
    count is drawn. Pinning it gives every count a known number of steps to be measured over.
    """

    def __init__(self, simulator, num_obs):
        self.simulator = simulator
        self.num_obs = num_obs

    def sample(self, batch_size, **kwargs):
        """Draw `batch_size` datasets of `num_obs` trials each."""
        return self.simulator.sample(batch_size, num_obs=np.array(self.num_obs), **kwargs)


class StepTimer(keras.callbacks.Callback):
    """Record when each train step finishes.

    JAX dispatches asynchronously, so the step has to be waited for before the clock is read. The
    logs are outputs of the jitted step, and converting them to numpy waits for it.
    """

    def __init__(self):
        super().__init__()
        self.ends = []

    def on_train_batch_end(self, batch, logs=None):  # noqa: ARG002 -- Keras's signature
        """Wait for the step's outputs, then take the time."""
        for value in (logs or {}).values():
            np.asarray(value)
        self.ends.append(time.perf_counter())


def search_space_corner(args):
    """The sweep's search space at its top corner, read from the config the sweep runs under.

    Every swept knob only adds memory as it grows (depths, widths, expansion factor, seeds, summary
    dimension), so the top corner is the most expensive architecture the sweep can select. The
    space is not restated here. It comes from `hydra.sweeper.params`, so this follows
    `conf/inference-method/` and `conf/summary-method/` as they change. hydra-optuna maps
    `range(a, b)` to an integer distribution whose upper end is *inclusive* (see
    `conf/summary-method/set_transformer.yaml`), so `b` is the corner.
    """
    cfg = compose_config(args, ["sweeper=optuna", "approximator=continuous_approximator"])
    parser = OverridesParser.create()
    corner = {}

    for key, spec in cfg["hydra"]["sweeper"]["params"].items():
        sweep = parser.parse_override(f"{key}={spec}").value()

        if isinstance(sweep, ChoiceSweep):
            corner[key] = max(sweep.list)
        elif isinstance(sweep, RangeSweep):
            corner[key] = max(sweep.start, sweep.stop)
        elif isinstance(sweep, IntervalSweep):
            corner[key] = max(sweep.start, sweep.end)
        else:
            msg = f"Cannot take the corner of '{key}: {spec}'."
            raise TypeError(msg)

    return corner


def compose_config(args, overrides):
    """Compose the experiment's config, with `HydraConfig` set so `${hydra:...}` resolves."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={args.experiment}", f"model={args.model}", *overrides],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)

    return cfg


def architecture_overrides(args):
    """The search-space corner if one was asked for, with explicit `key=value` overrides on top."""
    overrides = search_space_corner(args) if args.corner == "max" else {}

    for item in args.overrides:
        key, _, value = item.partition("=")
        overrides[key] = value

    return [f"{key}={value}" for key, value in overrides.items()]


def describe_architecture(cfg):
    """The architecture as trained, one `key=value` per swept knob, for the results table."""
    family = OmegaConf.load(CONFIG_DIR / "architecture" / f"{cfg['architecture_family']}.yaml")
    return " ".join(f"{key}={cfg[key]}" for key in family if key != "architecture_family")


def first_batch(approximator, simulator, num_obs, batch_size):
    """One adapted training batch at `num_obs`, as tensors, in the layout the train step takes.

    It is built through `build_dataset`, so an ensemble gets the per-member layout its train step
    expects.
    """
    dataset = approximator.build_dataset(
        simulator=FixedNumObs(simulator, num_obs),
        batch_size=batch_size,
        num_batches=1,
        adapter=approximator.adapter,
        workers=1,
    )
    return keras.tree.map_structure(keras.ops.convert_to_tensor, dataset[0])


def time_sampling(sample):
    """Call `sample` twice, and return the first call's wall-clock (with compilation) and the second's."""
    seconds = []

    for _ in range(2):
        start = time.perf_counter()
        sample()
        seconds.append(time.perf_counter() - start)

    return seconds


def train(cfg, approximator, simulator, args, results):
    """Fit at every trial count of the grid, then sample `train_npe`'s diagnostics."""
    approximator.compile(instantiate(cfg["optimizer"], _convert_="partial"))

    for num_obs in sorted(args.train_num_obs):
        timer = StepTimer()
        start = time.perf_counter()

        # As in `train_npe`, down to `workers=1`. Only the trial count is pinned.
        approximator.fit(
            simulator=FixedNumObs(simulator, num_obs),
            epochs=1,
            num_batches=args.warmup + args.steps,
            batch_size=cfg["batch_size"],
            callbacks=[timer],
            workers=1,
            verbose=0,
        )

        results.write(
            phase="train",
            num_obs=num_obs,
            num_datasets=cfg["batch_size"],
            steps=args.steps,
            first_seconds=timer.ends[args.warmup - 1] - start,
            steady_seconds=(timer.ends[-1] - timer.ends[args.warmup - 1]) / args.steps,
        )

    # `diag_sample_batch_size` datasets per call, which is the chunk `train_npe` hands to
    # `sample`. Memory depends on the chunk and not on `diag_batch_size`, so the full count would
    # only multiply the wall-clock.
    chunk = cfg["diag_sample_batch_size"]

    for num_obs in cfg["diag_num_obs"]:
        conditions = simulator.sample(chunk, num_obs=np.array(num_obs))

        first, steady = time_sampling(
            lambda conditions=conditions: approximator.sample(
                num_samples=cfg["diag_num_posterior_samples"],
                conditions=conditions,
                batch_size=chunk,
            ),
        )

        results.write(
            phase="diagnostics",
            num_obs=num_obs,
            num_datasets=chunk,
            num_samples=cfg["diag_num_posterior_samples"],
            first_seconds=first,
            steady_seconds=steady,
            message=f"x{cfg['diag_batch_size'] // chunk} calls per trial count in train_npe",
        )


def run_steps(compiled, state, batch, warmup, steps):
    """Time `steps` calls of a compiled train step, threading its donated state through."""
    for _ in range(warmup):
        _, state = compiled(state, batch)

    jax.block_until_ready(state)
    start = time.perf_counter()

    for _ in range(steps):
        _, state = compiled(state, batch)

    jax.block_until_ready(state)
    return (time.perf_counter() - start) / steps


def plan(cfg, approximator, simulator, args, results):
    """Compile the largest train step with and without XLA's memory limit, and time both."""
    num_obs = max(args.train_num_obs)

    approximator.compile(instantiate(cfg["optimizer"], _convert_="partial"))
    batch = first_batch(approximator, simulator, num_obs, cfg["batch_size"])
    approximator.build_from_data(batch)
    approximator.optimizer.build(approximator.trainable_variables)

    state = approximator._get_jax_state(  # noqa: SLF001 -- what Keras's own fit loop passes
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        metrics_variables=True,
        purge_model_variables=False,
    )

    # The step donates its state, as Keras's own does. The first variant donates the model's own
    # buffers, so its peak matches a real fit. The second starts from a host copy.
    host_state = jax.device_get(state)
    lowered = jax.jit(approximator.train_step, donate_argnums=0).lower(state, batch)

    measured = {}

    for variant, options in (
        ("limited", None),
        ("unlimited", {"xla_gpu_memory_limit_slop_factor": UNLIMITED_SLOP_FACTOR}),
    ):
        start = time.perf_counter()
        compiled = lowered.compile(options) if options else lowered.compile()
        compile_seconds = time.perf_counter() - start

        analysis = compiled.memory_analysis()
        temp = analysis.temp_size_in_bytes / GIB if analysis is not None else math.nan

        run_state = state if state is not None else jax.device_put(host_state)
        state = None

        row = {
            "phase": "plan",
            "variant": variant,
            "num_obs": num_obs,
            "num_datasets": cfg["batch_size"],
            "steps": args.steps,
            "first_seconds": compile_seconds,
            "plan_temp_gib": temp,
        }

        try:
            seconds = run_steps(compiled, run_state, batch, args.warmup, args.steps)
        except jax.errors.JaxRuntimeError as error:
            # The unlimited plan is allowed to fail: then it is simply larger than the device. It
            # is still evidence that the limited plan was squeezed to fit.
            results.write(**row, **failure(error))
            measured[variant] = (temp, math.nan)
        else:
            measured[variant] = (temp, seconds)
            message = ""

            if variant == "unlimited":
                limited_temp, limited_seconds = measured["limited"]
                message = (
                    f"limited/unlimited: step time x{limited_seconds / seconds:.3f}; "
                    f"plan {limited_temp:.2f}/{temp:.2f} GiB"
                )

            results.write(**row, steady_seconds=seconds, message=message)

        del compiled, run_state


def predict(cfg, approximator, simulator, args, results):
    """Sample every member at `test_batch_size` datasets at once, as `predict_npe` does."""
    approximator.build_from_data(first_batch(approximator, simulator, min(args.predict_num_obs), cfg["batch_size"]))

    for num_obs in args.predict_num_obs:
        conditions = simulator.sample(cfg["test_batch_size"], num_obs=np.array(num_obs))

        first, steady = time_sampling(
            lambda conditions=conditions: sample_members(
                approximator, conditions, num_samples=cfg["test_num_posterior_samples"],
            ),
        )

        results.write(
            phase="predict",
            num_obs=num_obs,
            num_datasets=cfg["test_batch_size"],
            num_samples=cfg["test_num_posterior_samples"],
            first_seconds=first,
            steady_seconds=steady,
            message="freshly initialised network: sampling time is indicative only",
        )


def failure(error):
    """The row fields for a run that raised: an OOM is a result, anything else is a bug."""
    text = str(error)
    out_of_memory = "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()
    # One line without commas, since the summary splits the table on them.
    first_line = text.strip().splitlines()[0] if text.strip() else type(error).__name__
    return {"status": "oom" if out_of_memory else "error", "message": first_line.replace(",", ";")[:240]}


PHASES = {"train": train, "plan": plan, "predict": predict}


def counts(text):
    """Parse a comma-separated list of trial counts."""
    return [int(item) for item in text.split(",") if item]


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--out", required=True, help="CSV to append the results to")
    parser.add_argument("--label", default="", help="name of this architecture in the results")
    parser.add_argument(
        "--approximator",
        default="continuous_approximator",
        choices=["continuous_approximator", "ensemble_approximator"],
    )
    parser.add_argument("--experiment", default="experiment_2")
    parser.add_argument("--model", default="rdm_simple_meta")
    parser.add_argument(
        "--corner",
        choices=["none", "max"],
        default="none",
        help="start from the top corner of the sweep's search space rather than conf/architecture/",
    )
    # Comma-separated rather than `nargs="+"`, which would swallow the Hydra overrides after it.
    parser.add_argument("--train-num-obs", type=counts, default=[100, 250, 500, 1000], help="e.g. 100,250,500,1000")
    parser.add_argument("--predict-num-obs", type=counts, default=[1200], help="e.g. 1200")
    parser.add_argument("--warmup", type=int, default=5, help="untimed steps; the first includes compilation")
    parser.add_argument("--steps", type=int, default=50, help="timed steps per trial count")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run without a GPU, to smoke-test this script; no memory is reported",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides applied on top of --corner, e.g. summary_embed_depth=2",
    )

    # As in `select_architecture.py`: lets the trailing overrides come after the flags.
    return parser.parse_intermixed_args()


def main():
    """Run one phase for one architecture, and exit non-zero if it did not complete."""
    # Libraries stay at WARNING, so the log holds this script's rows and XLA's own warnings.
    # `force`, because importing bayesflow has already given the root logger a handler.
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    logger.setLevel(logging.INFO)

    args = parse_args()

    device = jax.devices()[0]

    # The CUDA plugin is not a declared dependency, so an environment without it falls back to the
    # CPU without complaint. Every number below would then describe the wrong device.
    if device.platform != "gpu" and not args.allow_cpu:
        sys.exit(f"Expected a GPU, but JAX is running on '{device.platform}'.")

    cfg = compose_config(args, [f"approximator={args.approximator}", *architecture_overrides(args)])

    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")

    results = Results(
        args.out,
        {
            "label": args.label,
            "approximator": args.approximator,
            "members": num_members(approximator),
            "experiment": args.experiment,
            "model": args.model,
            "device": device.device_kind,
            "architecture": describe_architecture(cfg),
        },
    )

    logger.info("[%s] %s on %s: %s", args.phase, args.label, device.device_kind, describe_architecture(cfg))

    try:
        PHASES[args.phase](cfg, approximator, simulator, args, results)
    except Exception as error:
        results.write(phase=args.phase, **failure(error))
        raise


if __name__ == "__main__":
    main()
