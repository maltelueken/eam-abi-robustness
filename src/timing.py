"""Wall-clock timing for the pipeline's expensive stages.

Amortization is a claim about cost: a network is trained once and then answers every dataset,
where MCMC pays its full price per dataset. That claim is only worth reporting if the numbers
come from the runs the paper's results come from -- a training timed on a laptop, or an
inference timed at a different batch size, answers a different question -- so each stage
records its own timing next to the artifacts it produced, in the run directory Hydra put it in.

Four tasks are timed, and they are the four the comparison is made of:

* `train_npe` -- fitting the ensemble, the one-off cost amortization pays up front;
* `predict_npe` -- drawing a case's posteriors, the cost it buys;
* `fit_mcmc_cpu` -- the ground-truth fits for the same case, the cost being amortized away;
* the architecture sweep, which is a cost of arriving at the method rather than of any one
  run and is read back out of the Optuna study instead (`sweeper.sweep_timing`).

Every row carries both clocks. `wall_seconds` is what a user waits, and it is what the
comparison is about; `cpu_seconds` (`time.process_time`, which sums over threads and stops
while the process sleeps) says how much of a wall-clock number is work in this process rather
than waiting -- on the GPU stages most of the wall time is spent on a device the process clock
cannot see, so the two differ by design.

Timing a GPU stage means the timed block has to *end* on the device. JAX dispatches
asynchronously, so a `with timed(...)` around a call returning an unmaterialized array times
the dispatch and reports a fraction of a second for a job that ran for hours. Every call site
either blocks explicitly (`positions.block_until_ready()`) or converts to numpy inside the
block; that is not decoration, it is what makes the number real.

Rows are flushed as they are taken rather than at the end of the run, so a job the cluster
kills at its walltime still leaves behind the timings of everything it finished -- which is
exactly the job whose timing is most interesting.
"""

import contextlib
import csv
import logging
import os
import time
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

# One fixed schema for every task, so that the timing tables of every stage, experiment and
# model concatenate into one table. A task fills in the columns that apply to it and leaves the
# rest blank -- `num_chains` means nothing to an NPE, `epochs` means nothing to a fit.
FIELDS = (
    # Which run this is, so a pile of these files is still self-describing once collected out
    # of the directory tree that named them.
    "experiment",
    "model",
    "inference_method",
    # What was timed.
    "task",
    "case",
    # What it was timed on. `num_datasets` is how many datasets the block handled -- for
    # training, how many simulated datasets the fit consumed; `num_obs` their trial count,
    # blank where a case's datasets do not share one (study 4) or the design randomizes it.
    "num_datasets",
    "num_obs",
    # The cost knobs, per task: posterior draws and chains for a fit, ensemble members for a
    # network, epochs for a training run.
    "num_draws",
    "num_warmup",
    "num_members",
    "num_chains",
    "epochs",
    "wall_seconds",
    "cpu_seconds",
)


def run_labels():
    """The experiment / model / inference-method the running Hydra job composed.

    Read from `HydraConfig` rather than from `cfg`, because these are group *choices* and not
    config values: `cfg` knows what the chosen files contain, not which ones were chosen. The
    same three choices name the run directory, so this only writes into the table what the path
    already says -- the point being that the row survives being copied out of that path.

    Returns:
        A dict of the three label columns.
    """
    choices = HydraConfig.get().runtime.choices

    return {
        "experiment": choices["experiment"],
        "model": choices["model"],
        "inference_method": choices["inference-method"],
    }


class TimingLog:
    """Timings for one run, appended to a CSV as they are taken.

    Args:
        path: the CSV to write. Its parent directory is created on the first row, and an
            existing file is replaced -- a rerun of a stage owns its timings the same way it
            owns every other artifact it writes.
        labels: columns repeated on every row, typically `run_labels()`.
    """

    def __init__(self, path, labels=None):
        self.path = path
        self.labels = dict(labels or {})
        self._header_written = False

    @contextlib.contextmanager
    def timed(self, task, **context):
        """Time the block and append one row describing it.

        The row is written even when the block raises, so a stage that died halfway still
        accounts for what it spent getting there.

        Args:
            task: what is being timed, e.g. `"train"`, `"sample"`, `"fit"`.
            **context: any of `FIELDS` describing the work -- `case`, `num_datasets`, ...

        Yields:
            None.
        """
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        try:
            yield
        finally:
            self.record(
                task,
                time.perf_counter() - start_wall,
                time.process_time() - start_cpu,
                **context,
            )

    def record(self, task, wall_seconds, cpu_seconds, **context):
        """Append one already-measured timing.

        Args:
            task: what was timed.
            wall_seconds: elapsed wall-clock time.
            cpu_seconds: elapsed process CPU time.
            **context: any of `FIELDS` describing the work.

        Raises:
            ValueError: if `context` names a column outside `FIELDS`, which is a typo that
                would otherwise be written to a column nothing reads.
        """
        row = {
            **self.labels,
            "task": task,
            **context,
            "wall_seconds": round(wall_seconds, 3),
            "cpu_seconds": round(cpu_seconds, 3),
        }

        unknown = sorted(set(row) - set(FIELDS))

        if unknown:
            msg = f"Timing columns {unknown} are not in the timing schema {list(FIELDS)}."
            raise ValueError(msg)

        self._append(row)

        logger.info(
            "Timing: %s%s took %.1f s wall, %.1f s cpu",
            task,
            f" ({context['case']})" if context.get("case") else "",
            wall_seconds,
            cpu_seconds,
        )

    def _append(self, row):
        """Write one row, creating the file and its header on first use."""
        if not self._header_written:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "a" if self._header_written else "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")

            if not self._header_written:
                writer.writeheader()
                self._header_written = True

            writer.writerow(row)
