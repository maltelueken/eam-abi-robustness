"""Write the architecture an Optuna sweep selected into `conf/architecture/<family>.yaml`.

`slurm/sweep.sh` runs this as its last step, so the winner of a family's sweep reaches every
experiment of that family without anyone editing a config by hand -- which is the point of
`conf/architecture/` being a group of its own. The file it writes is git-tracked, so what the
networks were trained at stays reviewable and reproducible after the sweep database has been
archived.

The sweep has three objectives, so "best" is a Pareto front rather than a trial; the rule that
picks one of them lives in `sweeper.select_best_trial`, next to its tests. Everything here is the
plumbing around it: which study to read, and where its answer goes.

The study name, the storage URL and the set of swept parameters are never restated here -- they
are read back out of a composed config, so `conf/sweeper/optuna.yaml` and the
`inference-method`/`summary-method` groups stay their single home.
"""

import argparse
import datetime
import logging
import pathlib
import numpy as np
import optuna
import pandas as pd
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from sweeper import render_architecture
from sweeper import select_best_trial
from sweeper import sweep_timing

logger = logging.getLogger(__name__)

CONFIG_PATH = "../conf"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The three objectives `train_npe` returns, in order, for the provenance header.
OBJECTIVE_NAMES = ("rmse", "contraction", "calibration_error")


def compose_sweep_config(model: str, experiment: str, overrides: list[str]) -> DictConfig:
    """Compose the config the sweep ran under, with the `hydra.sweeper` node resolved."""
    with initialize(version_base=None, config_path=CONFIG_PATH):
        return compose(
            config_name="config",
            overrides=[
                "sweeper=optuna",
                "approximator=continuous_approximator",
                f"experiment={experiment}",
                f"model={model}",
                *overrides,
            ],
            return_hydra_config=True,
        )


def check_search_space(study: optuna.Study, trial: optuna.trial.FrozenTrial, swept: set[str]) -> None:
    """Fail if the study searched a different set of parameters than the config sweeps.

    The parameter names are the keys the architecture file has to define, so a database predating
    a change to the search space is caught here rather than at training time -- where it would
    surface as an unresolvable interpolation in a network node, hours into a cluster job.

    Raises:
        RuntimeError: if the two sets differ.
    """
    if swept != set(trial.params):
        msg = (
            f"study '{study.study_name}' searched {sorted(trial.params)}, but the config sweeps "
            f"{sorted(swept)}; re-run the sweep against the current search space"
        )
        raise RuntimeError(msg)


def write_trials_csv(study: optuna.Study, best: optuna.trial.FrozenTrial, path: pathlib.Path) -> None:
    """Dump the study's trials for `visualization/appendix_optimization.R`.

    The winner is marked in the file rather than left for the figure to hardcode by trial number
    -- which it did, and which silently became a lie the moment the sweep was rerun.
    """
    frame = study.trials_dataframe()
    frame["is_best"] = frame["number"] == best.number

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_timing_csv(study: optuna.Study, path: pathlib.Path) -> dict:
    """Write what the sweep cost, and return it.

    The architecture search is the pipeline's largest single expense and the one stage with no
    run directory of its own to record it in, so its timing is taken from the study here --
    alongside the per-trial `duration` column the trials dump already carries -- rather than
    left to be reconstructed from SLURM logs after the fact.
    """
    timing = sweep_timing(study)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([timing]).to_csv(path, index=False)

    return timing


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="a model of the family whose sweep to read, e.g. rdm_simple")
    parser.add_argument("--experiment", default="experiment_1", help="the experiment the sweep ran under")
    parser.add_argument(
        "--weights",
        default="1,1,1",
        help="comma-separated weights on (rmse, contraction, calibration error) in the distance",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the config instead of writing it")
    parser.add_argument("--no-csv", action="store_true", help="skip the trials and timing dumps")
    parser.add_argument("overrides", nargs="*", help="further Hydra overrides the sweep was given")

    # `parse_intermixed_args` rather than `parse_args`: the trailing `nargs="*"` positional is
    # otherwise only recognized when it comes before every flag, which is a trap for a script
    # whose whole job is to be handed the same overrides the sweep was given.
    return parser.parse_intermixed_args()


def main() -> None:
    """Select the best trial of a family's sweep and write it back into `conf/architecture/`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    cfg = compose_sweep_config(args.model, args.experiment, args.overrides)
    family = cfg["architecture_family"]
    sweeper = cfg["hydra"]["sweeper"]

    study = optuna.load_study(study_name=sweeper["study_name"], storage=sweeper["storage"])
    weights = np.array([float(w) for w in args.weights.split(",")], dtype=float)

    best, score = select_best_trial(study, weights)
    check_search_space(study, best, set(sweeper["params"]))

    objectives = ", ".join(f"{name}={value:.4g}" for name, value in zip(OBJECTIVE_NAMES, best.values, strict=True))
    provenance = [
        f"study:     {study.study_name}  ({sweeper['storage']})",
        f"trial:     {best.number} of {len(study.trials)}, Pareto front of {len(study.best_trials)}",
        f"objective: {objectives}",
        f"selected:  nearest ideal point, weights {tuple(float(w) for w in weights)}, distance {score:.4g}",
        f"written:   {datetime.datetime.now(tz=datetime.UTC).isoformat(timespec='seconds')}",
    ]

    rendered = render_architecture(family, dict(best.params), provenance)

    if args.dry_run:
        logger.info("%s", rendered)
        return

    path = PROJECT_ROOT / "conf" / "architecture" / f"{family}.yaml"
    path.write_text(rendered)
    logger.info("wrote %s from trial %d", path, best.number)

    if not args.no_csv:
        csv_path = PROJECT_ROOT / "multirun" / f"trials_{family}.csv"
        write_trials_csv(study, best, csv_path)
        logger.info("wrote %s", csv_path)

        timing_path = PROJECT_ROOT / "multirun" / f"timing_{family}.csv"
        timing = write_timing_csv(study, timing_path)
        logger.info(
            "wrote %s: %d trials in %.1f h wall, %.1f h of trials, median %.2f h per trial",
            timing_path,
            timing["num_trials"],
            timing["wall_seconds"] / 3600,
            timing["trial_seconds"] / 3600,
            timing["median_trial_seconds"] / 3600,
        )


if __name__ == "__main__":
    main()
