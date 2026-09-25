"""Sample the trained NPE's posterior for every test case."""

import logging

import hydra
from omegaconf import DictConfig

from data import save_posterior, stack_posterior_members
from ensemble import num_members, sample_members_by_group
from pipeline import load_case_data, num_obs_groups, setup
from timing import TimingLog, run_labels
from utils import load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def predict_npe(cfg: DictConfig):
    """Run posterior sampling per case and save the draws as NetCDF.

    Ensemble members are sampled separately rather than merged into one mixture, and stored
    one per `chain`, so that downstream comparisons can read either the combined posterior or
    a single member's. `test_num_posterior_samples` is drawn *per member*, which keeps each
    member directly comparable to the MCMC reference of the same size.

    Datasets are sampled one trial count at a time (`pipeline.num_obs_groups`): the empirical
    study's subjects each keep their own number of trials, and the summary network takes one
    rectangular batch. A simulated case is a single group, so this is one call as before.

    Each case's sampling is timed into `timing/predict_npe.csv`: this is the amortized side of
    the cost comparison, measured against the very same test data `fit_mcmc_cpu` is timed on.
    The first case also pays for the sampling function's XLA compilation, and the checkpoint
    load is timed on its own so that neither is hidden inside a per-dataset rate.
    """
    artifacts, cases, param_names = setup(cfg)

    timings = TimingLog(artifacts.csv("timing", "predict_npe"), run_labels())

    with timings.timed("load"):
        approximator, _ = load_approximator(cfg)

    for case in cases:
        logger.info("Loading test data for case %s", case.key)
        blocks = num_obs_groups(load_case_data(case, artifacts))

        logger.info(
            "Sampling %s dataset(s) for case %s, in %s trial-count group(s): %s",
            sum(len(index) for index, _ in blocks),
            case.key,
            len(blocks),
            [int(block["num_obs"]) for _, block in blocks],
        )

        # `num_obs` only where the case has one: study 4's subjects each keep their own trial
        # count, so the case is several blocks and no single number describes the batch. The
        # timing still covers the whole case, since that is the unit the MCMC job is timed over.
        with timings.timed(
            "sample",
            case=case.key,
            num_datasets=sum(len(index) for index, _ in blocks),
            num_obs=int(blocks[0][1]["num_obs"]) if len(blocks) == 1 else None,
            num_draws=cfg["test_num_posterior_samples"],
            num_members=num_members(approximator),
        ):
            posterior_samples = sample_members_by_group(
                approximator,
                blocks,
                num_samples=cfg["test_num_posterior_samples"],
            )

        path = artifacts.npe_samples(case)
        artifacts.ensure(path)
        logger.info("Saving NPE samples for %s member(s) to %s", len(posterior_samples), path)
        save_posterior(path, stack_posterior_members(posterior_samples, param_names), param_names)


if __name__ == "__main__":
    predict_npe()
