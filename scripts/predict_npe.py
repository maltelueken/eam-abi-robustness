"""Sample the trained NPE's posterior for every test case."""

import logging

import hydra
from omegaconf import DictConfig

from data import save_posterior, stack_posterior_members
from ensemble import sample_members_by_group
from pipeline import load_case_data, num_obs_groups, setup
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
    """
    approximator, _ = load_approximator(cfg)

    artifacts, cases, param_names = setup(cfg)

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
